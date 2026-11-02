#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查询扩写数据集工作流（单文件版）

数据流：
    datasets/*.jsonl
      -> 模型一：把原 user query 扩写成 N 个变体
      -> 模型二：对每个变体作答，思维链写入 assistant content
      -> 模型三：判断回答正确性
      -> 过滤 + 格式校验 + 查重
      -> outputs/<数据集名>_expanded.jsonl

中间文件（支持断点续跑）：
    work/queries/<数据集名>.jsonl    扩写结果
    work/answers/<数据集名>.jsonl    模型二回答
    work/judged/<数据集名>.jsonl     模型三判定

用法：
    python workflow.py --config config.yaml all
    python workflow.py --config config.yaml --fresh all
    python workflow.py --config config.yaml --mock --fresh all
    python workflow.py --config config.yaml expand   # 只跑某一阶段

本文件夹完全自包含：workflow.py + common.py + config.yaml + run.sh，
不依赖 self-cognition-WorkFlow 中的文件。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml

import common as C
from common import parse_record


MODEL_KEYS = ("model_expander", "model_answerer", "model_judge")

DEFAULTS: dict[str, Any] = {
    "paths": {
        "data_dir": "../datasets",
        "work_dir": "./work",
        "output_dir": "./outputs",
    },
    "dataset": {
        "dedupe": True,
        "require_reference": True,
        "min_question_chars": 1,
        "max_source_records": 0,
    },
    "expansion": {
        "multiplier": 3,
        "target_total": 0,
        "min_query_chars": 4,
        "max_query_chars": 500,
    },
    "export": {
        "min_correctness": 0.7,
        "require_keep": True,
        "skip_source_duplicates": True,
    },
    "runtime": {
        "expand_workers": 4,
        "answer_workers": 8,
        "judge_workers": 8,
        "max_context_chars": 16000,
        "request_interval_sec": 0.0,
        "fsync_every": 1,
        "progress_interval_sec": 0.2,
        "retry_failed": True,
        "mock_delay_ms": 30,
    },
}

for _key in MODEL_KEYS:
    DEFAULTS[_key] = dict(C.MODEL_DEFAULTS)
DEFAULTS["model_expander"].update({"temperature": 0.9, "max_tokens": 2048})
DEFAULTS["model_answerer"].update({"temperature": 0.7, "max_tokens": 8192})
DEFAULTS["model_judge"].update({"temperature": 0.0, "max_tokens": 2048})

# --mock 不新增配置文件，直接用这组覆盖值做本地全流程验证。
MOCK_OVERRIDES: dict[str, Any] = {
    "paths": {"work_dir": "./work/mock", "output_dir": "./outputs/mock"},
    "dataset": {"max_source_records": 3},
    "expansion": {"multiplier": 2, "target_total": 4},
    "export": {"min_correctness": 0.6},
    "runtime": {
        "expand_workers": 3,
        "answer_workers": 3,
        "judge_workers": 3,
        "mock_delay_ms": 20,
    },
    "model_expander": {"mode": "mock", "name": "mock-expander", "model": "mock-expander", "base_url": ""},
    "model_answerer": {"mode": "mock", "name": "mock-answerer", "model": "mock-answerer", "base_url": ""},
    "model_judge": {"mode": "mock", "name": "mock-judge", "model": "mock-judge", "base_url": ""},
}


# ---------------------------------------------------------------------------
# 配置与路径
# ---------------------------------------------------------------------------

def load_config(path: str | Path, mock: bool = False) -> dict:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise SystemExit(f"错误：配置文件不存在：{config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"错误：YAML 解析失败：{exc}")
    if not isinstance(raw, dict):
        raise SystemExit("错误：配置文件顶层必须是映射（key: value）")

    if mock:
        raw = C.deep_merge(raw, MOCK_OVERRIDES)
    cfg = C.deep_merge(DEFAULTS, raw)
    for key in MODEL_KEYS:
        cfg[key] = C.deep_merge(C.MODEL_DEFAULTS, cfg.get(key) or {})

    base = config_path.resolve().parent
    for key in ("data_dir", "work_dir", "output_dir"):
        value = Path(os.path.expandvars(str(cfg["paths"][key]))).expanduser()
        if not value.is_absolute():
            value = base / value
        cfg["paths"][key] = str(value.resolve())
    cfg["_config_path"] = str(config_path.resolve())
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    for key in MODEL_KEYS:
        model = cfg[key]
        if model.get("mode") not in ("api", "mock"):
            raise SystemExit(f"错误：{key}.mode 只能是 api 或 mock")
        if not str(model.get("model") or "").strip():
            raise SystemExit(f"错误：{key}.model 不能为空")
        if model.get("mode") == "api" and not str(model.get("base_url") or "").strip():
            raise SystemExit(f"错误：{key}.base_url 不能为空（mode=api）")

    multiplier = int(cfg["expansion"].get("multiplier") or 0)
    if not 1 <= multiplier <= 50:
        raise SystemExit("错误：expansion.multiplier 必须在 1~50 之间")
    if int(cfg["expansion"].get("target_total") or 0) < 0:
        raise SystemExit("错误：expansion.target_total 不能小于 0")

    for key in ("expand_workers", "answer_workers", "judge_workers"):
        if int(cfg["runtime"].get(key) or 0) < 1:
            raise SystemExit(f"错误：runtime.{key} 必须 >= 1")
    if int(cfg["runtime"].get("max_context_chars") or 0) < 500:
        raise SystemExit("错误：runtime.max_context_chars 必须 >= 500")

    if not 0 <= float(cfg["export"].get("min_correctness", 0.7)) <= 1:
        raise SystemExit("错误：export.min_correctness 必须在 0~1 之间")
    if int(cfg["dataset"].get("max_source_records") or 0) < 0:
        raise SystemExit("错误：dataset.max_source_records 不能小于 0")

    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    for key in ("work_dir", "output_dir"):
        if Path(cfg["paths"][key]).resolve() == data_dir:
            raise SystemExit(f"错误：paths.{key} 不能与 data_dir 相同")


def stage_dir(cfg: dict, kind: str) -> Path:
    return Path(cfg["paths"]["work_dir"]) / kind


def dataset_paths(cfg: dict, domain: str) -> dict[str, Path]:
    stem = C.dataset_stem(domain)
    return {
        "queries": stage_dir(cfg, "queries") / f"{stem}.jsonl",
        "answers": stage_dir(cfg, "answers") / f"{stem}.jsonl",
        "judged": stage_dir(cfg, "judged") / f"{stem}.jsonl",
        "final": Path(cfg["paths"]["output_dir"]) / f"{stem}_expanded.jsonl",
    }


def discover_files(cfg: dict) -> list[Path]:
    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"错误：数据集目录不存在：{data_dir}")

    excluded = [
        Path(cfg["paths"]["work_dir"]).resolve(),
        Path(cfg["paths"]["output_dir"]).resolve(),
        data_dir / "outputs",
    ]
    for name in ("work", "outputs"):
        excluded.append((data_dir / name).resolve())

    files: list[Path] = []
    for path in sorted(data_dir.rglob("*.jsonl")):
        resolved = path.resolve()
        if not resolved.is_file():
            continue
        if any(resolved == item or resolved.is_relative_to(item) for item in excluded):
            continue
        if resolved.name.endswith(C.OUTPUT_SUFFIXES) or resolved.name.endswith(
            ("_expanded.jsonl", "_queries.jsonl", "_answers.jsonl", "_judged.jsonl")
        ):
            continue
        if any(part.startswith(".") for part in resolved.relative_to(data_dir).parts):
            continue
        files.append(resolved)
    return files


def load_sources(cfg: dict) -> tuple[dict[str, list[dict]], dict]:
    """读取原始数据集，抽取最后一条 user 与 assistant 参考答案。"""
    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    files = discover_files(cfg)
    dataset = cfg["dataset"]
    dedupe = bool(dataset.get("dedupe", True))
    min_question = max(1, int(dataset.get("min_question_chars") or 1))
    require_reference = bool(dataset.get("require_reference", True))
    max_sources = max(0, int(dataset.get("max_source_records") or 0))

    datasets: dict[str, list[dict]] = {}
    file_stats: list[dict] = []
    used_domains: set[str] = set()
    used_keys: set[str] = set()
    total_kept = 0
    truncated = False

    for path in files:
        relative = path.relative_to(data_dir)
        domain = C.dataset_stem("__".join(relative.with_suffix("").parts))
        if domain in used_domains:
            domain = f"{domain}__{C.sha1_text(str(path))[:6]}"
        used_domains.add(domain)

        lines = bad = invalid = duplicates = no_ref = kept = 0
        seen_file: set[str] = set()
        records: list[dict] = []
        try:
            handle = open(path, "r", encoding="utf-8")
        except OSError as exc:
            C.log(f"  跳过无法读取的文件：{relative.as_posix()}（{exc}）")
            continue

        with handle:
            for line_no, raw_line in enumerate(handle, 1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                lines += 1
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not isinstance(record, dict):
                    bad += 1
                    continue
                try:
                    parsed, _ = parse_record(record, multi_turn=False, do_strip=True)
                except ValueError:
                    invalid += 1
                    continue

                question = str(parsed.get("question") or "").strip()
                reference = str(parsed.get("reference_answer") or "").strip()
                if len(question) < min_question:
                    invalid += 1
                    continue
                if require_reference and not reference:
                    no_ref += 1
                    continue

                if dedupe:
                    key = C.normalize_for_key(question)
                    if key in seen_file:
                        duplicates += 1
                        continue
                    seen_file.add(key)

                if max_sources and total_kept >= max_sources:
                    truncated = True
                    break

                meta = {key: value for key, value in record.items() if key != "messages"}
                source_id = str(record.get("id") or "").strip() or f"q_{C.sha1_text(question)[:12]}"
                source_key = f"{domain}#{source_id}"
                if source_key in used_keys:
                    source_key = f"{source_key}#{C.sha1_text(question)[:6]}"
                used_keys.add(source_key)
                system_prompt = "\n\n".join(
                    str(item.get("content") or "")
                    for item in parsed.get("prompt_messages") or []
                    if item.get("role") == "system" and str(item.get("content") or "").strip()
                )
                records.append(
                    {
                        "key": source_key,
                        "domain": domain,
                        "source_id": source_id,
                        "question": question,
                        "reference_answer": reference,
                        "system_prompt": system_prompt,
                        "category": meta.get("category") or "",
                        "multiturn": bool(meta.get("multiturn", False)),
                        "tool": bool(meta.get("tool", False)),
                        "meta": meta,
                        "source": {"file": relative.as_posix(), "line": line_no},
                    }
                )
                kept += 1
                total_kept += 1

        if kept:
            datasets.setdefault(domain, []).extend(records)
        file_stats.append(
            {
                "file": relative.as_posix(),
                "domain": domain,
                "lines": lines,
                "kept": kept,
                "duplicates": duplicates,
                "invalid": invalid,
                "no_reference": no_ref,
                "bad_lines": bad,
            }
        )
        if truncated:
            C.log(f"  已达到 max_source_records={max_sources}，停止读取后续文件")
            break

    stats = {
        "files": len(files),
        "lines": sum(item["lines"] for item in file_stats),
        "kept": total_kept,
        "duplicates": sum(item["duplicates"] for item in file_stats),
        "invalid": sum(item["invalid"] for item in file_stats),
        "no_reference": sum(item["no_reference"] for item in file_stats),
        "bad_lines": sum(item["bad_lines"] for item in file_stats),
        "truncated": truncated,
        "file_stats": file_stats,
    }
    return datasets, stats


def load_stage(cfg: dict, kind: str) -> dict[str, dict[str, dict]]:
    directory = stage_dir(cfg, kind)
    result: dict[str, dict[str, dict]] = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("*.jsonl")):
        rows, bad = C.read_jsonl(path)
        if bad:
            C.log(f"  警告：{path.name} 有 {bad} 行无法解析，已忽略")
        last: dict[str, dict] = {}
        for row in rows:
            key = str(row.get("key") or "")
            if key:
                last[key] = row
        if last:
            result[path.stem] = last
    return result


def reset_generated(cfg: dict, stage: str) -> None:
    """--fresh 时按阶段清空本工作流生成的中间文件和最终数据集。"""
    remove_stages: tuple[str, ...] = ()
    if stage in ("all", "expand"):
        remove_stages = ("queries", "answers", "judged")
    elif stage == "answer":
        remove_stages = ("answers", "judged")
    elif stage == "judge":
        remove_stages = ("judged",)
    if stage != "export":
        for kind in remove_stages:
            directory = stage_dir(cfg, kind)
            if directory.is_dir():
                shutil.rmtree(directory)
                C.log(f"  已清空中间目录：{directory}")
        output_dir = Path(cfg["paths"]["output_dir"])
        if output_dir.is_dir():
            for path in output_dir.glob("*_expanded.jsonl"):
                path.unlink()
                C.log(f"  已删除旧数据集：{path.name}")


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

def build_expand_messages(cfg: dict, source: dict, count: int) -> list[dict]:
    max_chars = int(cfg["runtime"].get("max_context_chars") or 16000)
    reference = C.clip(source.get("reference_answer") or "", max(800, max_chars // 3))
    system = """你是一名严谨的问答数据扩写专家。
你的任务不是回答问题，而是把给定的原问题改写成若干条语义等价、表达不同的独立用户提问。

要求：
1. 保持原问题的核心意图、事实约束、限定条件和语言（中文问题用中文改写，英文问题用英文改写）。
2. 每条改写都必须是完整、可独立作答的用户问题，不能依赖上下文。
3. 不得改变原问题的结论方向，不得加入参考答案中的答案内容，不得给出答案。
4. 不同改写之间要有明显措辞或视角差异，避免只替换个别同义词。
5. 不要输出编号、解释、Markdown 代码块或多余文字。

只输出一个 JSON 对象：
{"queries": ["第一条改写", "第二条改写"]}"""
    user = f"""【原问题】
{source.get('question', '')}

【参考答案（仅用于理解知识点，不要复制进问题）】
{reference or '（无）'}

【需要生成的数量】
{count}

请生成 {count} 条改写，并只输出 JSON。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_answer_messages(cfg: dict, query_record: dict, expanded_query: str) -> list[dict]:
    messages: list[dict] = []
    system_prompt = str(query_record.get("system_prompt") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": expanded_query})
    return messages


def build_correctness_messages(cfg: dict, answer_record: dict) -> list[dict]:
    threshold = float(cfg["export"].get("min_correctness", 0.7))
    max_chars = int(cfg["runtime"].get("max_context_chars") or 16000)
    budget = max(1500, max_chars - 1600)

    question = C.clip(answer_record.get("original_question") or "", max(500, budget // 5))
    expanded = C.clip(answer_record.get("expanded_query") or "", max(500, budget // 5))
    reference = C.clip(answer_record.get("reference_answer") or "（无参考答案）", max(800, budget // 3))
    final_answer = C.clip(answer_record.get("final_answer") or "（无最终回答）", max(800, budget // 3))
    reasoning = C.clip(answer_record.get("reasoning") or "", max(500, budget // 4))

    system = f"""你是一名严格、公正的问答正确性评审专家。
给定原问题、参考答案、扩写后的问题和模型回答，请判断模型回答是否正确。

# 判定要求
- 以参考答案和客观事实为准；语义等价但表述不同，视为正确。
- 回答正确且包含合理的额外信息，不扣分。
- 出现事实错误、关键结论错误、与参考答案冲突，应显著扣分。
- 回答是否真正回应了“扩写后的问题”，也要考虑；答非所问应低分。
- 思维链仅作为判断辅助，不因思维链风格、长度扣分；以最终回答为准。
- 最终回答为空、拒答、仅复述问题：correctness 记 0。
- 参考答案缺失时，依据客观事实判断。
- correctness 范围 0.00~1.00，保留两位小数；keep 表示是否达到通过阈值 {threshold:.2f}。

# 输出格式
只输出一个 JSON 对象，不要输出解释文字或 Markdown 代码块：
{{"correctness": 0.85, "keep": true, "reason": "<不超过 120 字的判断理由>"}}"""

    user = f"""【原问题】
{question}

【参考答案】
{reference}

【扩写后的问题】
{expanded}

【模型最终回答】
{final_answer}

【模型思维链（仅供参考）】
{reasoning or '（无）'}

请只输出 JSON 评审结果。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_expansion(raw: str, count: int, min_chars: int, max_chars: int) -> list[str]:
    candidates: list[str] = []
    obj = C.extract_json_obj(raw)
    if isinstance(obj, dict):
        for key in ("queries", "questions", "variants", "rewrites", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                candidates = [str(item) for item in value if isinstance(item, (str, int, float))]
                break
    if not candidates:
        try:
            value = json.loads(raw.strip())
            if isinstance(value, list):
                candidates = [str(item) for item in value if isinstance(item, (str, int, float))]
        except (json.JSONDecodeError, TypeError):
            pass
    if not candidates:
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("{"):
                continue
            candidates.append(line)

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = str(item).strip().strip('"').strip("'").strip()
        text = re.sub(r"^\s*(?:[-*•]|\d+[\.\)、])\s*", "", text)
        text = re.sub(r"^(?:问题|改写|扩写|变体)\s*[:：]\s*", "", text)
        text = text.strip()
        if len(text) < min_chars or len(text) > max_chars:
            continue
        key = C.normalize_for_key(text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= count:
            break
    return cleaned


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace("%", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "y", "1", "是", "通过", "保留"):
            return True
        if text in ("false", "no", "n", "0", "否", "不通过", "丢弃"):
            return False
    return None


def parse_correctness(raw: str, threshold: float) -> tuple[float | None, bool | None, str, str | None]:
    obj = C.extract_json_obj(raw)
    if not isinstance(obj, dict):
        return None, None, "", "无法从评分模型输出中解析出 JSON"
    lowered = {str(key).lower(): value for key, value in obj.items()}

    score = None
    for key in ("correctness", "score", "accuracy", "correct"):
        score = _to_float(lowered.get(key))
        if score is not None:
            break
    if score is None:
        return None, None, "", "评分结果缺少 correctness 字段"
    if score > 1:
        score = score / 10.0 if score <= 10 else score / 100.0
    score = round(max(0.0, min(1.0, score)), 4)

    keep = _to_bool(lowered.get("keep"))
    if keep is None:
        for key in ("valid", "accept"):
            keep = _to_bool(lowered.get(key))
            if keep is not None:
                break
    if keep is None:
        keep = score >= threshold

    reason = ""
    for key in ("reason", "reasons", "explanation", "comment", "rationale"):
        if lowered.get(key):
            reason = str(lowered[key]).strip()
            break
    return score, keep, reason, None


# ---------------------------------------------------------------------------
# 模型调用（保留 reasoning，不清洗 content）
# ---------------------------------------------------------------------------

class RawCallError(RuntimeError):
    pass


def _api_key(model_cfg: dict) -> str:
    key = str(model_cfg.get("api_key") or "").strip()
    if key:
        return key
    env_name = str(model_cfg.get("api_key_env") or "").strip()
    return str(os.environ.get(env_name, "")).strip() if env_name else ""


def _content_keep_cot(value: Any) -> str:
    """兼容 str / 多模态 list；保留 thinking 块并转成 <think>。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                kind = str(part.get("type") or "").lower()
                text = part.get("text")
                if text is None:
                    text = part.get("thinking")
                if text is None:
                    text = part.get("content")
                if not isinstance(text, str):
                    continue
                if kind in ("thinking", "reasoning", "analysis") and not C.has_cot(text):
                    text = f"<think>\n{text}\n</think>"
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        return _content_keep_cot([value])
    return str(value)


def _extract_raw(data: dict) -> tuple[str, str, str]:
    choices = data.get("choices") or []
    if not choices:
        return "", "", ""
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = _content_keep_cot(message.get("content"))
    if not content.strip():
        content = _content_keep_cot(choice.get("text"))
    reasoning = _content_keep_cot(message.get("reasoning_content") or message.get("reasoning") or "")
    return content, reasoning, str(choice.get("finish_reason") or "")


def chat_raw(model_cfg: dict, messages: list[dict], purpose: str, runtime: dict, label: str = "model") -> dict:
    max_chars = int(runtime.get("max_context_chars") or 16000)
    msgs = C.fit_messages(messages, max_chars)
    if model_cfg.get("mode") == "mock":
        return _mock_chat(model_cfg, msgs, purpose, runtime)

    url = C.resolve_chat_url(model_cfg)
    headers = {"Content-Type": "application/json"}
    key = _api_key(model_cfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload: dict[str, Any] = {
        "model": model_cfg.get("model"),
        "messages": msgs,
        "max_tokens": int(model_cfg.get("max_tokens") or 2048),
        "stream": False,
    }
    if model_cfg.get("temperature") is not None:
        payload["temperature"] = model_cfg["temperature"]
    if model_cfg.get("top_p") is not None:
        payload["top_p"] = model_cfg["top_p"]
    payload.update(model_cfg.get("extra_body") or {})
    for drop_key in model_cfg.get("drop_keys") or []:
        payload.pop(drop_key, None)

    timeout = float(model_cfg.get("timeout") or 120)
    retries = max(0, int(model_cfg.get("max_retries") or 0))
    interval = float(runtime.get("request_interval_sec") or 0)
    last_err = "未知错误"

    for attempt in range(retries + 1):
        C._throttle(interval)
        started = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_err = f"网络异常: {exc}"
        else:
            status = resp.status_code
            if status < 400:
                try:
                    data = resp.json()
                except ValueError:
                    last_err = f"响应非 JSON: {resp.text[:200]}"
                else:
                    content, reasoning, finish_reason = _extract_raw(data)
                    if not content.strip() and not reasoning.strip():
                        last_err = (
                            "响应中没有 content/reasoning"
                            f"（finish_reason={finish_reason}），可能需要调大 max_tokens"
                        )
                    else:
                        return {
                            "content": content,
                            "reasoning": reasoning,
                            "usage": data.get("usage") or {},
                            "latency_ms": int((time.time() - started) * 1000),
                            "finish_reason": finish_reason,
                        }
            elif status == 429 or status >= 500 or status in (408, 409, 425):
                last_err = f"HTTP {status}: {resp.text[:200]}"
            else:
                raise RawCallError(f"HTTP {status}: {resp.text[:500]}")

        if attempt < retries:
            wait = min(2 ** attempt, 15) + random.random() * 0.5
            C.log(f"    {label} 调用失败（{last_err}），{wait:.1f}s 后重试 {attempt + 1}/{retries}")
            time.sleep(wait)

    raise RawCallError(f"{label} 调用失败（已重试 {retries} 次）：{last_err}")


def merge_assistant_content(content: str, reasoning: str) -> tuple[str, str]:
    """reasoning 字段与 content 内思维链统一收进 assistant content。

    返回 (assistant_content, final_answer)。
    """
    content = (content or "").strip()
    reasoning = (reasoning or "").strip()
    final_answer = C.strip_cot(content).strip()

    if reasoning and not C.has_cot(content):
        block = reasoning if C.has_cot(reasoning) else f"<think>\n{reasoning}\n</think>"
        content = f"{block}\n\n{content}".strip() if content else block
    return content, final_answer


def extract_cot_text(content: str) -> str:
    """当思维链混在 content 中时，提取出来供评审模型参考。"""
    pattern = r"<(think|thinking|analysis|reasoning)\b[^>]*>(.*?)</\1>"
    parts = [match.group(2).strip() for match in re.finditer(pattern, content or "", re.S | re.I)]
    return "\n\n".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# mock 模式
# ---------------------------------------------------------------------------

def _mock_expand(prompt: str) -> str:
    match = re.search(r"【原问题】\n(.*?)(?:\n\n【参考答案|\n\n【需要生成的数量】)", prompt, re.S)
    question = match.group(1).strip() if match else prompt.strip()
    count_match = re.search(r"【需要生成的数量】\s*(\d+)", prompt)
    count = int(count_match.group(1)) if count_match else 1

    templates = [
        "请从基本原理出发，解释：{q}",
        "关于「{q}」，请给出一个准确、简洁的回答。",
        "如果从工程应用的角度看，{q} 应该如何理解？",
        "请说明这个问题的关键要点：{q}",
        "换一种说法：{q}，请给出你的结论。",
    ]
    variants = [templates[index % len(templates)].format(q=question) for index in range(count)]
    if count >= 2 and int(C.sha1_text(question)[:2], 16) % 3 == 0:
        variants[-1] = variants[0]  # 故意制造重复，验证扩写去重
    return json.dumps({"queries": variants}, ensure_ascii=False)


def _mock_answer(prompt: str) -> tuple[str, str]:
    query = prompt.strip()
    digest = C.sha1_text("mock-answer", query)
    reasoning = f"模拟推理：需要围绕「{C.clip(query, 50)}」拆解概念、前提和结论，注意不要脱离问题范围。"
    final = (
        "（模拟回答）这个问题可以从定义、关键参数和应用场景三个方面理解。"
        f"核心结论需要结合具体条件判断。问题摘要：{C.clip(query, 60)}"
    )
    if int(digest[:2], 16) % 2 == 0:
        return f"<think>\n{reasoning}\n</think>\n\n{final}", ""
    return final, reasoning


def _mock_judge(prompt: str) -> str:
    match = re.search(
        r"【原问题】\n.*?\n\n【参考答案】\n.*?\n\n【扩写后的问题】\n(.*?)\n\n【模型最终回答】\n(.*?)\n\n【模型思维链",
        prompt,
        re.S,
    )
    expanded = match.group(1).strip() if match else prompt
    final = match.group(2).strip() if match else ""
    threshold_match = re.search(r"通过阈值\s*([0-9.]+)", prompt)
    threshold = float(threshold_match.group(1)) if threshold_match else 0.7

    if not final or final.startswith("（无最终回答）"):
        return json.dumps({"correctness": 0.0, "keep": False, "reason": "没有最终回答。"}, ensure_ascii=False)

    digest = C.sha1_text("mock-correctness", expanded, final)
    score = round(0.35 + (int(digest[:6], 16) % 601) / 1000.0, 2)
    keep = score >= threshold
    reason = "模拟评审：回答覆盖了基本要点。" if keep else "模拟评审：回答与参考答案的要点对应不足。"
    return json.dumps({"correctness": score, "keep": keep, "reason": reason}, ensure_ascii=False)


def _mock_chat(model_cfg: dict, messages: list[dict], purpose: str, runtime: dict) -> dict:
    delay = max(0.0, float(runtime.get("mock_delay_ms") or 0)) / 1000.0
    if delay:
        time.sleep(random.uniform(delay * 0.5, delay * 1.5))
    prompt = messages[-1]["content"] if messages else ""
    content = ""
    reasoning = ""
    if purpose == "expand":
        content = _mock_expand(prompt)
    elif purpose == "answer":
        content, reasoning = _mock_answer(prompt)
    elif purpose == "judge":
        content = _mock_judge(prompt)
    else:
        content = f"（mock）{purpose}: {C.clip(prompt, 60)}"
    return {
        "content": content,
        "reasoning": reasoning,
        "usage": {"prompt_tokens": len(prompt) // 3, "completion_tokens": len(content) // 3},
        "latency_ms": int(delay * 1000),
        "finish_reason": "stop",
        "mock": True,
    }


# ---------------------------------------------------------------------------
# 哈希与计划
# ---------------------------------------------------------------------------

def _model_fingerprint(model_cfg: dict) -> dict:
    return {
        "model": model_cfg.get("model"),
        "base_url": model_cfg.get("base_url"),
        "temperature": model_cfg.get("temperature"),
        "top_p": model_cfg.get("top_p"),
        "max_tokens": model_cfg.get("max_tokens"),
        "extra_body": model_cfg.get("extra_body"),
    }


def expand_hash(cfg: dict, source: dict, count: int) -> str:
    return C.sha1_text(
        json.dumps(
            {
                "model": _model_fingerprint(cfg["model_expander"]),
                "question": source.get("question"),
                "reference": source.get("reference_answer"),
                "count": count,
                "prompt_version": 1,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def answer_hash(cfg: dict, query_record: dict, expanded_query: str) -> str:
    return C.sha1_text(
        json.dumps(
            {
                "model": _model_fingerprint(cfg["model_answerer"]),
                "query": expanded_query,
                "system": query_record.get("system_prompt") or "",
                "prompt_version": 1,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def judge_hash(cfg: dict, answer_record: dict) -> str:
    return C.sha1_text(
        json.dumps(
            {
                "model": _model_fingerprint(cfg["model_judge"]),
                "expanded": answer_record.get("expanded_query"),
                "final": answer_record.get("final_answer"),
                "reasoning": answer_record.get("reasoning"),
                "reference": answer_record.get("reference_answer"),
                "threshold": cfg["export"].get("min_correctness"),
                "prompt_version": 1,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def plan_expansions(sources: list[dict], multiplier: int, target_total: int) -> dict[str, int]:
    plan: dict[str, int] = {}
    remaining = target_total if target_total > 0 else None
    for source in sources:
        count = multiplier
        if remaining is not None:
            if remaining <= 0:
                break
            count = min(count, remaining)
            remaining -= count
        plan[source["key"]] = count
    return plan


# ---------------------------------------------------------------------------
# 阶段 1：扩写
# ---------------------------------------------------------------------------

def stage_expand(cfg: dict, fresh: bool) -> int:
    datasets, stats = load_sources(cfg)
    if not datasets:
        C.log("错误：没有解析到可扩写的源问题")
        return 2

    sources = [source for records in datasets.values() for source in records]
    multiplier = int(cfg["expansion"]["multiplier"])
    target_total = int(cfg["expansion"].get("target_total") or 0)
    plan = plan_expansions(sources, multiplier, target_total)
    if not plan:
        C.log("错误：扩写计划为空")
        return 2

    C.log(
        f"源数据：{stats['files']} 个文件，{stats['lines']} 行，保留 {stats['kept']} 个问题"
        f"（去重 {stats['duplicates']}，无效 {stats['invalid']}，无参考答案 {stats['no_reference']}）"
    )
    C.log(
        f"扩写计划：倍数 {multiplier}，目标总量 {target_total or '不限'}，"
        f"计划生成 {sum(plan.values())} 条"
    )
    if stats["truncated"]:
        C.log("  注意：已按 max_source_records 截断源问题")

    total_planned = sum(plan.values())
    total_actual = 0
    domain_results = []
    runtime = cfg["runtime"]

    for domain in datasets:
        domain_sources = [source for source in datasets[domain] if plan.get(source["key"], 0) > 0]
        if not domain_sources:
            continue
        paths = dataset_paths(cfg, domain)
        store = C.JsonlStore(paths["queries"], fsync_every=int(runtime.get("fsync_every") or 1))
        if fresh:
            store.truncate()

        existing, bad_lines, valid_lines = store.load_last()
        if bad_lines:
            C.log(f"  {domain}：扩写文件有 {bad_lines} 行无法解析，已忽略")
        duplicates = max(0, valid_lines - len(existing))
        if duplicates:
            store.compact(existing.values())

        pending: list[tuple[dict, int, str]] = []
        reused = skipped = replaced = 0
        for source in domain_sources:
            count = plan[source["key"]]
            input_hash = expand_hash(cfg, source, count)
            old = existing.get(source["key"])
            if old and old.get("input_hash") == input_hash and old.get("ok"):
                reused += 1
                continue
            if old and not old.get("ok") and not bool(runtime.get("retry_failed", True)):
                skipped += 1
                continue
            if old:
                replaced += 1
            pending.append((source, count, input_hash))

        C.log(f"领域 {domain}：计划 {len(domain_sources)} 个问题 | 待扩写 {len(pending)} | 已有 {reused}")
        if not pending:
            total_actual += sum(len(existing.get(source["key"], {}).get("variants") or []) for source in domain_sources)
            domain_results.append({"domain": domain, "planned": len(domain_sources), "ok": reused, "variants": 0})
            continue

        workers = max(1, min(int(runtime.get("expand_workers") or 1), len(pending)))
        progress = C.Progress(
            domain,
            len(domain_sources),
            initial_done=len(domain_sources) - len(pending),
            initial_ok=reused,
            initial_fail=skipped,
            interval=float(runtime.get("progress_interval_sec") or 0.2),
        )

        def run_one(source: dict, count: int, input_hash: str) -> dict:
            variants: list[str] = []
            error = None
            try:
                messages = build_expand_messages(cfg, source, count)
                result = chat_raw(cfg["model_expander"], messages, "expand", runtime, label="扩写模型")
                variants = parse_expansion(
                    result["content"],
                    count,
                    int(cfg["expansion"].get("min_query_chars") or 4),
                    int(cfg["expansion"].get("max_query_chars") or 500),
                )
                if not variants:
                    error = "扩写结果为空或无法解析"
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

            return {
                "key": source["key"],
                "domain": source["domain"],
                "source_id": source["source_id"],
                "original_question": source["question"],
                "reference_answer": source["reference_answer"],
                "system_prompt": source.get("system_prompt") or "",
                "category": source.get("category") or "",
                "multiturn": bool(source.get("multiturn", False)),
                "tool": bool(source.get("tool", False)),
                "meta": source.get("meta") or {},
                "source": source.get("source") or {},
                "variants": variants,
                "variant_count": len(variants),
                "requested": count,
                "warning": None if len(variants) >= count else f"请求 {count} 条，实际 {len(variants)} 条",
                "model": cfg["model_expander"].get("model"),
                "mode": cfg["model_expander"].get("mode"),
                "ok": bool(variants),
                "error": error,
                "input_hash": input_hash,
                "created_at": C.now_iso(),
            }

        from concurrent.futures import ThreadPoolExecutor, as_completed

        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {executor.submit(run_one, source, count, h): source for source, count, h in pending}
        interrupted = False
        try:
            for future in as_completed(futures):
                record = future.result()
                store.append(record)
                if not record["ok"]:
                    C.log(f"  扩写失败：{record['key']} -> {record['error']}")
                elif record["warning"]:
                    C.log(f"  提示：{record['key']} {record['warning']}")
                progress.update(ok=bool(record["ok"]))
        except KeyboardInterrupt:
            interrupted = True
            C.log("收到中断信号：已完成扩写已保存，重新运行即可续跑")
            for item in futures:
                item.cancel()
        finally:
            progress.close()
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

        if interrupted:
            return 130
        final, _, _ = store.load_last()
        if replaced:
            store.compact(final.values())

        completed = 0
        variants_count = 0
        for source in domain_sources:
            item = final.get(source["key"])
            if item and item.get("ok") and item.get("input_hash") == expand_hash(cfg, source, plan[source["key"]]):
                completed += 1
                variants_count += len(item.get("variants") or [])
        total_actual += variants_count
        domain_results.append({"domain": domain, "planned": len(domain_sources), "ok": completed, "variants": variants_count})
        C.log(f"领域 {domain}：完成 {completed}/{len(domain_sources)}，生成 {variants_count} 条变体")

    incomplete = [item for item in domain_results if item["ok"] < item["planned"]]
    C.log(f"阶段 1 汇总：计划 {total_planned} 条，实际保存 {total_actual} 条变体")
    if incomplete:
        C.log(f"仍有 {len(incomplete)} 个领域未完成扩写，请重新运行续跑")
        return 3
    return 0


# ---------------------------------------------------------------------------
# 阶段 2：模型二回答
# ---------------------------------------------------------------------------

def stage_answer(cfg: dict, fresh: bool) -> int:
    queries = load_stage(cfg, "queries")
    if not queries:
        C.log("错误：没有扩写结果，请先运行 expand")
        return 2

    runtime = cfg["runtime"]
    domain_results = []
    total_tasks = 0
    total_ok = 0

    for domain in queries:
        query_records = list(queries[domain].values())
        tasks: list[tuple[dict, int, str, str]] = []
        for query_record in query_records:
            if not query_record.get("ok"):
                continue
            for index, query in enumerate(query_record.get("variants") or [], 1):
                key = f"{query_record['key']}#v{index:03d}"
                input_hash = answer_hash(cfg, query_record, query)
                tasks.append((query_record, index, query, input_hash))
        if not tasks:
            C.log(f"领域 {domain}：没有可回答的扩写问题")
            continue

        paths = dataset_paths(cfg, domain)
        store = C.JsonlStore(paths["answers"], fsync_every=int(runtime.get("fsync_every") or 1))
        if fresh:
            store.truncate()

        existing, bad_lines, valid_lines = store.load_last()
        if bad_lines:
            C.log(f"  {domain}：回答文件有 {bad_lines} 行无法解析，已忽略")
        duplicates = max(0, valid_lines - len(existing))
        if duplicates:
            store.compact(existing.values())

        pending: list[tuple[dict, int, str, str]] = []
        reused = skipped = replaced = 0
        for query_record, index, query, input_hash in tasks:
            key = f"{query_record['key']}#v{index:03d}"
            old = existing.get(key)
            if old and old.get("input_hash") == input_hash and old.get("ok"):
                reused += 1
                continue
            if old and not old.get("ok") and not bool(runtime.get("retry_failed", True)):
                skipped += 1
                continue
            if old:
                replaced += 1
            pending.append((query_record, index, query, input_hash))

        total_tasks += len(tasks)
        C.log(f"领域 {domain}：扩写 {len(tasks)} 条 | 待回答 {len(pending)} | 已有 {reused}")
        if not pending:
            total_ok += reused
            domain_results.append({"domain": domain, "tasks": len(tasks), "ok": reused, "written": 0})
            continue

        workers = max(1, min(int(runtime.get("answer_workers") or 1), len(pending)))
        progress = C.Progress(
            domain,
            len(tasks),
            initial_done=len(tasks) - len(pending),
            initial_ok=reused,
            initial_fail=skipped,
            interval=float(runtime.get("progress_interval_sec") or 0.2),
        )

        def run_one(query_record: dict, index: int, query: str, input_hash: str) -> dict:
            key = f"{query_record['key']}#v{index:03d}"
            assistant_content = ""
            final_answer = ""
            reasoning = ""
            error = None
            result: dict | None = None
            try:
                messages = build_answer_messages(cfg, query_record, query)
                result = chat_raw(cfg["model_answerer"], messages, "answer", runtime, label="回答模型")
                result_reasoning = result.get("reasoning") or ""
                assistant_content, final_answer = merge_assistant_content(
                    result.get("content") or "", result_reasoning
                )
                reasoning = result_reasoning or extract_cot_text(assistant_content)
                if not final_answer:
                    error = "模型只返回思维链或空回答，没有最终回答"
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

            return {
                "key": key,
                "domain": query_record.get("domain"),
                "source_id": query_record.get("source_id"),
                "source_key": query_record.get("key"),
                "original_question": query_record.get("original_question"),
                "reference_answer": query_record.get("reference_answer"),
                "system_prompt": query_record.get("system_prompt") or "",
                "category": query_record.get("category") or "",
                "multiturn": bool(query_record.get("multiturn", False)),
                "tool": bool(query_record.get("tool", False)),
                "source": query_record.get("source") or {},
                "expanded_query": query,
                "variant_index": index,
                "assistant_content": assistant_content,
                "final_answer": final_answer,
                "reasoning": reasoning,
                "ok": bool(final_answer),
                "error": error,
                "input_hash": input_hash,
                "expander_model": query_record.get("model"),
                "model": cfg["model_answerer"].get("model"),
                "mode": cfg["model_answerer"].get("mode"),
                "latency_ms": (result or {}).get("latency_ms"),
                "usage": (result or {}).get("usage") or {},
                "created_at": C.now_iso(),
            }

        from concurrent.futures import ThreadPoolExecutor, as_completed

        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {executor.submit(run_one, q, i, text, h): text for q, i, text, h in pending}
        interrupted = False
        try:
            for future in as_completed(futures):
                record = future.result()
                store.append(record)
                if not record["ok"]:
                    C.log(f"  回答失败：{record['key']} -> {record['error']}")
                progress.update(ok=bool(record["ok"]))
        except KeyboardInterrupt:
            interrupted = True
            C.log("收到中断信号：已完成回答已保存，重新运行即可续跑")
            for item in futures:
                item.cancel()
        finally:
            progress.close()
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

        if interrupted:
            return 130
        final, _, _ = store.load_last()
        if replaced:
            store.compact(final.values())

        completed = 0
        for query_record, index, query, _ in tasks:
            key = f"{query_record['key']}#v{index:03d}"
            item = final.get(key)
            if item and item.get("ok") and item.get("input_hash") == answer_hash(cfg, query_record, query):
                completed += 1
        total_ok += completed
        domain_results.append({"domain": domain, "tasks": len(tasks), "ok": completed, "written": len(pending)})
        C.log(f"领域 {domain}：完成 {completed}/{len(tasks)}")

    incomplete = [item for item in domain_results if item["ok"] < item["tasks"]]
    C.log(f"阶段 2 汇总：回答 {total_ok}/{total_tasks}")
    if incomplete:
        C.log(f"仍有 {len(incomplete)} 个领域未完成回答，请重新运行续跑")
        return 3
    return 0


# ---------------------------------------------------------------------------
# 阶段 3：模型三正确性判断
# ---------------------------------------------------------------------------

def stage_judge(cfg: dict, fresh: bool) -> int:
    answers = load_stage(cfg, "answers")
    if not answers:
        C.log("错误：没有回答结果，请先运行 answer")
        return 2

    runtime = cfg["runtime"]
    threshold = float(cfg["export"].get("min_correctness", 0.7))
    domain_results = []
    total_tasks = 0
    total_ok = 0
    total_keep = 0

    for domain in answers:
        answer_records = list(answers[domain].values())
        tasks = [record for record in answer_records if record.get("ok") and record.get("final_answer")]
        if not tasks:
            C.log(f"领域 {domain}：没有可评判的回答")
            continue

        paths = dataset_paths(cfg, domain)
        store = C.JsonlStore(paths["judged"], fsync_every=int(runtime.get("fsync_every") or 1))
        if fresh:
            store.truncate()

        existing, bad_lines, valid_lines = store.load_last()
        if bad_lines:
            C.log(f"  {domain}：判定文件有 {bad_lines} 行无法解析，已忽略")
        duplicates = max(0, valid_lines - len(existing))
        if duplicates:
            store.compact(existing.values())

        pending: list[tuple[dict, str]] = []
        reused = skipped = replaced = 0
        for answer_record in tasks:
            input_hash = judge_hash(cfg, answer_record)
            old = existing.get(answer_record["key"])
            if old and old.get("input_hash") == input_hash and old.get("ok"):
                reused += 1
                continue
            if old and not old.get("ok") and not bool(runtime.get("retry_failed", True)):
                skipped += 1
                continue
            if old:
                replaced += 1
            pending.append((answer_record, input_hash))

        total_tasks += len(tasks)
        C.log(f"领域 {domain}：答案 {len(tasks)} 条 | 待评判 {len(pending)} | 已有 {reused}")
        if not pending:
            total_ok += reused
            task_keys = {record["key"] for record in tasks}
            total_keep += sum(
                1
                for key, record in existing.items()
                if key in task_keys and record.get("ok") and record.get("keep")
            )
            domain_results.append({"domain": domain, "tasks": len(tasks), "ok": reused, "keep": 0})
            continue

        workers = max(1, min(int(runtime.get("judge_workers") or 1), len(pending)))
        progress = C.Progress(
            domain,
            len(tasks),
            initial_done=len(tasks) - len(pending),
            initial_ok=reused,
            initial_fail=skipped,
            interval=float(runtime.get("progress_interval_sec") or 0.2),
        )

        def run_one(answer_record: dict, input_hash: str) -> dict:
            correctness = None
            keep = None
            reason = ""
            error = None
            result: dict | None = None
            try:
                messages = build_correctness_messages(cfg, answer_record)
                result = chat_raw(cfg["model_judge"], messages, "judge", runtime, label="评审模型")
                correctness, keep, reason, error = parse_correctness(result["content"], threshold)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

            record = dict(answer_record)
            record.update(
                {
                    "correctness": correctness,
                    "keep": bool(keep) if keep is not None else False,
                    "judge_reason": reason,
                    "ok": correctness is not None and error is None,
                    "error": error,
                    "input_hash": input_hash,
                    "judge_model": cfg["model_judge"].get("model"),
                    "mode": cfg["model_judge"].get("mode"),
                    "latency_ms": (result or {}).get("latency_ms"),
                    "usage": (result or {}).get("usage") or {},
                    "created_at": C.now_iso(),
                }
            )
            return record

        from concurrent.futures import ThreadPoolExecutor, as_completed

        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {executor.submit(run_one, record, h): record for record, h in pending}
        interrupted = False
        try:
            for future in as_completed(futures):
                record = future.result()
                store.append(record)
                if not record["ok"]:
                    C.log(f"  评判失败：{record['key']} -> {record['error']}")
                progress.update(ok=bool(record["ok"]))
        except KeyboardInterrupt:
            interrupted = True
            C.log("收到中断信号：已完成评判已保存，重新运行即可续跑")
            for item in futures:
                item.cancel()
        finally:
            progress.close()
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

        if interrupted:
            return 130
        final, _, _ = store.load_last()
        if replaced:
            store.compact(final.values())

        completed = 0
        keep_count = 0
        for answer_record in tasks:
            item = final.get(answer_record["key"])
            if item and item.get("ok") and item.get("input_hash") == judge_hash(cfg, answer_record):
                completed += 1
                if item.get("keep") and (item.get("correctness") or 0) >= threshold:
                    keep_count += 1
        total_ok += completed
        total_keep += keep_count
        domain_results.append({"domain": domain, "tasks": len(tasks), "ok": completed, "keep": keep_count})
        C.log(f"领域 {domain}：完成 {completed}/{len(tasks)}，可达标 {keep_count} 条")

    incomplete = [item for item in domain_results if item["ok"] < item["tasks"]]
    C.log(f"阶段 3 汇总：评判 {total_ok}/{total_tasks}，达标 {total_keep}")
    if incomplete:
        C.log(f"仍有 {len(incomplete)} 个领域未完成评判，请重新运行续跑")
        return 3
    return 0


# ---------------------------------------------------------------------------
# 阶段 4：过滤、格式校验、查重、写最终数据集
# ---------------------------------------------------------------------------

OUTPUT_KEYS = ("multiturn", "tool", "id", "category", "messages")


def validate_final_record(record: dict) -> list[str]:
    """校验内部候选记录；通过后再转换为与源数据集一致的 5 字段结构。"""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["记录不是对象"]
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        errors.append("id 为空")
    if not isinstance(record.get("category"), str) or not record["category"].strip():
        errors.append("category 为空")
    if not isinstance(record.get("multiturn"), bool):
        errors.append("multiturn 必须是布尔值")
    if not isinstance(record.get("tool"), bool):
        errors.append("tool 必须是布尔值")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return errors + ["messages 必须正好是 user + assistant 两条"]
    roles = [item.get("role") if isinstance(item, dict) else None for item in messages]
    if roles != ["user", "assistant"]:
        errors.append("messages 角色必须正好是 ['user', 'assistant']")
    for item in messages:
        if not isinstance(item, dict):
            errors.append("messages 中存在非对象项")
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append("messages 存在空 content")

    if errors:
        return errors

    final_answer = str(record.get("final_answer") or "").strip()
    assistant_content = messages[-1]["content"]
    if not final_answer:
        errors.append("final_answer 为空")
    elif C.normalize_for_key(final_answer) not in C.normalize_for_key(assistant_content):
        errors.append("assistant content 不包含最终回答")

    try:
        json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        errors.append("记录无法 JSON 序列化")
    return errors


def validate_output_record(record: dict) -> list[str]:
    """校验最终 JSONL 行：必须严格等于源数据集结构，不携带任何额外字段。"""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["记录不是对象"]
    if set(record.keys()) != set(OUTPUT_KEYS):
        errors.append(f"字段必须严格为 {list(OUTPUT_KEYS)}，实际为 {list(record.keys())}")
    if not isinstance(record.get("multiturn"), bool):
        errors.append("multiturn 必须是布尔值")
    if not isinstance(record.get("tool"), bool):
        errors.append("tool 必须是布尔值")
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        errors.append("id 为空")
    if not isinstance(record.get("category"), str) or not record["category"].strip():
        errors.append("category 为空")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return errors + ["messages 必须正好是 user + assistant 两条"]
    if [item.get("role") for item in messages] != ["user", "assistant"]:
        errors.append("messages 角色必须正好是 ['user', 'assistant']")
    for item in messages:
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append("messages 存在空 content")

    try:
        json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        errors.append("记录无法 JSON 序列化")
    return errors


def _to_output_record(record: dict) -> dict:
    """只保留源数据集 TX-CW-self-cognition.jsonl 同款 5 个字段。"""
    return {
        "multiturn": bool(record.get("multiturn", False)),
        "tool": bool(record.get("tool", False)),
        "id": record.get("id"),
        "category": record.get("category"),
        "messages": record.get("messages"),
    }


def write_final_dataset(path: Path, records: list[dict]) -> None:
    public_records = [_to_output_record(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for record in public_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(tmp, path)

    # 写后复核：严格字段集、格式、行数、ID/问题查重
    rows, bad = C.read_jsonl(path)
    if bad:
        raise SystemExit(f"错误：最终文件 {path} 含 {bad} 行无法解析")
    if len(rows) != len(public_records):
        raise SystemExit(f"错误：最终文件 {path} 行数不一致（{len(rows)} != {len(public_records)}）")
    seen: set[str] = set()
    seen_ids: set[str] = set()
    for row in rows:
        errors = validate_output_record(row)
        if errors:
            raise SystemExit(f"错误：最终文件 {path} 格式校验失败：{errors}")
        record_id = row["id"]
        if record_id in seen_ids:
            raise SystemExit(f"错误：最终文件 {path} 存在重复 id：{record_id}")
        seen_ids.add(record_id)
        user = row["messages"][0]["content"]
        key = C.normalize_for_key(user)
        if key in seen:
            raise SystemExit(f"错误：最终文件 {path} 存在重复问题：{user[:80]}")
        seen.add(key)


def stage_export(cfg: dict) -> int:
    judged = load_stage(cfg, "judged")
    if not judged:
        C.log("错误：没有评判结果，请先运行 judge")
        return 2
    queries = load_stage(cfg, "queries")

    min_correctness = float(cfg["export"].get("min_correctness", 0.7))
    require_keep = bool(cfg["export"].get("require_keep", True))
    skip_source_dup = bool(cfg["export"].get("skip_source_duplicates", True))
    domain_summaries = []

    for domain in sorted(judged):
        records = list(judged[domain].values())
        source_questions = {
            C.normalize_for_key(str(record.get("original_question") or ""))
            for record in records
            if str(record.get("original_question") or "").strip()
        }
        for query_record in (queries.get(domain) or {}).values():
            original = str(query_record.get("original_question") or "").strip()
            if original:
                source_questions.add(C.normalize_for_key(original))

        entries: list[dict] = []
        seen_query: dict[str, int] = {}
        used_ids: set[str] = set()
        rejected_low = rejected_keep = invalid = dup_source = dup_query = 0

        for record in records:
            if not record.get("ok"):
                invalid += 1
                continue
            correctness = record.get("correctness")
            if not isinstance(correctness, (int, float)):
                invalid += 1
                continue
            correctness = float(correctness)
            keep = bool(record.get("keep"))
            if correctness < min_correctness:
                rejected_low += 1
                continue
            if require_keep and not keep:
                rejected_keep += 1
                continue

            expanded_query = str(record.get("expanded_query") or "").strip()
            assistant_content = str(record.get("assistant_content") or "").strip()
            final_answer = str(record.get("final_answer") or "").strip()
            if not expanded_query or not assistant_content or not final_answer:
                invalid += 1
                continue

            norm_query = C.normalize_for_key(expanded_query)
            if skip_source_dup and norm_query in source_questions:
                dup_source += 1
                continue

            final_record = _build_final_record(cfg, record, domain, used_ids)
            errors = validate_final_record(final_record)
            if errors:
                C.log(f"  跳过格式不合法记录 {record.get('key')}：{errors}")
                invalid += 1
                continue

            if norm_query in seen_query:
                index = seen_query[norm_query]
                if correctness > float(entries[index]["judge"]["correctness"]):
                    entries[index] = final_record
                dup_query += 1
                continue

            seen_query[norm_query] = len(entries)
            entries.append(final_record)

        path = dataset_paths(cfg, domain)["final"]
        write_final_dataset(path, entries)
        domain_summaries.append(
            {
                "domain": domain,
                "judged": len(records),
                "written": len(entries),
                "rejected_low": rejected_low,
                "rejected_keep": rejected_keep,
                "invalid": invalid,
                "dup_source": dup_source,
                "dup_query": dup_query,
                "path": path,
            }
        )
        C.log(
            f"领域 {domain}：判定 {len(records)}，写入 {len(entries)}，"
            f"低分 {rejected_low}，keep=false {rejected_keep}，格式无效 {invalid}，"
            f"源问题重复 {dup_source}，扩写重复 {dup_query}"
        )

    print()
    C.print_table(
        ["领域", "判定", "写入", "低分", "keep=false", "格式无效", "源重复", "扩写重复"],
        [
            [
                item["domain"],
                item["judged"],
                item["written"],
                item["rejected_low"],
                item["rejected_keep"],
                item["invalid"],
                item["dup_source"],
                item["dup_query"],
            ]
            for item in domain_summaries
        ],
    )
    C.log("最终数据集：")
    for item in domain_summaries:
        C.log(f"  {item['domain']} -> {item['path']}")
    return 0


def _build_final_record(cfg: dict, record: dict, domain: str, used_ids: set[str]) -> dict:
    expanded_query = str(record.get("expanded_query") or "").strip()
    assistant_content = str(record.get("assistant_content") or "").strip()
    final_answer = str(record.get("final_answer") or "").strip()
    category = str(record.get("category") or domain).strip() or domain

    source_id = str(record.get("source_id") or "item").strip() or "item"
    variant = int(record.get("variant_index") or 0)
    record_id = f"{source_id}__exp_{variant:03d}"
    if record_id in used_ids:
        record_id = f"{record_id}__{C.sha1_text(expanded_query)[:6]}"
    used_ids.add(record_id)

    return {
        "multiturn": bool(record.get("multiturn", False)),
        "tool": bool(record.get("tool", False)),
        "id": record_id,
        "category": category,
        "messages": [
            {"role": "user", "content": expanded_query},
            {"role": "assistant", "content": assistant_content},
        ],
        "final_answer": final_answer,
        "judge": {
            "correctness": record.get("correctness"),
            "keep": bool(record.get("keep")),
        },
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> int:
    C.setup_stdout()
    parser = argparse.ArgumentParser(description="查询扩写数据集工作流（单文件版）")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件")
    parser.add_argument("--mock", action="store_true", help="三个模型都使用内置 mock，用于本地调试")
    parser.add_argument("--fresh", action="store_true", help="清空本阶段输出后重新开始")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=("all", "expand", "answer", "judge", "export"),
        help="要运行的阶段，默认 all",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, mock=args.mock)
    if args.fresh:
        reset_generated(cfg, args.stage)
    C.log(f"配置：{cfg['_config_path']}")
    for key in MODEL_KEYS:
        C.log(f"  {key}: {cfg[key].get('model')}（{cfg[key].get('mode')}）")
    C.log(
        f"数据目录：{cfg['paths']['data_dir']} | 中间目录：{cfg['paths']['work_dir']}"
        f" | 输出目录：{cfg['paths']['output_dir']}"
    )

    status = 0
    if args.stage in ("all", "expand"):
        code = stage_expand(cfg, fresh=args.fresh)
        if code == 2:
            return 2
        status = max(status, code)
    if args.stage in ("all", "answer"):
        code = stage_answer(cfg, fresh=args.fresh)
        if code == 2 and args.stage == "answer":
            return 2
        status = max(status, code)
    if args.stage in ("all", "judge"):
        code = stage_judge(cfg, fresh=args.fresh)
        if code == 2 and args.stage == "judge":
            return 2
        status = max(status, code)
    if args.stage in ("all", "export"):
        code = stage_export(cfg)
        status = max(status, code)

    if status == 0:
        C.log("工作流完成")
    else:
        C.log(f"工作流部分完成（退出码 {status}），重新运行可续跑")
    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        C.log("已中断，进度已保留")
        raise SystemExit(130)
