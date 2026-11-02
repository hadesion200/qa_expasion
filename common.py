#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QA 评测流水线公共模块。

包含：YAML 配置加载、思维链清洗、JSONL 断点存储、OpenAI 兼容调用、mock 模拟。
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("缺少依赖 PyYAML，请先执行：pip install -r requirements.txt\n")
    raise SystemExit(1)

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write("缺少依赖 requests，请先执行：pip install -r requirements.txt\n")
    raise SystemExit(1)


ISO_FMT = "%Y-%m-%dT%H:%M:%S"
OUTPUT_SUFFIXES = (
    ".answers.jsonl",
    ".scores.jsonl",
    ".report.jsonl",
    ".report.json",
    ".report.md",
    ".report.csv",
    "_model_response.jsonl",
    "_model_judge.jsonl",
    "_report.md",
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MODEL_DEFAULTS: dict[str, Any] = {
    "mode": "api",           # api | mock
    "name": "",              # 报告展示用 ID；为空时回退到 model
    "base_url": "",
    "api_key": "",
    "api_key_env": "",
    "model": "",
    "temperature": 0.7,
    "top_p": None,
    "max_tokens": 2048,
    "timeout": 120,
    "max_retries": 3,
    "extra_body": {},
    "drop_keys": [],
    "keep_raw": False,
}

DEFAULTS: dict[str, Any] = {
    "project": {"name": "qa-eval"},
    "paths": {
        "data_dir": "./datasets",
        "prepared_dir": "./work/prepared",
        "output_dir": "./outputs",
    },
    "dataset": {
        "pattern": "*.jsonl",
        "recursive": True,
        "exclude_dirs": ["outputs", "work"],
        "dedupe": True,
        "dedupe_scope": "file",
        "dedupe_key": "question",
        "strip_cot": True,
        "multi_turn": False,
        "min_question_chars": 1,
        "max_records_per_file": 0,
    },
    "model_under_test": dict(MODEL_DEFAULTS),
    "judge": dict(MODEL_DEFAULTS, temperature=0.0, max_tokens=1024),
    "runtime": {
        "workers": 4,
        "max_context_chars": 16000,
        "request_interval_sec": 0.0,
        "resume": True,
        "retry_failed": True,
        "fsync_every": 1,
        "progress_interval_sec": 0.2,
        "strip_reasoning": True,
        "mock_delay_ms": 30,
    },
    "scoring": {
        "pass_score": 0.6,
        "weights": {
            "correctness": 0.35,
            "completeness": 0.25,
            "relevance": 0.15,
            "faithfulness": 0.15,
            "clarity": 0.10,
        },
    },
    "report": {
        "bins": [0.0, 0.6, 0.7, 0.8, 0.9, 1.0],
        "grade_bands": [
            [0.85, "优秀"],
            [0.70, "良好"],
            [0.60, "及格"],
            [0.00, "待改进"],
        ],
        "top_n_bad_cases": 5,
    },
}

DIM_DESCRIPTIONS = {
    "correctness": "事实、结论、数值与逻辑是否准确，是否与参考答案一致",
    "completeness": "是否覆盖参考答案的全部关键要点，遗漏核心要点应扣分",
    "relevance": "是否紧扣问题作答，有无答非所问或大量无关内容",
    "faithfulness": "有无编造、幻觉，或参考答案之外的错误信息",
    "clarity": "条理、结构、语言是否清晰易读",
}


def setup_stdout() -> None:
    """Windows 控制台尽量使用 UTF-8，避免中文乱码。"""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            except Exception:
                pass


_LOG_LOCK = threading.RLock()
_ACTIVE_PROGRESS = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _LOG_LOCK:
        progress = _ACTIVE_PROGRESS
        if progress is not None and progress.enabled:
            progress._clear_locked()
            print(line, flush=True)
            progress._draw_locked(force=True)
        else:
            print(line, flush=True)


class Progress:
    """单行动态进度条：TTY 下用 \\r 原地刷新，非 TTY 下不刷屏。"""

    def __init__(
        self,
        label: str,
        total: int,
        initial_done: int = 0,
        initial_ok: int = 0,
        initial_fail: int = 0,
        interval: float = 0.2,
    ):
        global _ACTIVE_PROGRESS
        self.label = str(label)
        self.total = max(1, int(total))
        self.done = min(max(0, int(initial_done)), self.total)
        self.ok = max(0, int(initial_ok))
        self.fail = max(0, int(initial_fail))
        self._initial_done = self.done
        self.interval = max(0.05, float(interval or 0.2))
        self.started = time.time()
        self.last_draw = 0.0
        self.closed = False
        self.enabled = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.width = 24
        with _LOG_LOCK:
            _ACTIVE_PROGRESS = self
            if self.enabled:
                self._draw_locked(force=True)

    def _line(self) -> str:
        ratio = min(1.0, self.done / self.total)
        filled = int(round(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(0.001, time.time() - self.started)
        new_done = max(0, self.done - self._initial_done)
        rate = new_done / elapsed if new_done else 0.0
        remaining = self.total - self.done
        eta = int(remaining / rate) if rate > 0 else 0
        label = self.label if len(self.label) <= 28 else self.label[:25] + "..."
        return (
            f"  {label} [{bar}] {self.done}/{self.total} {ratio:4.0%}"
            f" | 成功 {self.ok} 失败 {self.fail} | {rate:.1f} 条/秒 | ETA {eta}s"
        )

    def _draw_locked(self, force: bool = False) -> None:
        if not self.enabled or self.closed:
            return
        now = time.time()
        if not force and now - self.last_draw < self.interval:
            return
        self.last_draw = now
        sys.stdout.write("\r\x1b[K" + self._line())
        sys.stdout.flush()

    def _clear_locked(self) -> None:
        if not self.enabled:
            return
        sys.stdout.write("\r\x1b[K")
        sys.stdout.flush()

    def update(self, ok: bool = True) -> None:
        self.done = min(self.total, self.done + 1)
        if ok:
            self.ok += 1
        else:
            self.fail += 1
        with _LOG_LOCK:
            self._draw_locked(force=self.done >= self.total)

    def close(self) -> None:
        global _ACTIVE_PROGRESS
        with _LOG_LOCK:
            if self.enabled:
                self._clear_locked()
            self.closed = True
            if _ACTIVE_PROGRESS is self:
                _ACTIVE_PROGRESS = None



def now_iso() -> str:
    return time.strftime(ISO_FMT)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_dir(base: Path, value: str) -> str:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def load_config(path: str | Path) -> dict:
    """读取 YAML 配置；所有相对路径都相对于配置文件所在目录。"""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise SystemExit(f"错误：配置文件不存在：{config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"错误：YAML 解析失败：{exc}")
    if not isinstance(raw, dict):
        raise SystemExit("错误：配置文件顶层必须是映射（key: value）")

    cfg = deep_merge(DEFAULTS, raw)
    cfg["model_under_test"] = deep_merge(MODEL_DEFAULTS, cfg.get("model_under_test") or {})
    cfg["judge"] = deep_merge(MODEL_DEFAULTS, cfg.get("judge") or {})

    base = config_path.resolve().parent
    for key in ("data_dir", "prepared_dir", "output_dir"):
        cfg["paths"][key] = _resolve_dir(base, cfg["paths"][key])
    cfg["_config_path"] = str(config_path.resolve())
    cfg["_project_root"] = str(base)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    for section in ("model_under_test", "judge"):
        model = cfg[section]
        if model.get("mode") not in ("api", "mock"):
            raise SystemExit(f"错误：{section}.mode 只能是 api 或 mock")
        if not str(model.get("model") or "").strip():
            raise SystemExit(f"错误：{section}.model 不能为空")
        if model.get("mode") == "api" and not str(model.get("base_url") or "").strip():
            raise SystemExit(f"错误：{section}.base_url 不能为空（mode=api）")
        if float(model.get("max_tokens") or 0) <= 0:
            raise SystemExit(f"错误：{section}.max_tokens 必须大于 0")

    runtime = cfg["runtime"]
    if int(runtime.get("workers") or 0) < 1:
        raise SystemExit("错误：runtime.workers 必须 >= 1")
    if int(runtime.get("max_context_chars") or 0) < 500:
        raise SystemExit("错误：runtime.max_context_chars 必须 >= 500")

    weights = cfg["scoring"].get("weights") or {}
    if not weights or any(float(w) < 0 for w in weights.values()) or sum(float(w) for w in weights.values()) <= 0:
        raise SystemExit("错误：scoring.weights 必须是非负权重且总和大于 0")
    pass_score = float(cfg["scoring"].get("pass_score", 0.6))
    if not 0 <= pass_score <= 1:
        raise SystemExit("错误：scoring.pass_score 必须在 0~1 之间")

    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    for key in ("prepared_dir", "output_dir"):
        target = Path(cfg["paths"][key]).resolve()
        if target == data_dir:
            raise SystemExit(f"错误：paths.{key} 不能与 data_dir 相同（避免误删原始数据）")


def sanitize_filename(name: str, fallback: str = "model") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", str(name or "")).strip("._ ")
    return cleaned[:80] or fallback


def model_id_of(section: dict) -> str:
    return sanitize_filename(section.get("name") or section.get("model") or "model")


def dataset_stem(name: str) -> str:
    """数据集文件名：保留中文和空格，只替换路径/Windows 非法字符。"""
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(name or "")).strip(" .")
    return cleaned[:120] or "dataset"


def dataset_output_paths(cfg: dict, domain: str) -> dict[str, Path]:
    """每个 jsonl 数据集独立的三份产物。"""
    stem = dataset_stem(domain)
    out = Path(cfg["paths"]["output_dir"])
    return {
        "responses": out / f"{stem}_model_response.jsonl",
        "judge": out / f"{stem}_model_judge.jsonl",
        "report": out / f"{stem}_report.md",
    }


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------

_THINK_TAGS = ("think", "thinking", "reasoning", "analysis", "thought", "reflection", "cot")


def content_to_text(content: Any) -> str:
    """兼容 str / 多模态 list 的 content，跳过 thinking 类型的块。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if str(part.get("type") or "").lower() in ("thinking", "reasoning", "redacted_thinking"):
                    continue
                text = part.get("text")
                if text is None:
                    text = part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(content, dict):
        return content_to_text([content])
    return str(content)


def has_cot(text: Any) -> bool:
    if not isinstance(text, str) or not text:
        return False
    for tag in _THINK_TAGS:
        if re.search(rf"<\s*/?\s*{tag}\b", text, flags=re.I):
            return True
        if re.search(rf"\[\s*/?\s*{tag}\s*\]", text, flags=re.I):
            return True
    return bool(re.search(r"<\|?\s*(?:begin|start)_of_(?:thought|thinking|reasoning)\s*\|?>", text, flags=re.I))


def strip_cot(text: Any) -> str:
    """去掉常见思维链标记，保留最终回答。

    支持 <think>/<thinking>/<analysis>/<reasoning> 等成对标签、未闭合标签、
    [think]...[/think]、```thinking 代码块、<|begin_of_thought|> 等格式。
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return ""

    for tag in _THINK_TAGS:
        s = re.sub(rf"<\s*{tag}\b[^>]*>.*?<\s*/\s*{tag}\s*>", "", s, flags=re.S | re.I)
        s = re.sub(rf"<\s*{tag}\b[^>]*>.*\Z", "", s, flags=re.S | re.I)
        s = re.sub(rf"\[\s*{tag}\s*\].*?\[\s*/\s*{tag}\s*\]", "", s, flags=re.S | re.I)
        s = re.sub(rf"\[\s*{tag}\s*\].*\Z", "", s, flags=re.S | re.I)

    s = re.sub(
        r"<\|?\s*(?:begin|start)_of_(?:thought|thinking|reasoning|solution)\s*\|?>"
        r".*?"
        r"<\|?\s*(?:end|stop|final)_of_(?:thought|thinking|reasoning|solution)\s*\|?>",
        "",
        s,
        flags=re.S | re.I,
    )
    s = re.sub(r"```(?:think|thinking|reasoning|analysis)\b.*?```", "", s, flags=re.S | re.I)
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_for_key(text: Any) -> str:
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = s.casefold().strip()
    return re.sub(r"\s+", " ", s)


def sha1_text(*parts: Any) -> str:
    payload = "\x1f".join(str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def clip(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    if limit <= 0 or len(s) <= limit:
        return s
    return s[: max(0, limit - 20)].rstrip() + "\n...[内容过长已截断]"


def fit_messages(messages: list[dict], max_chars: int) -> list[dict]:
    """把消息总长度压到 max_context_chars 以内：保留 system，优先保留最后一条 user。"""
    msgs = [{"role": m.get("role", "user"), "content": str(m.get("content") or "")} for m in messages]
    if not msgs:
        return msgs
    total = sum(len(m["content"]) for m in msgs)
    if max_chars <= 0 or total <= max_chars:
        return msgs

    # 先丢弃较早的非 system 消息
    while total > max_chars and len(msgs) > 2:
        drop_idx = next((i for i, m in enumerate(msgs) if m["role"] != "system"), None)
        if drop_idx is None:
            break
        total -= len(msgs[drop_idx]["content"])
        msgs.pop(drop_idx)

    # 仍然超长则截断最后一条消息
    total = sum(len(m["content"]) for m in msgs)
    if total > max_chars:
        last = msgs[-1]
        keep = max(200, len(last["content"]) - (total - max_chars))
        last["content"] = clip(last["content"], keep)
    return msgs


def content_hash_for_answer(cfg: dict, rec: dict) -> str:
    model = cfg["model_under_test"]
    payload = {
        "model": model.get("model"),
        "messages": rec.get("prompt_messages"),
        "max_context_chars": cfg["runtime"].get("max_context_chars"),
        "temperature": model.get("temperature"),
        "max_tokens": model.get("max_tokens"),
        "extra_body": model.get("extra_body"),
    }
    return sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def content_hash_for_score(cfg: dict, rec: dict, answer: str) -> str:
    judge = cfg["judge"]
    payload = {
        "judge_model": judge.get("model"),
        "question": rec.get("question"),
        "reference": rec.get("reference_answer"),
        "answer": answer,
        "weights": cfg["scoring"].get("weights"),
        "max_context_chars": cfg["runtime"].get("max_context_chars"),
    }
    return sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


# ---------------------------------------------------------------------------
# JSONL 存储（实时落盘 + 断点续跑）
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> tuple[list[dict], int]:
    """读取 JSONL，返回 (记录列表, 坏行数)。"""
    rows: list[dict] = []
    bad = 0
    p = Path(path)
    if not p.is_file():
        return rows, 0
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


class JsonlStore:
    """按行追加并实时 flush/fsync，进程被杀后最多丢失正在写的一行。"""

    def __init__(self, path: str | Path, fsync_every: int = 1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync_every = max(1, int(fsync_every or 1))
        self._writes = 0
        self._lock = threading.Lock()
        self._repair_tail()

    def _repair_tail(self) -> None:
        """进程被强杀时可能留下没有换行符的半行；补一个换行，避免下一条被拼接。"""
        try:
            if not self.path.is_file() or self.path.stat().st_size == 0:
                return
            with open(self.path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                last = fh.read(1)
            if last != b"\n":
                with open(self.path, "ab") as fh:
                    fh.write(b"\n")
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
        except OSError:
            pass

    def load_last(self) -> tuple[dict[str, dict], int, int]:
        """读取全部记录，按 key 保留最后一条；返回 (map, 坏行数, 有效行数)。"""
        rows, bad = read_jsonl(self.path)
        last: dict[str, dict] = {}
        for row in rows:
            key = str(row.get("key") or "")
            if not key:
                bad += 1
                continue
            last[key] = row
        return last, bad, len(rows)

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line)
                fh.flush()
                if self._writes % self.fsync_every == 0:
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
            self._writes += 1

    def truncate(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
            self._writes = 0

    def compact(self, records: Iterable[dict]) -> None:
        """原子重写：去掉重复 key 与坏行，只保留最后状态。"""
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, self.path)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# 模型调用
# ---------------------------------------------------------------------------

class ModelCallError(RuntimeError):
    pass


_THROTTLE_LOCK = threading.Lock()
_LAST_CALL = 0.0


def _throttle(interval: float) -> None:
    global _LAST_CALL
    if interval <= 0:
        return
    with _THROTTLE_LOCK:
        wait = _LAST_CALL + interval - time.time()
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL = time.time()


def _api_key(model_cfg: dict) -> str:
    key = str(model_cfg.get("api_key") or "").strip()
    if key:
        return key
    env_name = str(model_cfg.get("api_key_env") or "").strip()
    if env_name:
        return str(os.environ.get(env_name, "")).strip()
    return ""


def resolve_chat_url(model_cfg: dict) -> str:
    base = str(model_cfg.get("base_url") or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _extract_response_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    message = choice.get("message") or {}
    text = content_to_text(message.get("content"))
    if not text:
        text = content_to_text(choice.get("text"))
    if text.strip():
        return text

    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    finish_reason = choice.get("finish_reason") or ""
    if str(reasoning).strip() or finish_reason == "length":
        raise ModelCallError(
            "模型仅返回思维链或空内容"
            f"（finish_reason={finish_reason}，有 reasoning={'是' if str(reasoning).strip() else '否'}）。"
            "通常是 max_tokens 太小被思维链耗尽，请调大 max_tokens（如 8192）。"
        )
    return ""


def chat_completion(
    model_cfg: dict,
    messages: list[dict],
    purpose: str,
    runtime: dict,
    label: str = "model",
) -> dict:
    """调用 OpenAI 兼容接口；mode=mock 时返回本地模拟结果。

    返回：{"text", "usage", "latency_ms", "mock"(可选)}
    """
    max_chars = int(runtime.get("max_context_chars") or 16000)
    msgs = fit_messages(messages, max_chars)
    if model_cfg.get("mode") == "mock":
        return _mock_chat(model_cfg, msgs, purpose, runtime)

    url = resolve_chat_url(model_cfg)
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
        _throttle(interval)
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
                    text = _extract_response_text(data)
                    if not text.strip():
                        last_err = "响应中未找到有效 content"
                    else:
                        if runtime.get("strip_reasoning", True):
                            text = strip_cot(text)
                        if not text.strip():
                            raise ModelCallError("模型输出去除思维链后为空，请检查 max_tokens 或模型行为")
                        return {
                            "text": text,
                            "usage": data.get("usage") or {},
                            "latency_ms": int((time.time() - started) * 1000),
                        }
            elif status == 429 or status >= 500 or status in (408, 409, 425):
                last_err = f"HTTP {status}: {resp.text[:200]}"
            else:
                raise ModelCallError(f"HTTP {status}: {resp.text[:500]}")

        if attempt < retries:
            wait = min(2 ** attempt, 15) + random.random() * 0.5
            log(f"    {label} 调用失败（{last_err}），{wait:.1f}s 后重试 {attempt + 1}/{retries}")
            time.sleep(wait)

    raise ModelCallError(f"{label} 调用失败（已重试 {retries} 次）：{last_err}")


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------

def build_judge_messages(cfg: dict, question: str, reference: str, answer: str) -> list[dict]:
    weights: dict[str, float] = cfg["scoring"]["weights"]
    total_weight = sum(float(w) for w in weights.values()) or 1.0
    max_chars = int(cfg["runtime"].get("max_context_chars") or 16000)

    dim_lines = []
    for dim, weight in weights.items():
        desc = DIM_DESCRIPTIONS.get(dim, "该维度表现")
        dim_lines.append(f"- {dim}（{desc}，权重 {float(weight) / total_weight:.0%}）")
    dim_text = "\n".join(dim_lines)

    keys = list(weights.keys())
    example = ", ".join(f'"{k}": 0.85' for k in keys)
    system = f"""你是一名严谨、公正、无偏的问答质量评测专家。
你的任务是对比【参考答案】，从多个维度给【待评测回答】打分。

# 评分维度（每项 0.00 ~ 1.00，满分 1.00）
{dim_text}

# 评分规则
- 以【参考答案】为主要依据，同时以客观事实为准；若参考答案本身明显有误，请在 reason 中指出。
- 语义等价但表述不同：不扣分；补充了正确且相关的信息：不扣分。
- 出现参考答案之外且错误的信息（幻觉）：在 faithfulness 中扣分。
- 不得因语言、篇幅、格式、文风等非质量因素加减分。
- 空回答、明确拒答、仅复述问题而无实质内容：所有维度记 0 分。
- 参考答案缺失时，依据客观事实评判 correctness 与 faithfulness，其余维度照常评分。
- 分数保留两位小数，不要超出 0~1 范围。

# 输出格式（严格遵守）
只输出一个 JSON 对象，不要输出解释文字，不要使用 Markdown 代码块。
格式：{{{example}, "reason": "<不超过 120 字的评分理由，说明主要得分点与扣分点>"}}
"""
    ref_text = (reference or "").strip() or "（无参考答案，请依据客观事实评判）"
    ans_text = (answer or "").strip() or "（空回答）"

    budget = max(1500, max_chars - len(system) - 400)
    q_text = clip((question or "").strip(), max(500, budget // 4))
    r_text = clip(ref_text, max(800, budget // 3))
    a_text = clip(ans_text, max(800, budget // 3))

    user = (
        f"【用户问题】\n{q_text}\n\n"
        f"【参考答案】\n{r_text}\n\n"
        f"【待评测回答】\n{a_text}\n\n"
        "请严格按系统提示的规则输出 JSON 评分结果。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_obj(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(s[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _to_score(value: Any) -> float | None:
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


def parse_judge_output(raw: str, weights: dict[str, float]) -> tuple[dict[str, float | None] | None, float | None, str, str | None]:
    """解析评分模型输出，返回 (dims, overall, reason, error)。"""
    obj = extract_json_obj(raw)
    if not obj:
        return None, None, "", "无法从评分模型输出中解析出 JSON"

    lowered = {str(k).lower(): v for k, v in obj.items()}
    candidates: dict[str, Any] = {}
    nested = obj.get("scores")
    if isinstance(nested, dict):
        candidates.update({str(k).lower(): v for k, v in nested.items()})
    candidates.update(lowered)

    raw_scores: dict[str, float] = {}
    for dim in weights:
        value = _to_score(candidates.get(dim))
        if value is not None:
            raw_scores[dim] = value

    if not raw_scores:
        return None, None, "", "评分结果中缺少全部维度字段"

    # 兼容 0~10 / 0~100 制：最大值超过 1 时整体缩放
    max_value = max(raw_scores.values())
    scale = 1.0
    if max_value > 1:
        scale = 10.0 if max_value <= 10 else 100.0

    dims: dict[str, float | None] = {}
    for dim in weights:
        value = raw_scores.get(dim)
        if value is None:
            dims[dim] = None
        else:
            dims[dim] = round(max(0.0, min(1.0, value / scale)), 4)

    reason = ""
    for key in ("reason", "reasons", "explanation", "comment", "rationale"):
        if candidates.get(key):
            reason = str(candidates[key]).strip()
            break

    return dims, weighted_overall(dims, weights), reason, None


def weighted_overall(dims: dict[str, float | None], weights: dict[str, float]) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for dim, weight in weights.items():
        value = dims.get(dim)
        if value is None:
            continue
        total += float(value) * float(weight)
        weight_sum += float(weight)
    if weight_sum <= 0:
        return None
    return round(total / weight_sum, 4)


# ---------------------------------------------------------------------------
# 本地 mock：无需真实 API 也能跑通全流程
# ---------------------------------------------------------------------------

_MOCK_TEMPLATES = [
    "（模拟回答）针对这个问题，可以分三点说明：第一，明确核心概念与适用范围；"
    "第二，给出关键参数或判断依据；第三，补充一个实际应用中的注意事项。",
    "（模拟回答）这个问题需要结合具体场景分析。总体结论是：先确认前提条件，"
    "再比较不同方案的收益与代价，最后给出可执行的建议。",
    "（模拟回答）简要回答如下：该问题涉及基础原理与工程实践两个层面，"
    "原理层面应关注定义和边界条件，实践层面应关注可靠性、成本与可维护性。",
    "（模拟回答）可以这样理解：核心结论成立，但在极端条件下需要修正；"
    "建议结合最新资料复核关键数据，并以实际测试结果为准。",
]


def _mock_answer(prompt: str) -> str:
    question = prompt.strip()
    digest = sha1_text("mock-answer", question)
    index = int(digest[:4], 16) % len(_MOCK_TEMPLATES)
    body = _MOCK_TEMPLATES[index]
    # 故意带上思维链，用来验证最终落盘文件是否已清洗
    think = f"<think>模拟推理：用户问的是「{clip(question, 40)}」，需要给出结构化、简洁的回答。</think>\n\n"
    return f"{think}{body}\n\n（用于流水线模拟测试，问题摘要：{clip(question, 60)}）"


def _parse_judge_sections(prompt: str) -> tuple[str, str, str]:
    pattern = re.compile(
        r"【用户问题】\n(.*?)\n\n【参考答案】\n(.*?)\n\n【待评测回答】\n(.*?)\n\n请严格",
        re.S,
    )
    match = pattern.search(prompt)
    if not match:
        return "", "", ""
    return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()


def _similarity(a: str, b: str) -> float:
    def norm(text: str) -> str:
        text = unicodedata.normalize("NFKC", text or "").lower()
        return re.sub(r"\s+", "", text)

    left, right = norm(a)[:3000], norm(b)[:3000]
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _mock_judge(prompt: str) -> str:
    question, reference, answer = _parse_judge_sections(prompt)
    if not answer or answer.startswith("（空回答）"):
        dims = {dim: 0.0 for dim in DEFAULTS["scoring"]["weights"]}
        dims["reason"] = "回答为空，无法评分。"
        return json.dumps(dims, ensure_ascii=False)

    sim = _similarity(reference, answer) if reference and not reference.startswith("（无参考答案") else 0.55
    seed = sha1_text("mock-judge", question, reference, answer)
    # 模拟一个“有高有低”的评分分布：内容重合度 + 按样本散列的稳定扰动
    quality = ((int(seed[:6], 16) % 401) - 200) / 1000.0  # -0.20 ~ +0.20
    base = 0.68 + 0.22 * sim + quality

    dims: dict[str, float] = {}
    for offset, dim in enumerate(DEFAULT_DIM_ORDER):
        jitter = (int(sha1_text(seed, dim)[:4], 16) % 201 - 100) / 1000.0  # -0.10 ~ +0.10
        adjust = {
            "correctness": 0.0,
            "completeness": -0.06,
            "relevance": 0.08,
            "faithfulness": 0.0,
            "clarity": 0.10,
        }.get(dim, 0.0)
        dims[dim] = round(max(0.0, min(1.0, base + adjust + jitter)), 2)

    worst = min(dims, key=dims.get)
    if base >= 0.80:
        tone = "整体回答质量较好"
    elif base >= 0.65:
        tone = "整体回答质量一般"
    else:
        tone = "整体回答质量偏弱"
    reason = f"模拟评分：与参考答案文本重合度约 {sim:.0%}，{tone}，主要短板在 {worst} 维度。"
    payload: dict[str, Any] = {dim: dims[dim] for dim in DEFAULT_DIM_ORDER}
    payload["reason"] = reason
    return json.dumps(payload, ensure_ascii=False)


DEFAULT_DIM_ORDER = ("correctness", "completeness", "relevance", "faithfulness", "clarity")


def _mock_chat(model_cfg: dict, messages: list[dict], purpose: str, runtime: dict) -> dict:
    delay = max(0.0, float(runtime.get("mock_delay_ms") or 0)) / 1000.0
    if delay:
        time.sleep(random.uniform(delay * 0.5, delay * 1.5))
    prompt = messages[-1]["content"] if messages else ""
    raw = _mock_judge(prompt) if purpose == "judge" else _mock_answer(prompt)
    text = strip_cot(raw) if purpose != "judge" else raw
    return {
        "text": text,
        "usage": {"prompt_tokens": len(prompt) // 3, "completion_tokens": len(text) // 3},
        "latency_ms": int(delay * 1000),
        "mock": True,
    }


# ---------------------------------------------------------------------------
# 通用统计
# ---------------------------------------------------------------------------

def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def score_distribution(values: list[float], bins: list[float]) -> dict[str, int]:
    edges = sorted(float(b) for b in bins)
    result: dict[str, int] = {}
    for lo, hi in zip(edges, edges[1:]):
        label = f"[{lo:.1f},{hi:.1f})"
        result[label] = sum(1 for v in values if lo <= v < hi)
    if edges and any(v >= edges[-1] for v in values):
        result[f"[{edges[-1]:.1f},{edges[-1]:.1f}]"] = sum(1 for v in values if v >= edges[-1])
    return result


def grade_of(mean: float | None, bands: list) -> str:
    if mean is None:
        return "无数据"
    for threshold, label in bands:
        if mean >= float(threshold):
            return str(label)
    return "待改进"


def load_prepared_by_domain(cfg: dict) -> dict[str, list[dict]]:
    """读取 work/prepared 下的文件，每个 jsonl 文件是一个独立知识领域。"""
    prepared_dir = Path(cfg["paths"]["prepared_dir"])
    datasets: dict[str, list[dict]] = {}
    if not prepared_dir.is_dir():
        return datasets
    for path in sorted(prepared_dir.glob("*.jsonl")):
        rows, bad = read_jsonl(path)
        if bad:
            log(f"  警告：{path.name} 有 {bad} 行无法解析，已忽略")
        deduped: dict[str, dict] = {}
        for record in rows:
            key = str(record.get("key") or "")
            if key:
                deduped[key] = record
        if deduped:
            datasets[path.stem] = list(deduped.values())
    return dict(sorted(datasets.items(), key=lambda item: item[0].casefold()))


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


# ---- 从 qa_prepare.py 内联：源记录解析 ----

def parse_record(record: dict, multi_turn: bool, do_strip: bool) -> tuple[dict, bool]:
    """解析一条记录，返回 (标准化字段, 是否清洗过思维链)。"""
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("字段 messages 缺失或为空")

    normalized: list[dict] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] 不是对象")
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"messages[{index}] 的 role 非法：{role!r}")
        content = content_to_text(message.get("content")).strip()
        normalized.append({"role": role, "content": content})

    last_assistant = next((i for i in range(len(normalized) - 1, -1, -1) if normalized[i]["role"] == "assistant"), None)
    if normalized[-1]["role"] == "user":
        # 数据以用户提问结尾：视为“无参考答案”的待答题，问题就是最后一条 user
        reference_raw = ""
        context = normalized
        last_user = normalized[-1]
    else:
        if last_assistant is None:
            reference_raw = ""
            context = normalized
        else:
            reference_raw = normalized[last_assistant]["content"]
            context = normalized[:last_assistant]
        last_user = next((m for m in reversed(context) if m["role"] == "user"), None)

    if last_user is None or not last_user["content"].strip():
        raise ValueError("未找到有效的用户提问（user 消息）")

    question = last_user["content"].strip()
    cot_removed = False
    reference = reference_raw
    if do_strip and has_cot(reference_raw):
        reference = strip_cot(reference_raw)
        cot_removed = True

    if multi_turn:
        prompt_messages: list[dict] = []
        for message in context:
            content = message["content"]
            if do_strip and message["role"] == "assistant":
                content = strip_cot(content)
            if content.strip():
                prompt_messages.append({"role": message["role"], "content": content})
        if not prompt_messages:
            prompt_messages = [{"role": "user", "content": question}]
    else:
        last_system = next((m for m in reversed(context) if m["role"] == "system"), None)
        prompt_messages = []
        if last_system and last_system["content"].strip():
            prompt_messages.append({"role": "system", "content": last_system["content"]})
        prompt_messages.append({"role": "user", "content": question})

    return {
        "question": question,
        "reference_answer": reference.strip(),
        "prompt_messages": prompt_messages,
        "cot_removed": cot_removed,
    }, cot_removed
