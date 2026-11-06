#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_jev.py — jev-infer 概率输出能力一键测试

用途：
    对已启动的 jev-infer 服务跑一遍三题型（noul / choice / score）+ 一致性测试，
    把所有原始返回落盘，并生成可读报告。

依赖：Python 3.8+，纯标准库，无需 pip install。
平台：Linux / Windows 通用。

用法：
    python3 test_jev.py
    python3 test_jev.py --base http://127.0.0.1:18080
    python3 test_jev.py --base http://10.102.80.140:18080 --outdir ./runs
    python3 test_jev.py --no-engine          # 跳过引擎层对比
    python3 test_jev.py --endpoint /v1/systemone

产物（默认 ./jev_test_runs/<时间戳>/）：
    raw/*.json        每个 case 的 HTTP 状态、耗时、原始响应体
    summary.json      结构化的判定结果
    report.md         人读的报告
    report.txt        终端同款纯文本
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

SEP = "=" * 62

# 统一 stdout 编码，避免 Windows 控制台 GBK 无法输出 ✔/✘ 或中文
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass



# ---------------------------------------------------------------- HTTP

def _request(url, payload=None, timeout=120):
    t0 = time.time()
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return {"ok": True, "status": r.status, "body": body,
                    "ms": int((time.time() - t0) * 1000)}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return {"ok": False, "status": e.code, "body": body,
                "ms": int((time.time() - t0) * 1000)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": None, "body": f"{type(e).__name__}: {e}",
                "ms": int((time.time() - t0) * 1000)}


def get(url, timeout=30):
    return _request(url, None, timeout)


def post(url, payload, timeout=120):
    return _request(url, payload, timeout)


# ------------------------------------------------------- 响应容错解析

def _walk(obj, out, depth=0):
    """递归找出所有"看起来是概率分布"的浮点数组（和≈1）。"""
    if depth > 8:
        return
    if isinstance(obj, list) and len(obj) >= 2 and \
            all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
        s = sum(obj)
        if 0.95 <= s <= 1.05:
            out.append([float(x) for x in obj])
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out, depth + 1)


def extract_target(body, expected_len=None):
    """从响应体里提取分布。返回 (target 或 None, 解析后的 json 或 None)。"""
    try:
        j = json.loads(body)
    except Exception:  # noqa: BLE001
        return None, None
    cands = []
    _walk(j, cands)
    if expected_len:
        exact = [c for c in cands if len(c) == expected_len]
        if exact:
            return exact[0], j
    if cands:
        return max(cands, key=len), j
    return None, j


def tv(p, q):
    """全变差距离。"""
    if not p or not q or len(p) != len(q):
        return None
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


# ---------------------------------------------------------------- 用例

STATE_ORBIT = (
    "【任务】校验星历外推结果。"
    "【依据】GNSS 广播星历有效期为 2 小时，超期后轨道误差按每小时 1 km 量级增长。"
    "【证据】本次外推使用 3 小时前的广播星历，输出的卫星位置残差为 1.8 km。"
)

STATE_METHOD = (
    "【任务】选择定轨方法。"
    "【依据】仅有单站测距数据，观测弧长 8 分钟，无测速数据，无法进行力学模型精化。"
    "【证据】历史同类任务中，短弧单站定轨的收敛率不足 40%。"
)

STATE_SCORE = (
    "【任务】评定方案可行性。"
    "【方案】采用双频 GPS 单点定位为低轨卫星提供自主导航，不引入星间链路，"
    "仅依赖星上接收机在轨自主解算。"
)

CRIT_CHOICE = {
    "c0": "批处理最小二乘",
    "c1": "扩展卡尔曼滤波",
    "c2": "无迹卡尔曼滤波",
    "c3": "序贯单点定位",
}

CRIT_CHOICE_SWAPPED = {
    "c0": "批处理最小二乘",
    "c1": "序贯单点定位",
    "c2": "无迹卡尔曼滤波",
    "c3": "扩展卡尔曼滤波",
}

CRIT_SCORE = ["0", "1", "2", "3", "4", "5"]


def build_cases(model):
    return [
        {
            "name": "noul",
            "kind": "noul",
            "expect_len": 2,
            "payload": {
                "model": model,
                "state": STATE_ORBIT,
                "questions": {"q": {"type": "noul",
                                    "instructions": "该外推结果是否仍可用于精密定轨？"}},
            },
        },
        {
            "name": "choice",
            "kind": "choice",
            "expect_len": len(CRIT_CHOICE),
            "payload": {
                "model": model,
                "state": STATE_METHOD,
                "questions": {"q": {"type": "choice",
                                    "instructions": "在仅有单站测距数据的条件下，应优先采用哪种定轨方法？",
                                    "criteria": CRIT_CHOICE}},
            },
        },
        {
            "name": "choice_swapped",
            "kind": "choice",
            "expect_len": len(CRIT_CHOICE_SWAPPED),
            "payload": {
                "model": model,
                "state": STATE_METHOD,
                "questions": {"q": {"type": "choice",
                                    "instructions": "在仅有单站测距数据的条件下，应优先采用哪种定轨方法？",
                                    "criteria": CRIT_CHOICE_SWAPPED}},
            },
        },
        {
            "name": "score",
            "kind": "score",
            "expect_len": len(CRIT_SCORE),
            "payload": {
                "model": model,
                "state": STATE_SCORE,
                "questions": {"q": {"type": "score",
                                    "instructions": "按 0–5 评定该方案的工程可行性（0=不可行，3=需重大修改，5=可直接实施）",
                                    "criteria": CRIT_SCORE}},
            },
        },
        {
            "name": "noul_repeat",
            "kind": "noul",
            "expect_len": 2,
            "payload": {
                "model": model,
                "state": STATE_ORBIT,
                "questions": {"q": {"type": "noul",
                                    "instructions": "该外推结果是否仍可用于精密定轨？"}},
            },
        },
    ]


# ---------------------------------------------------------------- 判定

def judge(case, target):
    """返回 (verdict_dict, 说明文本)。"""
    out = {"target": target}
    if target is None:
        return {"pass": False, "reason": "未在响应中找到归一化分布"}, "✘ 无分布", False
    s = sum(target)
    out["sum"] = round(s, 6)
    out["max"] = round(max(target), 6)
    out["argmax"] = int(max(range(len(target)), key=lambda i: target[i]))

    why = []
    ok = True
    if abs(s - 1) > 0.01:
        ok = False
        why.append(f"和={s:.4f}≠1")
    if len(target) != case["expect_len"]:
        ok = False
        why.append(f"维度={len(target)}≠{case['expect_len']}")
    if max(target) < 0.34:
        ok = False
        why.append(f"无区分度(max={max(target):.3f})")

    if case["kind"] == "score":
        exp = sum(i * p for i, p in enumerate(target))
        out["expectation"] = round(exp, 3)
        why.append(f"期望分={exp:.2f}")

    return out, ("✔ " + "；".join(why) if ok else "✘ " + "；".join(why)), ok


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("JEV_BASE", "http://127.0.0.1:18080"),
                    help="decision_api 地址，默认 http://127.0.0.1:18080")
    ap.add_argument("--engine", default=os.environ.get("JEV_ENGINE", "http://127.0.0.1:18000"),
                    help="vLLM 引擎层地址，默认 http://127.0.0.1:18000")
    ap.add_argument("--model", default=os.environ.get("JEV_MODEL", ""),
                    help="模型 id，留空则从 /v1/models 自动取")
    ap.add_argument("--endpoint", default="/v1/systemone")
    ap.add_argument("--outdir", default="./jev_test_runs")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--no-engine", action="store_true", help="跳过引擎层 /pooling 对比")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, ts)
    rawdir = os.path.join(outdir, "raw")
    os.makedirs(rawdir, exist_ok=True)

    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say(SEP)
    say("jev-infer 概率输出能力测试")
    say(f"时间     : {datetime.now().isoformat(timespec='seconds')}")
    say(f"base     : {base}")
    say(f"endpoint : {args.endpoint}")
    say(f"产物目录 : {os.path.abspath(outdir)}")
    say(SEP)

    # ---- 1. 探活
    say("\n[1/4] 探活")
    h = get(f"{base}/health")
    say(f"  GET /health      -> {h['status']}  ({h['ms']} ms)  {h['body'][:80]}")

    m = get(f"{base}/v1/models")
    model = args.model
    if not model:
        try:
            j = json.loads(m["body"])
            items = j.get("data") or j.get("models") or []
            if items:
                model = items[0].get("id") or items[0].get("name")
        except Exception:  # noqa: BLE001
            pass
    say(f"  GET /v1/models   -> {m['status']}  ({m['ms']} ms)")
    say(f"  使用 model id    : {model or '(未知，沿用空字符串)'}")
    if not model:
        say("  ! 未能自动识别 model，请用 --model 显式指定")

    with open(os.path.join(rawdir, "_probe_health.txt"), "w", encoding="utf-8") as f:
        f.write(f"status={h['status']} ms={h['ms']}\n{h['body']}\n")
    with open(os.path.join(rawdir, "_probe_models.json"), "w", encoding="utf-8") as f:
        f.write(m["body"])

    # ---- 2. 跑用例
    say("\n[2/4] 三题型")
    cases = build_cases(model)
    summary = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "base": base,
        "endpoint": args.endpoint,
        "model": model,
        "cases": {},
    }
    for c in cases:
        r = post(base + args.endpoint, c["payload"], args.timeout)
        target, _ = extract_target(r["body"], c["expect_len"])
        verdict, why, ok = judge(c, target) if target is not None else \
            ({"pass": False}, "✘ 未找到分布", False)
        verdict["http_status"] = r["status"]
        verdict["ms"] = r["ms"]
        verdict["pass"] = ok
        summary["cases"][c["name"]] = verdict
        with open(os.path.join(rawdir, f"{c['name']}.json"), "w", encoding="utf-8") as f:
            json.dump({"request": c["payload"], "status": r["status"],
                       "ms": r["ms"], "response_raw": r["body"]},
                      f, ensure_ascii=False, indent=2)
        say(f"  {c['name']:<14} HTTP {str(r['status']):<4} {r['ms']:>6} ms  {why}")

    # ---- 3. 一致性
    say("\n[3/4] 一致性")
    def tgt(name):
        return summary["cases"].get(name, {}).get("target")

    # 3a 选项换序（c1 <-> c3 对调）
    a, b = tgt("choice"), tgt("choice_swapped")
    if a and b and len(a) == 4 and len(b) == 4:
        b_restored = [b[0], b[3], b[2], b[1]]      # 把 c1/c3 换回来
        d = tv(a, b_restored)
        summary["consistency_order_tv"] = round(d, 4)
        ok = d <= 0.05
        summary["consistency_order_pass"] = ok
        say(f"  选项换序 TV = {d:.4f}  {'✔ 等变（≤0.05）' if ok else '✘ 不等变（>0.05），概率可能是伪概率'}")
    else:
        say("  选项换序  跳过（缺少有效分布）")

    # 3b 重复同问
    a, b = tgt("noul"), tgt("noul_repeat")
    if a and b:
        d = tv(a, b)
        summary["determinism_tv"] = round(d, 4)
        ok = d <= 1e-6
        summary["determinism_pass"] = ok
        say(f"  重复同问 TV = {d:.6f}  {'✔ 确定性' if ok else '⚠ 有抖动（采样未关闭？）'}")
    else:
        say("  重复同问  跳过")

    # ---- 4. 引擎层对比
    if not args.no_engine:
        say("\n[4/4] 引擎层对比（判断故障在哪一层）")
        e = post(f"{args.engine}/pooling",
                 {"model": model, "input": STATE_ORBIT}, args.timeout)
        say(f"  POST /pooling  -> {e['status']}  ({e['ms']} ms)  响应长度 {len(e['body'])}")
        with open(os.path.join(rawdir, "_engine_pooling.json"), "w", encoding="utf-8") as f:
            json.dump({"status": e["status"], "ms": e["ms"], "response_raw": e["body"]},
                      f, ensure_ascii=False, indent=2)
        if e["status"] == 200:
            say("  → 引擎层正常；若上面三题型失败，问题在 decision_api（head/归一化）")
        else:
            say("  → 引擎层也异常，问题在 vLLM / 权重 / 启动")
    else:
        say("\n[4/4] 引擎层对比  已跳过")

    # ---- 汇总
    total = sum(1 for v in summary["cases"].values() if v.get("pass"))
    say("\n" + SEP)
    say(f"汇总：{total}/{len(summary['cases'])} 通过")
    say(SEP)
    say(f"{'case':<16}{'HTTP':<6}{'ms':>7}  {'sum':>8}  {'max':>7}  target")
    for name, v in summary["cases"].items():
        t = v.get("target")
        ts_str = "[" + ", ".join(f"{x:.3f}" for x in t) + "]" if t else "(none)"
        say(f"{name:<16}{str(v.get('http_status')):<6}{v.get('ms', 0):>7}  "
            f"{v.get('sum', 0):>8.4f}  {v.get('max', 0):>7.3f}  {ts_str}")

    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    summary["passed"] = total
    summary["total"] = len(summary["cases"])

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("# jev-infer 概率输出测试报告\n\n```\n" + "\n".join(lines) + "\n```\n")

    print(f"\n产物已保存: {os.path.abspath(outdir)}")
    print("  raw/            每个 case 的请求+原始响应")
    print("  summary.json    结构化判定")
    print("  report.md       可读报告")

    return 0 if total == len(summary["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
