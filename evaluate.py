#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py —— 通用 Jev 数据集测评（一个入口，扫 datasets/ 下所有 jsonl）

流程
    datasets/*.jsonl  →  逐条调本地 Jev API  →  predictions  →  打分  →  results/<模型名>/

打分
    直接复用 OpenDecisions 官方 scorer（od/score.py，未改一行）：
        Capability = (Intelligence + Calibration) / 2
        Intelligence = 三题型等权；题型内按 tier 加权（easy .1 / standard .2 / judge .3 / hard .4）
                       choice: 100(acc−c)/(1−c)；noul: 100(acc−0.5)/0.5；score: 100(1−nMAE/nMAE_mid)
        Calibration  = 三题型等权；choice 只统计 hard 档；各 100(1−ECE/0.5)
    另外补三块官方没有的：可靠性分桶表、一致性（换序/重复）、延迟与失败分析。

支持的 jsonl 格式（缺字段会自动兜底）
    {"id", "state", "question": {"type","instructions","criteria"}, "labels", "expected", "tier"}
    {"id", "state", "question"(str), "options": [...], "target": [p...]}      # jev-distill 风格
    {"id", "state", "question": {"...criteria"}, "answer"/"gold"/"label"}

用法
    ./run.sh                        # 一键：全部数据集
    ./run.sh --limit 50             # 冒烟
    ./run.sh --dataset opendecisions.jsonl
    ./run.sh --fresh                # 忽略断点
"""

import argparse
import configparser
import http.client
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "od"))
import score as official  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

SEP = "=" * 78


def cget(cp, sec, key, default):
    v = cp.get(sec, key, fallback=None)
    return default if v is None else v


# ================================================================ 读入 & 归一化

def normalize(raw, ds_name, idx):
    """把各种 jsonl 形态统一成官方 scorer 需要的 task。"""
    q = raw.get("question")
    if isinstance(q, str):
        q = {"type": raw.get("type") or raw.get("kind") or "choice",
             "instructions": q,
             "criteria": raw.get("options") or raw.get("criteria") or {}}
    elif not isinstance(q, dict):
        q = {}

    qtype = str(q.get("type") or raw.get("type") or raw.get("kind") or "choice").lower()
    crit = q.get("criteria") or raw.get("options") or raw.get("criteria") or {}
    if isinstance(crit, list):
        crit = {str(x): str(x) for x in crit}

    labels = raw.get("labels")
    if not labels:
        labels = [str(k) for k in crit.keys()] if crit else []
    labels = [str(x) for x in labels]

    expected = raw.get("expected")
    if expected is None:
        expected = raw.get("answer", raw.get("gold", raw.get("label")))
    target = raw.get("target_probs") or raw.get("target")
    target_probs = None
    if isinstance(target, dict) and target:
        target_probs = {str(k): float(v) for k, v in target.items()}
    elif isinstance(target, list) and target and len(target) == len(labels):
        target_probs = {str(k): float(v) for k, v in zip(labels, target)}
    if expected is None and target_probs:
        expected = max(target_probs, key=target_probs.get)

    if expected is None or not labels:
        return None
    if str(expected) not in labels:                      # 大小写不一致时按忽略大小写对齐
        low = {x.lower(): x for x in labels}
        if str(expected).lower() not in low:
            return None
        expected = low[str(expected).lower()]
    expected = str(expected)

    extra = {}
    if qtype in ("noul", "yesno", "yes/no"):
        qtype = "noul"
        # 官方 scorer 要求 noul 的 labels 是 yes/no（item_outcome 直接取 probs["yes"]），
        # 但数据集可能是 true/false，这里做一次规范化，并记住原始键
        low = {x.lower(): x for x in labels}
        yes_lab = next((v for k, v in low.items() if k in ("yes", "true", "1", "成立")), None)
        no_lab = next((v for k, v in low.items() if k in ("no", "false", "0", "不成立")), None)
        if not yes_lab or not no_lab:
            return None
        expected = "yes" if expected.lower() in ("yes", "true", "1", "成立") else "no"
        labels = ["no", "yes"]
        extra = {"_yes": yes_lab, "_no": no_lab}
        if target_probs:
            py = target_probs.get(yes_lab, target_probs.get("yes", 0.0))
            target_probs = {"no": 1.0 - float(py), "yes": float(py)}

    task = {
        "id": str(raw.get("id") or f"{ds_name}:{idx}"),
        "state": raw.get("state") if isinstance(raw.get("state"), str)
                 else json.dumps(raw.get("state") or "", ensure_ascii=False),
        "question": {"type": qtype,
                     "instructions": q.get("instructions") or raw.get("instructions") or "",
                     "criteria": crit},
        "labels": labels,
        "expected": expected,
        "tier": raw.get("tier") or "standard",
        "target_probs": target_probs,
    }
    task.update(extra)
    return task


def load_dataset(path, limit=0):
    rows = []
    name = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            t = normalize(raw, name, i)
            if t:
                rows.append(t)
    if limit and limit < len(rows):
        # 均匀抽样而不是取前 N 条：数据文件常按 family 排序，取头部会只覆盖一类
        step = len(rows) / float(limit)
        rows = [rows[min(int(i * step), len(rows) - 1)] for i in range(limit)]
    return rows


# ================================================================ 调 API

class Client:
    def __init__(self, host, port, headers, timeout):
        self.host, self.port, self.headers, self.timeout = host, port, headers, timeout
        self.conn = None

    def post(self, path, body):
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        for attempt in (0, 1):
            try:
                self.conn.request("POST", path, body=body, headers=self.headers)
                r = self.conn.getresponse()
                return r.status, r.read()
            except Exception:  # noqa: BLE001
                try:
                    if self.conn:
                        self.conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self.conn = None
                if attempt:
                    raise
                self.conn = http.client.HTTPConnection(self.host, self.port,
                                                       timeout=self.timeout)


def to_probs(ans, task):
    """对齐到 task.labels；noul 统一成 {"no","yes"}（官方 scorer 要这个键）。"""
    p = (ans or {}).get("probabilities")
    if not isinstance(p, dict) or not p:
        return None
    p = {str(k): float(v) for k, v in p.items()}
    if task["question"]["type"] == "noul":
        # 数据集原本的键可能是 true/false，normalize() 里记在 _yes/_no
        py = None
        for k in (task.get("_yes"), "yes", "true", "True", "1"):
            if k and k in p:
                py = p[k]
                break
        if py is None:
            for k in (task.get("_no"), "no", "false", "False", "0"):
                if k and k in p:
                    py = 1.0 - p[k]
                    break
        if py is None and len(p) == 2:                 # 最后按顺序兜底
            py = list(p.values())[0]
        return None if py is None else {"no": 1.0 - py, "yes": py}
    labels = task["labels"]
    if set(p) == set(labels):
        vals = {k: p[k] for k in labels}
    elif len(p) == len(labels):
        vals = dict(zip(labels, p.values()))
    else:
        return None
    s = sum(vals.values())
    return {k: v / s for k, v in vals.items()} if s > 0 else None


def predict(tasks, cfg, state_path, fresh=False, progress=True):
    done = {}
    if os.path.exists(state_path) and not fresh:
        try:
            done = json.load(open(state_path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            done = {}
    todo = [t for t in tasks if t["id"] not in done]
    if done:
        print(f"      断点续跑：已完成 {len(done)}，本次待做 {len(todo)}")

    lock = threading.Lock()
    n = [0]

    def work(start):
        cli = Client(cfg["host"], cfg["port"], cfg["headers"], cfg["timeout"])
        for i in range(start, len(todo), cfg["conc"]):
            t = todo[i]
            qid = t["id"]
            payload = {"model": cfg["model"], "state": t["state"],
                       "questions": {qid: {"type": t["question"]["type"],
                                           "instructions": t["question"]["instructions"],
                                           "criteria": t["question"]["criteria"]}}}
            rec = {"ok": False}
            t0 = time.perf_counter()
            try:
                st, raw = cli.post(cfg["endpoint"],
                                   json.dumps(payload, ensure_ascii=False).encode())
                if 200 <= st < 300:
                    ans = (json.loads(raw).get("answers") or {}).get(qid) or {}
                    probs = to_probs(ans, t)
                    if probs:
                        rec = {"ok": True, "probs": probs}
                    else:
                        rec["error"] = "no_probs"
                else:
                    rec["error"] = f"http_{st}"
            except Exception as e:  # noqa: BLE001
                rec["error"] = type(e).__name__
            rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            with lock:
                done[qid] = rec
                n[0] += 1
                if n[0] % 50 == 0:
                    json.dump(done, open(state_path + ".tmp", "w", encoding="utf-8"),
                              ensure_ascii=False)
                    os.replace(state_path + ".tmp", state_path)
                    if progress:
                        print(f"      {n[0]}/{len(todo)}")

    ths = [threading.Thread(target=work, args=(k,), daemon=True) for k in range(cfg["conc"])]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    json.dump(done, open(state_path, "w", encoding="utf-8"), ensure_ascii=False)
    return done


# ================================================================ 一致性

def consistency(tasks, cfg, n_pairs, rng):
    cand = [t for t in tasks if len(t["labels"]) >= 3]
    rng.shuffle(cand)
    n_order = int(n_pairs * 0.75)
    plan = []
    for t in cand[:n_order]:
        crit = t["question"]["criteria"]
        if isinstance(crit, dict):
            keys = list(crit.keys())
            perm = keys[:]
            for _ in range(5):
                rng.shuffle(perm)
                if perm != keys:
                    break
            plan.append(("order", t, {k: crit[k] for k in perm}))
        else:
            keys = [str(x) for x in crit]
            perm = keys[:]
            for _ in range(5):
                rng.shuffle(perm)
                if perm != keys:
                    break
            plan.append(("order", t, perm))
    for t in cand[n_order:n_order + (n_pairs - n_order)]:
        plan.append(("repeat", t, t["question"]["criteria"]))

    out, lock = {}, threading.Lock()

    def one(i):
        kind, t, crit = plan[i]
        cli = Client(cfg["host"], cfg["port"], cfg["headers"], cfg["timeout"])
        qid = t["id"]
        res = []
        for c in (t["question"]["criteria"], crit):
            try:
                body = json.dumps({"model": cfg["model"], "state": t["state"],
                                   "questions": {qid: {"type": t["question"]["type"],
                                                       "instructions": t["question"]["instructions"],
                                                       "criteria": c}}},
                                  ensure_ascii=False).encode()
                st, raw = cli.post(cfg["endpoint"], body)
                if 200 <= st < 300:
                    ans = (json.loads(raw).get("answers") or {}).get(qid) or {}
                    res.append(to_probs(ans, t))
                else:
                    res.append(None)
            except Exception:  # noqa: BLE001
                res.append(None)
        with lock:
            out[i] = res

    ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(plan))]
    running = []
    for t in ths:
        t.start()
        running.append(t)
        if len(running) >= cfg["conc"]:
            running[0].join()
            running.pop(0)
    for t in running:
        t.join()

    rows = []
    for i, (kind, t, _) in enumerate(plan):
        a, b = (out.get(i) or [None, None])[:2]
        if not a or not b:
            continue
        keys = set(a) & set(b)
        if not keys:
            continue
        d = 0.5 * sum(abs(a[k] - b[k]) for k in keys)
        rows.append({"id": t["id"], "kind": kind, "tv": round(d, 4),
                     "flip": max(a, key=a.get) != max(b, key=b.get)})
    return rows


# ================================================================ 分析

def reliability(outs, bins=10):
    rows = [(o["conf"], o["correct"]) for o in outs if o.get("conf") is not None]
    if not rows:
        return []
    buckets = [[] for _ in range(bins)]
    for c, ok in rows:
        buckets[min(int(c * bins), bins - 1)].append((c, ok))
    table, total = [], len(rows)
    ece = 0.0
    for i, b in enumerate(buckets):
        if not b:
            continue
        mc = sum(x for x, _ in b) / len(b)
        ma = sum(1 for _, o in b if o) / len(b)
        ece += abs(mc - ma) * len(b) / total
        table.append((i / bins, (i + 1) / bins, len(b), mc, ma, abs(mc - ma)))
    return table, ece


def pctl(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(int(len(s) * p), len(s) - 1)]


def analyze(tasks, preds):
    """返回该数据集的完整分析；打分失败不抛异常，返回 error 让主流程继续。"""
    probs = {}
    for t in tasks:
        r = preds.get(t["id"])
        if r and r.get("ok"):
            probs[t["id"]] = r["probs"]
    try:
        result = official.score(tasks, [{"id": k, "probs": v} for k, v in probs.items()])
    except Exception as e:  # noqa: BLE001
        bad = [k for k, v in probs.items() if abs(sum(v.values()) - 1.0) > 0.01]
        return {"error": f"{type(e).__name__}: {e}", "bad": bad[:5],
                "latency": {}, "fails": 0, "errs": {}, "reliability": [],
                "result": {"capability": None, "intelligence": None, "calibration": None,
                           "accuracy_per_type": {}, "competence_per_type": {},
                           "calibration_per_type": {}, "predicted": len(probs),
                           "items": len(tasks)}}

    outs = []
    for t in tasks:
        try:
            outs.append(official.item_outcome(t, probs.get(t["id"])))
        except Exception:  # noqa: BLE001
            continue
    rel = reliability(outs)

    ms = sorted(r["ms"] for r in preds.values() if r.get("ms"))
    fails = [r for r in preds.values() if not r.get("ok")]
    from collections import Counter
    errs = Counter(r.get("error") for r in fails)

    return {"result": result, "reliability": rel, "latency": {
        "p50": pctl(ms, 0.5), "p95": pctl(ms, 0.95), "n": len(ms)},
        "fails": len(fails), "errs": dict(errs)}


# ================================================================ 报告

def fmt(x, nd=1, suffix=""):
    return "—" if x is None else f"{x:.{nd}f}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.ini"))
    ap.add_argument("--base")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--dataset", help="只跑某一个文件名")
    ap.add_argument("--limit", type=int, default=0, help="每个数据集最多多少条")
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--out-dir")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--pairs", type=int, default=-1, help="每个数据集的一致性对数")
    a = ap.parse_args()

    cp = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(a.config):
        raise SystemExit(f"找不到配置：{a.config}")
    cp.read(a.config, encoding="utf-8")

    base = a.base or cget(cp, "api", "base", "http://127.0.0.1:9900")
    endpoint = cget(cp, "api", "endpoint", "/v1/systemone")
    model = a.model or cget(cp, "api", "model", "")
    key = a.api_key or os.environ.get("JEV_API_KEY") or cget(cp, "api", "api_key", "")
    ds_dir = cget(cp, "run", "datasets", "datasets")
    ds_dir = ds_dir if os.path.isabs(ds_dir) else os.path.join(HERE, ds_dir)
    out_root = a.out_dir or cget(cp, "run", "out_dir", "results")
    out_root = out_root if os.path.isabs(out_root) else os.path.join(HERE, out_root)
    limit = a.limit or int(cget(cp, "run", "limit", "0"))
    pairs = a.pairs if a.pairs >= 0 else int(cget(cp, "run", "consistency_pairs", "100"))
    bins = int(cget(cp, "run", "bins", "10"))
    conc = a.concurrency or int(cget(cp, "api", "concurrency", "8"))
    timeout = float(cget(cp, "api", "timeout", "120"))

    u = urllib.parse.urlparse(base)
    if not u.hostname:
        raise SystemExit(f"base 不合法：{base}")
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if not model:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=8) as r:
                j = json.loads(r.read().decode("utf-8", "replace"))
            it = (j.get("data") or j.get("models") or [])
            model = (it[0].get("id") or it[0].get("name")) if it else ""
        except Exception:  # noqa: BLE001
            pass
    if not model:
        raise SystemExit("无法确定 model：请在 config.ini 的 [api] model 填写")

    files = sorted(f for f in os.listdir(ds_dir) if f.endswith(".jsonl"))
    if a.dataset:
        files = [f for f in files if f == a.dataset]
    if not files:
        raise SystemExit(f"{ds_dir} 下没有 .jsonl")

    slug = re.sub(r"[^\w.\-]+", "_", model)[:60]
    outdir = os.path.join(out_root, slug)
    os.makedirs(os.path.join(outdir, ".state"), exist_ok=True)

    cfg = {"host": u.hostname, "port": u.port or 8000, "endpoint": endpoint,
           "headers": headers, "timeout": timeout, "conc": conc, "model": model}

    print(SEP)
    print("Jev 数据集测评")
    print(f"  模型 : {model}")
    print(f"  服务 : {base}{endpoint}   并发 {conc}")
    print(f"  数据 : {ds_dir}（{len(files)} 个文件）")
    print(f"  输出 : {outdir}")
    print(SEP)

    all_res = []
    for name in files:
        print(f"\n[{name}]")
        tasks = load_dataset(os.path.join(ds_dir, name), limit)
        if not tasks:
            print("      跳过：没有可解析的样本")
            continue
        print(f"      {len(tasks)} 条")
        st_path = os.path.join(outdir, ".state", f"{name}.json")
        preds = predict(tasks, cfg, st_path, fresh=a.fresh)
        ana = analyze(tasks, preds)
        cons = consistency(tasks, cfg, pairs, random.Random(20261110)) if pairs else []
        all_res.append({"name": name, "n": len(tasks), "analyze": ana, "cons": cons})
        r = ana["result"]
        print(f"      Capability {fmt(r['capability'])}  Intelligence {fmt(r['intelligence'])}  "
              f"Calibration {fmt(r['calibration'])}  |  预测 {r['predicted']}/{r['items']}")
        if ana.get("error"):
            print(f"      ! 打分失败：{ana['error']}")
            if ana.get("bad"):
                print(f"        概率异常的条目：{ana['bad']}")

    if not all_res:
        raise SystemExit("没有产出任何结果")

    # ---- 汇总
    def agg(key):
        vals = [x["analyze"]["result"][key] for x in all_res
                if x["analyze"]["result"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    md = [f"# {model} 测评报告", "",
          f"> 数据集 {len(all_res)} 个 ｜ 目标 {base}{endpoint} ｜ 并发 {conc} ｜ "
          f"打分 = OpenDecisions 官方 scorer v1.6", "",
          "## 0. 总览", "",
          "| 数据集 | 条数 | Capability | Intelligence | Calibration | 一致性TV | 覆盖 |",
          "|---|---|---|---|---|---|---|"]
    for x in all_res:
        r = x["analyze"]["result"]
        tv = (sum(c["tv"] for c in x["cons"]) / len(x["cons"])) if x["cons"] else None
        md.append(f"| {x['name']} | {x['n']} | **{fmt(r['capability'])}** | "
                  f"{fmt(r['intelligence'])} | {fmt(r['calibration'])} | "
                  f"{'—' if tv is None else f'{tv:.4f}'} | {r['predicted']}/{r['items']} |")
    md += ["", f"**跨数据集平均**：Capability {fmt(agg('capability'))} ｜ "
               f"Intelligence {fmt(agg('intelligence'))} ｜ "
               f"Calibration {fmt(agg('calibration'))}", ""]

    for x in all_res:
        r = x["analyze"]["result"]
        md += ["", f"## {x['name']}", "",
               "| 指标 | 值 |", "|---|---|",
               f"| Capability | **{fmt(r['capability'])}** |",
               f"| Intelligence | {fmt(r['intelligence'])} |",
               f"| Calibration | {fmt(r['calibration'])} |",
               f"| 预测覆盖 | {r['predicted']}/{r['items']} |",
               f"| 延迟 p50 / p95 | {x['analyze']['latency']['p50']:.0f} / "
               f"{x['analyze']['latency']['p95']:.0f} ms |",
               f"| 失败 | {x['analyze']['fails']}"
               + (f"（{x['analyze']['errs']}）" if x['analyze']['errs'] else "")
               + " |", "",
               "| 题型 | 准确率 | 能力 | 校准 |", "|---|---|---|---|"]
        for t in ("choice", "noul", "score"):
            if t in r["accuracy_per_type"]:
                md.append(f"| {t} | {r['accuracy_per_type'][t]:.1f}% | "
                          f"{fmt(r['competence_per_type'].get(t))} | "
                          f"{fmt(r['calibration_per_type'].get(t))} |")
        if r.get("calibration_items_tvd") is not None:
            md += ["", f"真概率题（{r['calibration_items']} 条）平均 TVD："
                       f"**{r['calibration_items_tvd']:.3f}**"]
        if x["analyze"].get("error"):
            md += ["", f"> ⚠️ 打分失败：`{x['analyze']['error']}`",
                   f"> 概率异常的条目（前 5 个）：{x['analyze'].get('bad')}",
                   "> 说明这些条目的返回概率没有落到该题的 labels 上。"]

        rel = x["analyze"]["reliability"]
        if rel and rel[0]:
            table, ece = rel
            md += ["", "**可靠性**（置信度 vs 实际准确率）", "",
                   "| 置信区间 | n | 平均置信 | 实际准确 | 偏差 |", "|---|---|---|---|---|"]
            for lo, hi, n, mc, ma, gap in table:
                md.append(f"| {lo:.1f}–{hi:.1f} | {n} | {mc:.3f} | {ma:.3f} | {gap:.3f} |")
            md += ["", f"ECE = {ece:.4f}"]

        if x["cons"]:
            tvs = [c["tv"] for c in x["cons"]]
            order = [c["tv"] for c in x["cons"] if c["kind"] == "order"]
            rep = [c["tv"] for c in x["cons"] if c["kind"] == "repeat"]
            md += ["", "**一致性**", "", "| 变体 | 对数 | TV 均值 | 通过率(≤0.05) |",
                   "|---|---|---|---|"]
            if order:
                md.append(f"| 换序 | {len(order)} | {sum(order)/len(order):.4f} | "
                          f"{sum(1 for v in order if v <= 0.05)/len(order):.1%} |")
            if rep:
                md.append(f"| 重复 | {len(rep)} | {sum(rep)/len(rep):.4f} | "
                          f"{sum(1 for v in rep if v <= 0.05)/len(rep):.1%} |")
            md += ["", f"整体 TV 均值 {sum(tvs)/len(tvs):.4f}，"
                       f"翻转率 {sum(1 for c in x['cons'] if c['flip'])/len(tvs):.1%}"]

    md += ["", "## 口径", "",
           "- 三项分数由官方 `od/score.py` 计算，本包未改一行。",
           "- Capability 不是准确率：Intelligence 已扣除随机猜对的基线，并按难度加权。",
           "- Calibration 的 choice 只统计 hard 档，缺 hard 档时该题型不参与。",
           "- 没预测的条目按错处理（官方约定）。", ""]

    json.dump({"model": model,
               "datasets": [{"name": x["name"], "n": x["n"],
                             "result": x["analyze"]["result"],
                             "latency": x["analyze"]["latency"],
                             "fails": x["analyze"]["fails"],
                             "errs": x["analyze"]["errs"],
                             "reliability": x["analyze"]["reliability"][0]
                             if x["analyze"]["reliability"] else [],
                             "consistency": x["cons"]} for x in all_res]},
              open(os.path.join(outdir, f"{slug}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    with open(os.path.join(outdir, f"{slug}.csv"), "w", encoding="utf-8") as f:
        f.write("dataset,n,capability,intelligence,calibration,coverage,consistency_tv\n")
        for x in all_res:
            r = x["analyze"]["result"]
            tv = (sum(c["tv"] for c in x["cons"]) / len(x["cons"])) if x["cons"] else ""
            f.write(f"{x['name']},{x['n']},{r['capability']},{r['intelligence']},"
                    f"{r['calibration']},{r['predicted']}/{r['items']},{tv}\n")

    rpt = os.path.join(outdir, f"{slug}.md")
    open(rpt, "w", encoding="utf-8").write("\n".join(md))
    print("\n" + "\n".join(md[:22]))
    print(f"\n报告：{rpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
