# -*- coding: utf-8 -*-
"""把 out/（v1）与 out_v2/（v2）各自的 class1 / class2 合并成一套数据。

  python merge_out.py                    # 合并到 out_merged/
  python merge_out.py --out out_merged --keep-split
  python merge_out.py --classes 1        # 只合并第一类

合并规则：
  1. 按 (tree, kind, state, options) + id 双重去重，v1 与 v2 的重复项只留一条；
  2. 每条优先用「已经过 assemble 的最终记录」（带模型场景），没有就用 items + scenario 现场补；
  3. 默认**按树重新切分**（90/5/5），保证同一棵树的题目不跨 train/dev/benchmark；
     `--keep-split` 则沿用各自原来的 split；
  4. 重写 items.jsonl / scenario.jsonl / solve.jsonl / train·dev·benchmark / report.json。
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
SPLITS = ("train", "dev", "benchmark")


def read_jsonl(p):
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def fingerprint(it):
    return (it["meta"]["tree"], it["kind"], it["state"], tuple(it["options"]))


def split_by_tree(tree):
    h = int(hashlib.md5(tree.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 90:
        return "train"
    return "dev" if h < 95 else "benchmark"


def load_version(root, cls):
    """读取一个版本某类的全部材料。返回 dict。"""
    d = root / f"class{cls}"
    if not d.exists():
        return None
    items = list(read_jsonl(d / "items.jsonl"))
    scen = {r["id"]: r["card"] for r in read_jsonl(d / "scenario.jsonl")}
    sols = defaultdict(list)
    for r in read_jsonl(d / "solve.jsonl"):
        sols[r["id"].split("#")[0]].append(r["card"])
    final, final_split = {}, {}
    for s in SPLITS:
        for it in read_jsonl(d / f"{s}.jsonl"):
            final[it["id"]] = it
            final_split[it["id"]] = s
    return {"dir": d, "items": items, "scen": scen, "sols": dict(sols),
            "final": final, "final_split": final_split}


def apply_scenario(it, scenario):
    i = it["state"].find("\n")
    head, rest = (it["state"], "") if i < 0 else (it["state"][:i], it["state"][i:])
    new = dict(it)
    new["state"] = "【场景】" + scenario.strip() + rest
    new["llm_scenario"] = True
    return new


def merge_class(v1, v2, cls, keep_split):
    seen_fp, seen_id = {}, set()
    merged, src_stats = [], Counter()

    # 先 v2 再 v1：v2 是扩量版，优先保留它的场景与 verified 标记
    for tag, data in (("v2", v2), ("v1", v1)):
        if not data:
            continue
        for it in data["items"]:
            fp = fingerprint(it)
            rid = it["id"]
            if fp in seen_fp or rid in seen_id:
                src_stats[f"{tag}_dup"] += 1
                continue
            seen_fp[fp] = tag
            seen_id.add(rid)

            rec = data["final"].get(rid)
            if rec is not None:
                out = dict(rec)
            else:
                card = data["scen"].get(rid)
                out = apply_scenario(it, card["scenario"]) if card else dict(it)
                out["verified"] = False

            out.setdefault("split", data["final_split"].get(rid) or it.get("split", "train"))
            out["src_version"] = tag
            merged.append(out)
            src_stats[f"{tag}_kept"] += 1

    if not keep_split:
        for it in merged:
            it["split"] = split_by_tree(it["meta"]["tree"])

    # 同一棵树只能落在一侧
    by_tree = defaultdict(set)
    for it in merged:
        by_tree[it["meta"]["tree"]].add(it["split"])
    leakage = {t: s for t, s in by_tree.items() if len(s) > 1}
    if leakage:                       # 极少见：强制归到 train
        for it in merged:
            if it["meta"]["tree"] in leakage:
                it["split"] = "train"

    # scenario / solve 合并（按 id 去重）
    scen_out, sols_out = {}, {}
    for data in (v1, v2):
        if not data:
            continue
        for rid, card in data["scen"].items():
            scen_out.setdefault(rid, card)
        for rid, cards in data["sols"].items():
            sols_out.setdefault(rid, cards)

    out_dir = MERGE_DIR / f"class{cls}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for it in merged:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    outs = {s: [] for s in SPLITS}
    for it in merged:
        outs[it["split"]].append(it)
    for s, rows in outs.items():
        with open(out_dir / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(out_dir / "scenario.jsonl", "w", encoding="utf-8") as f:
        for rid, card in scen_out.items():
            f.write(json.dumps({"id": rid, "card": card}, ensure_ascii=False) + "\n")
    with open(out_dir / "solve.jsonl", "w", encoding="utf-8") as f:
        for rid, cards in sols_out.items():
            for k, card in enumerate(cards):
                f.write(json.dumps({"id": f"{rid}#r{k + 1}", "card": card},
                                   ensure_ascii=False) + "\n")

    rep = {
        "class": cls,
        "items": len(merged),
        "trees": len(by_tree),
        "sources": dict(src_stats),
        "by_split": {s: len(v) for s, v in outs.items()},
        "by_kind": dict(Counter(x["kind"] for x in merged)),
        "by_gen": dict(Counter(x["meta"]["gen"] for x in merged)),
        "by_domain": dict(Counter(x["meta"]["domain"] for x in merged)),
        "by_tree_source": dict(Counter(x.get("tree_source", "?") for x in merged)),
        "has_scenario": sum(1 for x in merged if x.get("llm_scenario")),
        "verified": sum(1 for x in merged if x.get("verified")),
        "leakage_fixed": len(leakage),
        "split_mode": "keep" if keep_split else "by_tree",
    }
    (out_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return rep, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "out_merged"))
    ap.add_argument("--classes", default="1,2")
    ap.add_argument("--keep-split", action="store_true")
    ap.add_argument("--v1", default="")
    ap.add_argument("--v2", default="")
    args = ap.parse_args()

    global MERGE_DIR
    MERGE_DIR = Path(args.out)

    v1_root = Path(args.v1 or BASE / "out")
    v2_root = Path(args.v2 or BASE / "out_v2")

    for c in [int(x) for x in args.classes.split(",") if x.strip()]:
        v1 = load_version(v1_root, c)
        v2 = load_version(v2_root, c)
        if not v1 and not v2:
            print(f"[class{c}] 两个目录都没有数据，跳过")
            continue
        rep, out_dir = merge_class(v1, v2, c, args.keep_split)
        print(f"\n=== class{c} → {out_dir} ===")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
