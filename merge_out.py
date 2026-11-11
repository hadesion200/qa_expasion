# -*- coding: utf-8 -*-
"""把 out/（v1）与 out_v2/（v2）各自的 class1 / class2 合并成一套数据。

修正：只合并**最终保留**的记录（train/dev/benchmark），不再从 items.jsonl 全量合并，
否则会把 assemble 阶段被拒的条目重新捡回来（v1 1309 + v2 7176 应≈8485，不是 11000）。

  python merge_out.py                       # → out_merged/
  python merge_out.py --keep-split          # 沿用各自原来的 split
  python merge_out.py --include-unassembled # 没跑过 assemble 的版本退回 items+scenario

隔离：每条记录都会用 trees_pack 校验「树属于哪个 class」，不属于本类的直接丢弃并计数，
避免 class1 / class2 互相污染。
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
SPLITS = ("train", "dev", "benchmark")
MERGE_DIR = BASE / "out_merged"


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


def tree_class_map():
    """从 trees_pack 读 树名 → class，用于隔离校验。"""
    m = {}
    for d in read_jsonl(BASE / "data" / "trees_pack.jsonl"):
        m[d["name"]] = d["class"]
    return m


def load_version(root, cls):
    d = root / f"class{cls}"
    if not d.exists():
        return None
    final, final_split = {}, {}
    for s in SPLITS:
        for it in read_jsonl(d / f"{s}.jsonl"):
            final[it["id"]] = it
            final_split[it["id"]] = s
    return {
        "dir": d,
        "final": final,
        "final_split": final_split,
        "items": list(read_jsonl(d / "items.jsonl")),
        "scen": {r["id"]: r["card"] for r in read_jsonl(d / "scenario.jsonl")},
        "sols": {},
    }


def apply_scenario(it, scenario):
    i = it["state"].find("\n")
    head, rest = (it["state"], "") if i < 0 else (it["state"][:i], it["state"][i:])
    new = dict(it)
    new["state"] = "【场景】" + scenario.strip() + rest
    new["llm_scenario"] = True
    return new


def candidates(data, cls, include_unassembled, stats):
    """返回该版本该类的"应合并记录"。优先 final，其次（可选）items+scenario。"""
    if not data:
        return []
    if data["final"]:
        stats["from_final"] += len(data["final"])
        return list(data["final"].values())
    if not include_unassembled:
        stats["skipped_unassembled"] += 1
        return []
    out = []
    for it in data["items"]:
        card = data["scen"].get(it["id"])
        if not card:
            continue
        rec = apply_scenario(it, card["scenario"])
        rec["verified"] = False
        out.append(rec)
    stats["from_items"] += len(out)
    return out


def merge_class(v1, v2, cls, keep_split, include_unassembled, tcm):
    seen_fp, seen_id = set(), set()
    merged = []
    stats = Counter()

    for tag, data in (("v2", v2), ("v1", v1)):
        for it in candidates(data, cls, include_unassembled, stats):
            tree = it["meta"]["tree"]
            owner = tcm.get(tree)
            if owner is not None and owner != cls:
                stats[f"{tag}_cross_class"] += 1
                continue
            fp = fingerprint(it)
            rid = it["id"]
            if fp in seen_fp or rid in seen_id:
                stats[f"{tag}_dup"] += 1
                continue
            seen_fp.add(fp)
            seen_id.add(rid)
            out = dict(it)
            out.setdefault("split", (data or {}).get("final_split", {}).get(rid, "train"))
            out["src_version"] = tag
            merged.append(out)
            stats[f"{tag}_kept"] += 1

    total_in = len((v1 or {}).get("final", {})) + len((v2 or {}).get("final", {}))
    if merged and len(merged) > total_in:
        print(f"[WARN] 合并后 {len(merged)} 条 > 输入 {total_in} 条，逻辑异常，请检查")

    if not keep_split:
        for it in merged:
            it["split"] = split_by_tree(it["meta"]["tree"])

    by_tree = defaultdict(set)
    for it in merged:
        by_tree[it["meta"]["tree"]].add(it["split"])
    leakage = [t for t, s in by_tree.items() if len(s) > 1]
    for it in merged:
        if it["meta"]["tree"] in leakage:
            it["split"] = "train"

    scen_out, sols_out = {}, {}
    for data in (v1, v2):
        if not data:
            continue
        for rid, card in data["scen"].items():
            scen_out.setdefault(rid, card)

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

    rep = {
        "class": cls,
        "items": len(merged),
        "trees": len(by_tree),
        "sources": dict(stats),
        "by_split": {s: len(v) for s, v in outs.items()},
        "by_kind": dict(Counter(x["kind"] for x in merged)),
        "by_gen": dict(Counter(x["meta"]["gen"] for x in merged)),
        "by_domain": dict(Counter(x["meta"]["domain"] for x in merged)),
        "by_tree_source": dict(Counter(x.get("tree_source", "?") for x in merged)),
        "has_scenario": sum(1 for x in merged if x.get("llm_scenario")),
        "verified": sum(1 for x in merged if x.get("verified")),
        "leakage_fixed": len(leakage),
        "split_mode": "keep" if keep_split else "by_tree",
        "input_counts": {
            "v1_final": len((v1 or {}).get("final", {})),
            "v2_final": len((v2 or {}).get("final", {})),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return rep, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "out_merged"))
    ap.add_argument("--classes", default="1,2")
    ap.add_argument("--v1", default="")
    ap.add_argument("--v2", default="")
    ap.add_argument("--keep-split", action="store_true")
    ap.add_argument("--include-unassembled", action="store_true",
                    help="某版本没跑过 assemble 时，退回 items+scenario（默认不退回）")
    args = ap.parse_args()

    global MERGE_DIR
    MERGE_DIR = Path(args.out)
    v1_root = Path(args.v1) if args.v1 else BASE / "out"
    v2_root = Path(args.v2) if args.v2 else BASE / "out_v2"
    tcm = tree_class_map()
    print(f"树归属表 {len(tcm)} 棵；v1={v1_root} v2={v2_root}")

    for c in [int(x) for x in args.classes.split(",") if x.strip()]:
        v1, v2 = load_version(v1_root, c), load_version(v2_root, c)
        if not v1 and not v2:
            print(f"[class{c}] 两个目录都没有数据，跳过")
            continue
        rep, out_dir = merge_class(v1, v2, c, args.keep_split,
                                   args.include_unassembled, tcm)
        print(f"\n=== class{c} → {out_dir} ===")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
