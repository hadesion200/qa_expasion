# -*- coding: utf-8 -*-
"""命令行入口。

  python3 run.py judge      [--limit N] [--mock]   逐条判定
  python3 run.py repair     [--limit N] [--mock]   只改写【任务】行（可修复件）
  python3 run.py rejudge    [--limit N] [--mock]   修复件复核
  python3 run.py assemble                          汇总出最终文件
  python3 run.py all        [--limit N] [--mock]   依次跑完
  python3 run.py report                            只看统计
"""
import argparse
import asyncio
import json
import sys

import config
import pipeline
import prompts
from llm import Teacher, run_batch


def _judge_jobs(items, limit):
    jobs = [(it["id"], pipeline.judge_fn(it)) for it in items]
    return jobs[:limit] if limit else jobs


async def cmd_judge(args):
    items = pipeline.load_items()
    jobs = _judge_jobs(items, args.limit)
    async with Teacher(mock=args.mock) as t:
        await run_batch(t, jobs, config.OUT / "judge.jsonl",
                        lambda t, rid, text: t.ask("judge", prompts.SYSTEM, text))


def _load(path):
    if not path.exists():
        return {}
    return {r["id"]: r["card"] for r in pipeline.read_jsonl(path)}


async def cmd_repair(args):
    items = {it["id"]: it for it in pipeline.load_items()}
    judged = _load(config.OUT / "judge.jsonl")
    todo = []
    for rid, it in items.items():
        c = judged.get(rid)
        if not c or int(c.get("space", 0)) < config.SPACE_MIN:
            continue
        ok, _ = pipeline.judge_ok(c, it)
        if ok or not c.get("fixable"):
            continue
        if c.get("issue") not in ("B", "C", "none"):
            continue
        todo.append((rid, it))
    todo.sort()
    if args.limit:
        todo = todo[:args.limit]
    print(f"可修复候选 {len(todo)} 条")
    if not todo:
        return

    async def one(t, rid, it):
        old_task, rest = pipeline.state_rest(it["state"])
        if old_task is None:
            return {"id": rid, "ok": False, "why": "no_task_line"}
        text = prompts.REPAIR.format(task=old_task.replace("【任务】", "")[:400],
                                     rest=rest[:1200])
        card = await t.ask("repair", prompts.SYSTEM, text)
        if not card:
            return {"id": rid, "ok": False, "why": "no_card"}
        new_task = "【任务】" + str(card.get("task", "")).strip()
        ok, why = pipeline.repair_guard(old_task, new_task, it["state"])
        return {"id": rid, "ok": ok, "why": why, "task": new_task,
                "new_state": new_task + rest if ok else None,
                "old_task": old_task}

    async with Teacher(mock=args.mock) as t:
        await run_batch(t, todo, config.OUT / "repair.jsonl",
                        lambda t, rid, it: one(t, rid, it))


async def cmd_rejudge(args):
    items = {it["id"]: it for it in pipeline.load_items()}
    rep = _load(config.OUT / "repair.jsonl")
    todo = [(rid, r) for rid, r in rep.items() if r.get("ok") and r.get("new_state")]
    todo.sort()
    if args.limit:
        todo = todo[:args.limit]
    print(f"待复核 {len(todo)} 条")
    if not todo:
        return

    async def one(t, rid, r):
        it = dict(items[rid])
        it["state"] = r["new_state"]
        return await t.ask("judge", prompts.SYSTEM, pipeline.judge_fn(it))

    async with Teacher(mock=args.mock) as t:
        await run_batch(t, todo, config.OUT / "rejudge.jsonl",
                        lambda t, rid, r: one(t, rid, r))


def cmd_assemble(args):
    # 允许不改判定、只放宽阈值重新汇总（不产生新的 API 调用）
    if getattr(args, "min_score", None) is not None:
        config.MIN_SCORE = args.min_score
    if getattr(args, "space_min", None) is not None:
        config.SPACE_MIN = args.space_min
    return pipeline.main()


def cmd_report(_args):
    p = config.OUT / "report.json"
    if not p.exists():
        print("还没有 report.json，先跑 assemble")
        return 1
    print(p.read_text(encoding="utf-8"))
    return 0


def main():
    KNOWN = ("judge", "repair", "rejudge", "assemble", "all", "report")
    if len(sys.argv) > 1 and sys.argv[1] not in KNOWN + ("-h", "--help"):
        print(f"[error] 未知子命令 {sys.argv[1]!r}\n"
              f"        可选：{', '.join(KNOWN)}\n"
              f"        例：python3 run.py judge --limit 300          # 直接调 python\n"
              f"            ./run.sh judge --limit 300                # 用脚本")
        return 2
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in KNOWN:
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--mock", action="store_true")
        p.add_argument("--min-score", type=int, default=None, dest="min_score")
        p.add_argument("--space-min", type=int, default=None, dest="space_min")
    args = ap.parse_args()

    if args.cmd == "judge":
        asyncio.run(cmd_judge(args))
    elif args.cmd == "repair":
        asyncio.run(cmd_repair(args))
    elif args.cmd == "rejudge":
        asyncio.run(cmd_rejudge(args))
    elif args.cmd == "assemble":
        return cmd_assemble(args)
    elif args.cmd == "report":
        return cmd_report(args)
    else:
        asyncio.run(cmd_judge(args))
        asyncio.run(cmd_repair(args))
        asyncio.run(cmd_rejudge(args))
        return cmd_assemble(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
