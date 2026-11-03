#!/usr/bin/env bash
# jeval_filter — 语义筛选 + 改写 + 复核
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
export LANG="${LANG:-C.UTF-8}"
export PYTHONIOENCODING=utf-8

usage() {
  cat <<'EOF'
用法: ./run.sh <command> [--limit N] [--mock]

  check     自检（不联网，用 mock 跑通全流程）
  judge     逐条判定（21,098 次调用）
  repair    只改写【任务】行（可修复件）
  rejudge   修复件复核
  assemble  汇总出 train/dev/benchmark + rejected
  all       judge → repair → rejudge → assemble
  report    打印统计

环境变量:
  TEACHER_BASE_URL / TEACHER_MODEL / TEACHER_API_KEY / TEACHER_CONCURRENCY
  JEVAL_SRC   输入目录（默认 .../jeval_space/out）
  JEVAL_DST   输出目录（默认 ./out_clean）
  MIN_SCORE   质量分下限（默认 4）
  SPACE_MIN   2=只要明确卫星/航天（默认），1=允许邻近

建议先跑:
  ./run.sh judge --limit 300 && ./run.sh assemble   # 看留存率
  ./run.sh all                                      # 全量
EOF
}

case "${1:-help}" in
  help|-h|--help) usage ;;
  check)
    "$PY" selftest.py ;;
  judge|repair|rejudge|assemble|all|report)
    # 注意：这里不能 shift，否则把子命令吃掉，run.py 会收到 "--limit 300"
    "$PY" run.py "$@" ;;
  *) usage; exit 1 ;;
esac
