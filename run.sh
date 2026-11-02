#!/bin/sh
# 查询扩写数据集工作流一键运行
# 用法：sh run.sh [--mock] [--fresh] [--config file.yaml]

set -eu
cd "$(dirname "$0")" || exit 1

CONFIG="config.yaml"
MOCK=0
FRESH=0

usage() {
  cat <<'EOF'
用法：sh run.sh [--mock] [--fresh] [--config <file>]

  --mock           三个模型都使用内置 mock，不需要真实 API
  --fresh          清空中间结果后重新开始
  --config <file>  指定 YAML 配置（默认 config.yaml）
  -h, --help       显示帮助
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mock) MOCK=1; shift ;;
    --fresh) FRESH=1; shift ;;
    --config)
      [ "$#" -ge 2 ] || { echo "错误：--config 需要路径" >&2; exit 2; }
      CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "错误：未知参数 $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -f "$CONFIG" ] || { echo "错误：配置文件不存在：$CONFIG" >&2; exit 2; }

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PY="python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "错误：找不到 python，请先安装 Python 3.9+" >&2
  exit 1
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

"$PY" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("yaml", "requests") if importlib.util.find_spec(name) is None]
if missing:
    print("缺少依赖：" + ", ".join(missing), file=sys.stderr)
    print("请先安装：pip install PyYAML requests", file=sys.stderr)
    sys.exit(1)
PY

MOCK_ARG=""
[ "$MOCK" -eq 1 ] && MOCK_ARG="--mock"
FRESH_ARG=""
[ "$FRESH" -eq 1 ] && FRESH_ARG="--fresh"

exec "$PY" workflow.py --config "$CONFIG" $MOCK_ARG $FRESH_ARG all
