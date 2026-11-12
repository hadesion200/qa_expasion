BASE=http://127.0.0.1:8000

# 1. 进程活着
curl -fsS $BASE/health

# 2. 模型已加载
curl -fsS $BASE/v1/models

# 3. 路由和字段是否存在（未关闭 OpenAPI 时）
curl -fsS $BASE/openapi.json > /tmp/openapi.json
jq '.paths["/v1/responses"]' /tmp/openapi.json
jq '.components.schemas.ResponsesRequest.properties.instructions' /tmp/openapi.json

# 4. 真正打一次带 instructions 的请求
curl -sS -o /tmp/resp.json -w '%{http_code}\n' \
  -X POST $BASE/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev",
    "instructions": "只输出 OK",
    "input": "ping",
    "max_output_tokens": 8
  }'