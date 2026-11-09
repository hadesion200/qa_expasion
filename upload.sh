curl -sS --max-time 10 http://127.0.0.1:18080/health
curl -sS --max-time 10 http://127.0.0.1:18080/v1/models


curl -sS -X POST "http://127.0.0.1:18080/v1/systemone" \
  -H "Content-Type: application/json" \
  --max-time 30 \
  --data-binary @- <<'JSON'
{
  "model": "sdec-4b-satellite-soft25-seed33",
  "state": "【任务】选择定轨方法。【依据】仅有单站测距数据，观测弧长 8 分钟，无测速数据，无法进行力学模型精化。【证据】历史同类任务中，短弧单站定轨的收敛率不足 40%。",
  "questions": {
    "q0": {
      "type": "choice",
      "instructions": "在仅有单站测距数据的条件下，应优先采用哪种定轨方法？",
      "criteria": {"c0": "批处理最小二乘", "c1": "扩展卡尔曼滤波", "c2": "无迹卡尔曼滤波", "c3": "序贯单点定位"}
    }
  }
}
JSON