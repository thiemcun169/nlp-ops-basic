#!/usr/bin/env bash
# Chạy đo tải trên NHIỀU backend rồi in bảng so sánh p50/p90/p99/thông lượng.
#
# Ý nghĩa: cùng một model, cùng một API, chỉ khác chỗ chạy suy luận. Chênh lệch
# đo được vì thế đúng là chênh lệch của phần cứng và của máy chủ suy luận, không
# lẫn yếu tố nào khác.
#
# Ví dụ:
#   bash load_test/run_stress_test.sh                          # 80 user, 30 giây
#   USERS=150 RUN_TIME=60s bash load_test/run_stress_test.sh
#   MODELS="phobert-ner-gpu" USERS=200 bash load_test/run_stress_test.sh
#
# KHÔNG dùng `set -e`: locust thoát với mã 1 khi có DÙ CHỈ MỘT request lỗi. Đó là
# thông tin cần đọc tiếp, không phải lý do để script dừng ngang. Đây là lỗi thật
# đã mắc phải khi viết script này.
set -uo pipefail
cd "$(dirname "$0")"

HOST="${API_HOST:-http://localhost:8080}"
USERS="${USERS:-80}"
SPAWN_RATE="${SPAWN_RATE:-20}"
RUN_TIME="${RUN_TIME:-30s}"
MODELS="${MODELS:-phobert-ner-gpu phobert-ner-cpu}"

if ! command -v locust >/dev/null 2>&1; then
    echo "Chưa có locust. Cài bằng:  pip install locust"
    exit 1
fi

echo "== Kiểm tra hệ thống trước khi đo =="
if ! curl -sf "$HOST/health" > /tmp/nlpops_health.json; then
    echo "Không gọi được $HOST/health. Hệ thống đã chạy chưa? (docker compose up -d)"
    exit 1
fi
python3 -m json.tool < /tmp/nlpops_health.json

for model in $MODELS; do
    echo
    echo "=========================================================================="
    echo "Đo tải model: $model"
    echo "   $USERS user song song, tăng dần $SPAWN_RATE user/giây, chạy $RUN_TIME"
    echo "=========================================================================="
    NER_MODEL="$model" locust -f locustfile.py --headless \
        -u "$USERS" -r "$SPAWN_RATE" --run-time "$RUN_TIME" \
        --host "$HOST" --csv "results_${model}" --only-summary
done

echo
python3 summarize_results.py $MODELS
