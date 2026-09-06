# -*- coding: utf-8 -*-
"""Số đo của LỚP ỨNG DỤNG, tách bạch với số đo của Triton.

Vì sao cần cả hai chứ không chỉ số đo Triton:

Triton chỉ biết phần việc của nó, tính từ lúc nhận tensor tới lúc trả tensor. Nó
không hề biết tới thời gian tokenize, thời gian gom nhãn BIO thành span, hay thời
gian FastAPI đọc và kiểm tra body. Nếu chỉ nhìn số đo Triton thì một hệ thống
chậm vì tokenize sẽ trông hoàn toàn khoẻ mạnh.

Ba nhóm số đo ở đây:

    nlpops_e2e_latency_seconds     toàn bộ thời gian trong FastAPI, từ lúc vào
                                   handler tới lúc có kết quả
    nlpops_stage_latency_seconds   bóc tách thành ba chặng: tiền xử lý, suy luận,
                                   hậu xử lý
    nlpops_requests_total          đếm request theo endpoint, model, trạng thái

Điểm quan trọng: đây là **Histogram**, không phải bộ đếm cộng dồn như Triton. Nhờ
vậy dựng được p50, p90, p99 THẬT bằng `histogram_quantile`. Số đo Triton chỉ cho
ra trung bình, mà trung bình che mất phần đuôi.

Hiệu số `e2e - inference` chính là chi phí của lớp ứng dụng. Không đo riêng thì
không có cách nào biết nên tối ưu model hay tối ưu code Python.
"""
import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

# Thang bucket chọn theo dải đã đo thật: model tự host khoảng 8 tới 60 ms, LLM
# vài giây. Bucket mặc định của thư viện dừng ở 10 giây và quá thưa ở vùng mili
# giây, dùng thẳng thì p50 của model tự host rơi hết vào một bucket.
_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

E2E = Histogram(
    "nlpops_e2e_latency_seconds",
    "Thời gian FastAPI xử lý một request, gồm cả tiền xử lý và hậu xử lý.",
    ["endpoint", "model"],
    buckets=_BUCKETS,
)

STAGE = Histogram(
    "nlpops_stage_latency_seconds",
    "Thời gian từng chặng: preprocess (chữ -> tensor), inference (gọi nơi chạy "
    "model), postprocess (nhãn -> span).",
    ["stage", "model"],
    buckets=_BUCKETS,
)

REQUESTS = Counter(
    "nlpops_requests_total",
    "Số request theo endpoint, model và kết quả.",
    ["endpoint", "model", "status"],
)


@contextmanager
def stage(name: str, model: str):
    """Đo một chặng xử lý.

    Dùng dạng context manager để thời gian luôn được ghi lại kể cả khi chặng đó
    ném lỗi: một chặng chậm rồi hỏng vẫn là thông tin cần thấy trên biểu đồ.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        STAGE.labels(stage=name, model=model).observe(time.perf_counter() - start)
