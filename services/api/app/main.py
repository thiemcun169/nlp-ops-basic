# -*- coding: utf-8 -*-
"""Điểm khởi động ứng dụng FastAPI.

File này CHỈ lắp ráp: cấu hình, router, thư mục tĩnh. Không có logic nghiệp vụ.
Giữ nó mỏng là có chủ ý -- mọi thứ đáng test đều nằm ở nơi test được mà không
cần dựng cả một web server.

Phần `description` bên dưới hiện ngay đầu trang tài liệu tự sinh tại `/docs`,
nên nó được viết cho người sắp tích hợp đọc, không phải cho lập trình viên của
chính dịch vụ này.
"""
import logging
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import settings
from app.routers import health, ner

logging.getLogger("uvicorn.error").setLevel(logging.INFO)

DESCRIPTION = """
Nhận diện thực thể có tên (NER) cho tiếng Việt: tên người (PER), tổ chức (ORG),
địa điểm (LOC).

Cùng một endpoint chạy được nhiều model, chọn bằng trường `model` trong body.
Tên model tự host thì chạy tại chỗ, tên nào khác thì tự đẩy sang nhà cung cấp LLM:

| model | Chạy ở đâu | Dùng khi nào |
|---|---|---|
| `phobert-ner-gpu` | Triton, GPU | mặc định khi chạy thật |
| `phobert-ner-cpu` | Triton, CPU | đo xem GPU đáng bao nhiêu tiền |
| `phobert-ner-local` | ngay trong tiến trình API | dự phòng khi Triton hỏng |
| `groq/compound`, ... | nhà cung cấp LLM | đối chứng, xử lý câu lạ |

**Bắt đầu từ đâu:** gọi `GET /v1/models` để biết cái nào đang sống, rồi
`POST /v1/ner`. Muốn so hai model trên cùng một câu thì dùng `POST /v1/ner/compare`.

Vị trí thực thể trả về theo offset ký tự, bảo đảm `text[start:end]` đúng bằng
`entity.text` -- tô màu trên giao diện không cần so khớp chuỗi.

Hợp đồng dữ liệu đầy đủ và quy tắc thay đổi phiên bản: `docs/API_CONTRACT.md`.
"""

app = FastAPI(
    title="Vietnamese NER Service",
    description=DESCRIPTION,
    version="1.0.0",
    contact={"name": "Nguyễn Bá Thiêm"},
)

app.include_router(health.router)
app.include_router(ner.router)


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Số đo của LỚP ỨNG DỤNG, theo định dạng Prometheus.

    Tách bạch với số đo của Triton ở cổng 8002: Triton chỉ biết phần việc của nó
    (nhận tensor tới trả tensor), còn ở đây đo cả tokenize, gom span và thời gian
    FastAPI xử lý. Chỉ nhìn một trong hai là thấy thiếu.

    Không đưa vào tài liệu OpenAPI vì đây là endpoint vận hành, không phải một
    phần của hợp đồng API.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def web_ui():
    """Trang demo. Không đưa vào tài liệu OpenAPI vì nó là giao diện, không phải API."""
    return FileResponse(os.path.join(settings.STATIC_DIR, "index.html"))
