# -*- coding: utf-8 -*-
"""Health check: thứ đầu tiên mọi hệ thống chạy thật cần có.

Docker Compose và Kubernetes dựa vào endpoint này để quyết định có đẩy traffic
vào container hay không, nên nó phải trả lời "dịch vụ PHỤC VỤ ĐƯỢC chưa", không
phải "tiến trình còn sống chưa".

Ba mức, không phải hai:
    ok        model mặc định phục vụ được
    degraded  model mặc định hỏng nhưng đường dự phòng còn chạy
    down      không model tự host nào còn sống

Mức `degraded` là mức đáng giá nhất: hệ thống vẫn trả lời người dùng nhưng đang
chạy bằng đường phụ, và ai đó cần biết điều đó TRƯỚC khi tải tăng lên.
"""
from fastapi import APIRouter

from app.backends import llm_backend, registry
from app.config import settings
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse,
            summary="Trạng thái phục vụ của dịch vụ")
def health():
    # Hỏi mỗi nơi chạy ĐÚNG MỘT LẦN rồi tra lại từ bảng. Gọi is_ready() nhiều lần
    # nghĩa là nhiều lượt gọi mạng tới Triton cho mỗi lần health check, mà
    # orchestrator gọi endpoint này vài giây một lần.
    runtimes = {rt.name: rt.is_ready() for rt in registry.RUNTIMES}

    default_ok = runtimes.get(registry.route(registry.default_model()).name, False)
    fallback_ok = bool(settings.FALLBACK_MODEL) and runtimes.get(
        registry.route(settings.FALLBACK_MODEL).name, False)
    status = "ok" if default_ok else ("degraded" if fallback_ok else "down")

    llm_model = None
    if runtimes.get("llm"):
        try:
            llm_model = llm_backend.resolve_model()
        except Exception:
            llm_model = None      # không dò được model không phải lý do để health sập

    return HealthResponse(
        status=status,
        default_model=registry.default_model(),
        runtimes=runtimes,
        llm_model=llm_model,
    )
