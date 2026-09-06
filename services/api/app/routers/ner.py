# -*- coding: utf-8 -*-
"""Các endpoint NER.

Router chỉ làm bốn việc: đọc request, gọi model, đổi lỗi thành mã HTTP đúng,
trả kết quả. Không có logic model nào ở đây (phần đó ở `app/backends/`), nên
thay Triton bằng thứ khác không phải đụng vào tầng HTTP.

Mã lỗi dùng trong dịch vụ này:
    400  người gọi gửi sai nội dung (văn bản chỉ có khoảng trắng)
    422  FastAPI tự trả khi body sai kiểu hoặc thiếu trường
    502  dịch vụ ngoài (LLM) trả lời lỗi
    503  nơi chạy model không phục vụ được và không có đường dự phòng
Chọn đúng mã lỗi không phải chuyện hình thức: người tích hợp dựa vào nó để quyết
định thử lại hay dừng hẳn.
"""
import logging
import time

from fastapi import APIRouter, HTTPException

from app.backends import registry
from app.config import settings
from app.core import agreement
from app.metrics import E2E, REQUESTS
from app.schemas import (Agreement, CompareRequest, CompareResponse, ModelListResponse,
                         ModelResult, NERRequest, NERResponse)

logger = logging.getLogger("uvicorn.error")
router = APIRouter(prefix="/v1", tags=["ner"])


@router.get("/models", response_model=ModelListResponse,
            summary="Liệt kê các model gọi được")
def list_models():
    """Model tự host, cộng với model của nhà cung cấp LLM nếu đã cấu hình key.

    Client nên gọi endpoint này thay vì viết cứng danh sách model: thêm bớt
    model ở máy chủ sẽ không làm hỏng client.
    """
    return ModelListResponse(
        default=registry.default_model(),
        fallback=settings.FALLBACK_MODEL or None,
        models=registry.describe(),
    )


def _run(text: str, model: str, options: dict):
    """Gọi model và quy mọi loại lỗi về đúng mã HTTP."""
    try:
        return registry.predict(text, model, options)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        if not registry.is_local(model):
            logger.warning("Dịch vụ LLM lỗi (model=%s): %s", model, e)
            raise HTTPException(502, f"Dịch vụ LLM lỗi: {e}") from e
        logger.exception("Model %s thất bại, text=%r", model, text[:120])
        raise HTTPException(503, f"Model '{model}' không phục vụ được: {e}") from e


@router.post("/ner", response_model=NERResponse, summary="Nhận diện thực thể có tên")
def ner(req: NERRequest):
    """Chạy NER trên một văn bản.

    Chọn model bằng trường `model` trong body, không phải bằng đường dẫn riêng.
    Đổi model lõi khi đó chỉ là đổi một chuỗi, client không phải sửa URL, và một
    request A/B test chỉ khác nhau đúng một trường.

        {"text": "...", "model": "phobert-ner-gpu"}   -> chạy tại chỗ trên GPU
        {"text": "...", "model": "groq/compound"}     -> đẩy sang nhà cung cấp LLM
    """
    if not req.text.strip():
        raise HTTPException(400, "Trường 'text' chỉ chứa khoảng trắng.")

    model = req.model or registry.default_model()

    # Đo TOÀN BỘ thời gian trong handler, không chỉ phần gọi model. Hiệu số giữa
    # con số này và thời gian suy luận chính là chi phí của lớp ứng dụng.
    start = time.perf_counter()
    try:
        prediction, runtime = _run(req.text, model, req.options.model_dump())
    except HTTPException as e:
        REQUESTS.labels(endpoint="/v1/ner", model=model, status=str(e.status_code)).inc()
        raise
    E2E.labels(endpoint="/v1/ner", model=model).observe(time.perf_counter() - start)
    REQUESTS.labels(endpoint="/v1/ner", model=model, status="200").inc()

    return NERResponse(
        text=req.text,
        model=prediction.model,
        runtime=runtime,
        entities=prediction.entities,
        latency_ms=round(prediction.latency_ms, 2),
        meta=prediction.meta,
    )


@router.post("/ner/compare", response_model=CompareResponse,
             summary="Chạy cùng một văn bản qua nhiều model")
def compare(req: CompareRequest):
    """Chạy CÙNG một văn bản qua nhiều model và trả kết quả cạnh nhau.

    Một model hỏng không làm hỏng cả response: lỗi của nó nằm trong trường
    `error` của riêng nó, các model khác vẫn trả kết quả bình thường.

    Phần `agreements` so từng model với model ĐẦU TIÊN trong danh sách. Đó là
    mức khớp nhau, không phải độ chính xác: cả hai cùng sai giống nhau vẫn cho
    F1 bằng 1. Muốn độ chính xác thật thì dùng `benchmarks/`.
    """
    if not req.text.strip():
        raise HTTPException(400, "Trường 'text' chỉ chứa khoảng trắng.")

    names = req.models or [registry.default_model(), "groq/compound"]
    options = req.options.model_dump()
    results, entity_sets = [], {}

    for name in names:
        start = time.perf_counter()
        try:
            prediction, runtime = registry.predict(req.text, name, options)
            E2E.labels(endpoint="/v1/ner/compare", model=name).observe(
                time.perf_counter() - start)
            REQUESTS.labels(endpoint="/v1/ner/compare", model=name, status="200").inc()
            results.append(ModelResult(
                model=name, runtime=runtime, entities=prediction.entities,
                latency_ms=round(prediction.latency_ms, 2), meta=prediction.meta))
            entity_sets[name] = prediction.entities
        except Exception as e:
            logger.warning("Model %s lỗi trong lúc so sánh: %s", name, e)
            REQUESTS.labels(endpoint="/v1/ner/compare", model=name, status="error").inc()
            results.append(ModelResult(model=name, error=str(e)[:400]))

    agreements = []
    if len(entity_sets) >= 2:
        reference = next(iter(entity_sets))
        for name, entities in entity_sets.items():
            if name == reference:
                continue
            scores = agreement.compare(entity_sets[reference], entities, req.text)
            agreements.append(Agreement(reference=reference, candidate=name, **scores))

    return CompareResponse(text=req.text, results=results, agreements=agreements)
