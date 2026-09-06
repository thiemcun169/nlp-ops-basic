# -*- coding: utf-8 -*-
"""Định tuyến: cho một tên model, tìm nơi chạy nó.

Người gọi chỉ gửi `model`. Quy tắc định tuyến gọn trong một câu:

    tên nằm trong bảng model tự host  ->  chạy tại chỗ
    còn lại                           ->  đẩy sang nhà cung cấp LLM

Nhờ vậy `{"model": "phobert-ner-gpu"}` và `{"model": "groq/compound"}` đi qua
cùng một endpoint, và thêm model mới ở nhà cung cấp thì không phải sửa gì cả.

Cách này cũng khiến việc so sánh trở nên rẻ: đổi một chuỗi trong body là đổi
model lõi, không phải đổi URL hay viết nhánh code riêng.
"""
import logging
from typing import Dict, List, Tuple

from app.backends.base import Prediction, Runtime
from app.backends.llm_backend import LLMRuntime
from app.backends.onnx_local_backend import OnnxLocalRuntime
from app.backends.triton_backend import TritonRuntime
from app.config import settings

logger = logging.getLogger("uvicorn.error")

TRITON_GPU = TritonRuntime(
    name="triton-gpu",
    model_name=settings.LOCAL_MODEL_GPU,
    triton_model=settings.TRITON_MODEL_GPU,
    description="PhoBERT tự huấn luyện, ONNX Runtime trên GPU qua Triton.")
TRITON_CPU = TritonRuntime(
    name="triton-cpu",
    model_name=settings.LOCAL_MODEL_CPU,
    triton_model=settings.TRITON_MODEL_CPU,
    description="Cùng file trọng số đó nhưng Triton chạy trên CPU.")
ONNX_LOCAL = OnnxLocalRuntime()
LLM = LLMRuntime()

RUNTIMES: List[Runtime] = [TRITON_GPU, TRITON_CPU, ONNX_LOCAL, LLM]

# Bảng model tự host. Chỉ những tên này mới chạy tại chỗ; mọi tên khác đi ra LLM.
_LOCAL: Dict[str, Runtime] = {}
for _rt in (TRITON_GPU, TRITON_CPU, ONNX_LOCAL):
    for _m in _rt.models():
        _LOCAL[_m] = _rt


def default_model() -> str:
    return settings.DEFAULT_MODEL


def route(model: str) -> Runtime:
    """Tên model -> nơi chạy nó."""
    return _LOCAL.get(model, LLM)


def is_local(model: str) -> bool:
    return model in _LOCAL


def describe() -> List[Dict]:
    """Danh sách model gọi được, kèm nơi chạy và trạng thái.

    Client nên gọi endpoint này thay vì viết cứng danh sách model: thêm bớt
    model ở máy chủ sẽ không làm hỏng client.
    """
    out = []
    for name, runtime in _LOCAL.items():
        out.append({"name": name, "runtime": runtime.name, "kind": runtime.kind,
                    "description": runtime.description, "ready": runtime.is_ready()})
    if LLM.is_ready():
        for name in LLM.models():
            out.append({"name": name, "runtime": LLM.name, "kind": LLM.kind,
                        "description": LLM.description, "ready": True})
    return out


def predict(text: str, model: str, options: Dict) -> Tuple[Prediction, str]:
    """Chạy dự đoán, tự chuyển sang model dự phòng nếu nơi chạy chính hỏng.

    Chỉ chuyển dự phòng cho model TỰ HOST. Lỗi của LLM không được âm thầm thay
    bằng kết quả của model khác: người gọi hỏi model nào thì phải nhận câu trả
    lời của model đó, hoặc một lỗi rõ ràng.

    Trả về (kết quả, tên nơi đã chạy). Khi đã chuyển dự phòng, `meta` ghi rõ
    điều đó: một sự cố hạ tầng trôi qua mà không ai biết sẽ quay lại vào lúc tệ hơn.
    """
    runtime = route(model)
    try:
        return runtime.predict(text, model, options), runtime.name
    except ValueError:
        raise                       # đầu vào sai: lỗi của người gọi, không dự phòng
    except Exception as primary_error:
        fallback = settings.FALLBACK_MODEL
        if (runtime.kind != "inference-server" or not fallback or fallback == model
                or not is_local(fallback)):
            raise

        fb_runtime = route(fallback)
        if not fb_runtime.is_ready():
            raise

        logger.warning("Model '%s' lỗi (%s). Chuyển sang dự phòng '%s'.",
                       model, primary_error, fallback)
        prediction = fb_runtime.predict(text, fallback, options)
        prediction.meta["fallback_from"] = model
        prediction.meta["fallback_reason"] = str(primary_error)[:300]
        return prediction, fb_runtime.name
