# -*- coding: utf-8 -*-
"""Hợp đồng dữ liệu (API contract) của dịch vụ.

Đây là file quan trọng nhất khi có người khác tích hợp với bạn. Mọi mô tả viết
ở đây sẽ tự động xuất hiện trong tài liệu OpenAPI tại `/docs` và trong file
`/openapi.json`, tức là:

- Frontend sinh được client TypeScript từ `/openapi.json`, không phải chép tay.
- Backend khác biết chính xác trường nào bắt buộc, kiểu gì, giới hạn bao nhiêu.
- Bạn không phải viết một tài liệu riêng rồi để nó lạc hậu so với mã nguồn.

Nguyên tắc khi sửa file này: **thêm trường thì thoải mái, đổi tên hay bỏ trường
là phá vỡ hợp đồng** với mọi ứng dụng đang gọi. Việc đó phải đi kèm một phiên
bản đường dẫn mới (`/v2/...`). Chi tiết: docs/API_CONTRACT.md.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import settings


# --------------------------------------------------------------------------
# Đầu vào
# --------------------------------------------------------------------------
class PredictOptions(BaseModel):
    """Tuỳ chọn cho một lần chạy. Bỏ trống hết cũng chạy được."""

    return_scores: bool = Field(
        False, description="Kèm độ tin cậy cho mỗi thực thể. Chỉ có với model tự host.")


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.MAX_TEXT_CHARS,
                      description="Văn bản tiếng Việt cần nhận diện thực thể.",
                      examples=["Nguyễn Bá Thiêm làm việc tại Google ở Hà Nội."])
    model: Optional[str] = Field(
        None, description="Tên model muốn dùng. Tên model tự host thì chạy tại chỗ, "
                          "tên nào khác được đẩy sang nhà cung cấp LLM. Bỏ trống thì "
                          "dùng mặc định. Danh sách: GET /v1/models.",
        examples=["phobert-ner-gpu"])
    options: PredictOptions = Field(default_factory=PredictOptions)


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.MAX_TEXT_CHARS,
                      description="Văn bản chạy qua TẤT CẢ model được liệt kê.")
    models: Optional[List[str]] = Field(
        None, description="Danh sách model cần so. Bỏ trống thì so model mặc định "
                          "với LLM.",
        examples=[["phobert-ner-gpu", "phobert-ner-cpu", "groq/compound"]])
    options: PredictOptions = Field(default_factory=PredictOptions)


# --------------------------------------------------------------------------
# Đầu ra
# --------------------------------------------------------------------------
class Entity(BaseModel):
    """Một thực thể, định vị bằng offset ký tự trong `text` gốc.

    Bất biến quan trọng: `text[start:end]` luôn bằng đúng trường `text` của thực
    thể (trừ khi `matched = false`). Nhờ vậy frontend tô màu bằng offset, không
    cần so khớp chuỗi -- so khớp chuỗi hỏng ngay khi từ dính dấu câu.
    """
    text: str = Field(..., description="Đoạn văn bản của thực thể.", examples=["Hà Nội"])
    label: str = Field(..., description="Loại thực thể: PER, ORG hoặc LOC.", examples=["LOC"])
    start: int = Field(..., description="Vị trí ký tự bắt đầu (tính từ 0). "
                                        "Bằng -1 nghĩa là không tìm thấy trong văn bản gốc.")
    end: int = Field(..., description="Vị trí ký tự kết thúc, không bao gồm.")
    score: Optional[float] = Field(
        None, description="Độ tin cậy trong khoảng 0-1. Chỉ có khi options.return_scores = true.")
    matched: Optional[bool] = Field(
        None, description="Chỉ có ở model LLM. False nghĩa là LLM trả về một cụm "
                          "không hề xuất hiện nguyên văn trong đầu vào.")


class NERResponse(BaseModel):
    text: str = Field(..., description="Văn bản gốc, trả lại nguyên vẹn để đối chiếu offset.")
    model: str = Field(..., description="Model đã thực sự chạy.", examples=["phobert-ner-gpu"])
    runtime: str = Field(..., description="Nơi đã chạy model đó: triton-gpu, triton-cpu, "
                                          "onnx-local hoặc llm.", examples=["triton-gpu"])
    entities: List[Entity]
    latency_ms: float = Field(..., description="Thời gian xử lý đo tại phía máy chủ, "
                                               "gồm cả tokenize và giải mã nhãn.")
    meta: Dict = Field(default_factory=dict,
                       description="Thông tin chẩn đoán: số từ, có bị cắt do vượt độ dài "
                                   "tối đa không, có phải kết quả từ đường dự phòng không.")


class ModelResult(BaseModel):
    """Kết quả của MỘT model trong phép so sánh.

    Có `error` riêng cho từng model là điều bắt buộc: một model hỏng thì những
    model còn lại vẫn phải trả kết quả. Toàn bộ request cùng chết vì một dịch vụ
    ngoài trục trặc là lỗi thiết kế, không phải chuyện đương nhiên.
    """
    model: str
    runtime: Optional[str] = None
    entities: List[Entity] = []
    latency_ms: Optional[float] = None
    meta: Dict = {}
    error: Optional[str] = None


class Agreement(BaseModel):
    """Mức khớp nhau giữa hai model trên cùng một văn bản.

    Coi model đầu là mốc tham chiếu rồi tính precision/recall/F1 cho model kia. Đây KHÔNG phải độ chính xác so với đáp án đúng -- không có đáp án đúng
    ở đây. Muốn con số đó thì chạy `benchmarks/run_benchmark.py` trên tập có
    nhãn người gán.
    """
    reference: str
    candidate: str
    precision: float
    recall: float
    f1: float
    matched: int
    only_in_reference: List[str] = []
    only_in_candidate: List[str] = []


class CompareResponse(BaseModel):
    text: str
    results: List[ModelResult]
    agreements: List[Agreement] = []


class ModelInfo(BaseModel):
    name: str = Field(..., description="Tên để điền vào trường `model` của request.")
    runtime: str = Field(..., description="Nơi chạy model này.")
    kind: str = Field(..., description="inference-server, in-process hoặc external-api.")
    description: str
    ready: bool


class ModelListResponse(BaseModel):
    default: str
    fallback: Optional[str] = None
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' khi model mặc định phục vụ được, "
                                         "'degraded' khi chỉ còn đường dự phòng.")
    default_model: str
    runtimes: Dict[str, bool] = Field(..., description="Tên nơi chạy -> sẵn sàng hay không.")
    llm_model: Optional[str] = None
