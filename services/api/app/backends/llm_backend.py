# -*- coding: utf-8 -*-
"""Chạy NER bằng một LLM gọi qua mạng (giao diện chuẩn OpenAI).

Đặt cạnh model tự train để trả lời câu hỏi thực tế nhất khi bắt đầu một dự án
NLP: *có cần tự huấn luyện không, hay gọi API là đủ?* Muốn trả lời được thì phải
đo cả hai trên CÙNG dữ liệu, CÙNG endpoint.

Vấn đề kỹ thuật lớn nhất khi dùng LLM cho bài toán có cấu trúc: LLM sinh ra
VĂN BẢN, còn ta cần DỮ LIỆU. Ép nó trả JSON đúng dạng gọi là structured output,
và mức hỗ trợ **khác nhau tuỳ model** -- đã đo trên Groq:

    model                json_schema   json_object   ghi chú
    -------------------  ------------  ------------  -----------------------------
    openai/gpt-oss-20b   được          hỏng          json_object lỗi 400 với văn
                                                     bản dài (json_validate_failed)
    groq/compound        LỖI 400       được          model có tích hợp công cụ,
                                                     không nhận json_schema
    groq/compound-mini   LỖI 400       được          như trên

Kết luận: không có chế độ nào chạy được với mọi model. Nên ở đây dùng THANG
DỰ PHÒNG, tự dò một lần cho mỗi model rồi nhớ lại:

    json_schema  ->  json_object  ->  văn bản thường + bóc JSON thủ công

Hạ một bậc không phải thất bại: đó là cách một tích hợp bền vững hoạt động khi
nhà cung cấp đổi model dưới chân mình.
"""
import json
import logging
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

from app.backends.base import Prediction, Runtime
from app.config import settings
from app.core import decoding
from app.metrics import stage

logger = logging.getLogger("uvicorn.error")

# Lược đồ JSON mô tả CHÍNH XÁC dạng dữ liệu mong muốn. Với model hỗ trợ
# json_schema, nhà cung cấp ép bộ sinh chỉ phát ra token hợp lệ -- không còn
# chuyện "LLM quên đóng ngoặc".
NER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "label"],
                "properties": {
                    "text": {"type": "string"},
                    "label": {"type": "string", "enum": list(decoding.VALID_LABELS)},
                },
            },
        }
    },
}

# Cùng một yêu cầu được lặp lại trong prompt dù đã có json_schema: model ở bậc
# thấp hơn của thang dự phòng không nhận schema, chỉ còn prompt để dựa vào.
PROMPT = """Bạn là bộ trích xuất thực thể có tên (NER) cho tiếng Việt.

Tìm mọi thực thể thuộc ba loại sau trong văn bản:
- PER: tên riêng của người.
- ORG: tên tổ chức, công ty, cơ quan, đội nhóm.
- LOC: tên địa danh, quốc gia, tỉnh, thành phố, xã, sông núi.

Quy tắc bắt buộc:
1. Chỉ trả về đúng một JSON object, không giải thích, không dùng markdown.
2. Cấu trúc: {{"entities": [{{"text": "<chuỗi>", "label": "PER"}}]}}
3. Trường "text" phải sao chép NGUYÊN VĂN từ văn bản, giữ đúng dấu và chữ hoa.
4. Không đưa danh từ chung ("anh", "chị", "ông", "xã", "tỉnh", "công ty") vào "text".
5. Không có thực thể nào thì trả về {{"entities": []}}.

Văn bản:
\"\"\"{text}\"\"\""""

_lock = threading.Lock()
_resolved_model: Optional[str] = None
_structured_mode: Dict[str, str] = {}      # model -> "json_schema" | "json_object" | "text"


def _client():
    from openai import OpenAI
    return OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL,
                  timeout=settings.LLM_TIMEOUT_S)


_models_cache: Tuple[float, List[str]] = (0.0, [])


def list_available_models(max_age_s: float = 300.0) -> List[str]:
    """Danh sách model nhà cung cấp đang có, nhớ lại trong 5 phút.

    Không có bộ nhớ đệm thì mỗi lần mở trang demo là một lượt gọi ra Internet,
    vì giao diện gọi GET /v1/models ngay khi nạp trang. Danh sách model của nhà
    cung cấp gần như không đổi, nên hỏi lại mỗi request là lãng phí thuần tuý.
    """
    global _models_cache
    fetched_at, cached = _models_cache
    now = time.monotonic()
    if cached and now - fetched_at < max_age_s:
        return cached
    try:
        models = sorted(m.id for m in _client().models.list().data)
        _models_cache = (now, models)
        return models
    except Exception as e:                                    # noqa: BLE001
        logger.warning("Không liệt kê được model từ %s: %s", settings.LLM_BASE_URL, e)
        return cached          # thà trả danh sách cũ còn hơn trả rỗng


def resolve_model(requested: Optional[str] = None) -> str:
    """Chọn model dùng được.

    Ưu tiên: model người gọi chỉ định trong body -> biến môi trường LLM_MODEL ->
    model đầu tiên trong danh sách ưu tiên mà nhà cung cấp THỰC SỰ có -> model
    bất kỳ còn khả dụng.

    Không bao giờ hard-code một tên model duy nhất: nhà cung cấp gỡ và đổi tên
    model liên tục. Hệ thống này đã dính đúng lỗi đó một lần
    (`llama-3.3-70b-versatile` bị gỡ -> toàn bộ endpoint trả 500).
    Xem docs/TROUBLESHOOTING.md muc E04.
    """
    global _resolved_model
    if requested:
        return requested
    if _resolved_model:
        return _resolved_model
    with _lock:
        if _resolved_model:
            return _resolved_model
        available = list_available_models()
        if settings.LLM_MODEL:
            if available and settings.LLM_MODEL not in available:
                logger.warning("LLM_MODEL=%s không nằm trong danh sách khả dụng: %s",
                               settings.LLM_MODEL, available)
            _resolved_model = settings.LLM_MODEL
            return _resolved_model
        for cand in settings.LLM_MODEL_CANDIDATES:
            if cand in available:
                _resolved_model = cand
                logger.info("Tự chọn LLM model: %s", cand)
                return _resolved_model
        if available:
            _resolved_model = available[0]
            logger.warning("Không tên nào trong danh sách ưu tiên khả dụng, dùng tạm: %s",
                           _resolved_model)
            return _resolved_model
        raise RuntimeError(
            f"Không tìm được model khả dụng nào tại {settings.LLM_BASE_URL}. "
            "Kiểm tra LLM_API_KEY còn hiệu lực không.")


def _response_format(mode: str):
    if mode == "json_schema":
        return {"type": "json_schema",
                "json_schema": {"name": "ner_result", "strict": True,
                                "schema": NER_JSON_SCHEMA}}
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def _is_unsupported_format_error(exc: Exception) -> bool:
    """Lỗi này có phải "model không hỗ trợ chế độ structured output" không?

    Phân biệt cho đúng là điều bắt buộc. Ban đầu code ở đây bắt MỌI exception rồi
    hạ bậc, nên khi nhà cung cấp trả 429 (quá hạn mức tần suất) hệ thống lại đi
    thử lại cả ba bậc, đốt thêm quota, rồi báo lỗi sai hoàn toàn:
    "không chế độ nào chạy được" trong khi sự thật là "gọi quá nhanh".

    Quy tắc chung rút ra: chỉ bắt đúng loại lỗi mà mình biết cách xử lý. Bắt tất
    rồi xử lý như nhau là cách nhanh nhất để giấu mất nguyên nhân thật.
    """
    status = getattr(exc, "status_code", None)
    if status != 400:
        return False
    message = str(getattr(exc, "message", "") or exc).lower()
    return "response format" in message or "response_format" in message \
        or "json_schema" in message or "json_object" in message


def _retryable_status(exc: Exception) -> bool:
    """429 (quá hạn mức) và 5xx (lỗi phía nhà cung cấp) thì thử lại có ích.
    401/403/404 thì thử lại bao nhiêu lần cũng vô nghĩa."""
    status = getattr(exc, "status_code", None)
    return status == 429 or (isinstance(status, int) and 500 <= status < 600)


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    """Ưu tiên khoảng chờ do chính nhà cung cấp yêu cầu; không có thì lùi theo
    cấp số nhân. Tự đoán một con số trong khi máy chủ đã nói rõ phải chờ bao lâu
    là cách chắc chắn để bị chặn tiếp."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        raw = headers.get(key)
        if raw:
            try:
                return min(float(str(raw).rstrip("s")), 30.0)
            except ValueError:
                pass
    return min(2.0 ** attempt, 16.0)


def _call(model: str, text: str, mode: str):
    """Gọi LLM, tự thử lại khi bị giới hạn tần suất hoặc nhà cung cấp trục trặc."""
    kwargs = {"model": model,
              "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
              "temperature": 0.0}
    rf = _response_format(mode)
    if rf:
        kwargs["response_format"] = rf

    last_error = None
    for attempt in range(settings.LLM_MAX_RETRIES + 1):
        try:
            return _client().chat.completions.create(**kwargs)
        except Exception as e:                                     # noqa: BLE001
            last_error = e
            if not _retryable_status(e) or attempt == settings.LLM_MAX_RETRIES:
                raise
            wait = _retry_after_seconds(e, attempt)
            logger.info("LLM trả lỗi tạm thời (%s), chờ %.1fs rồi thử lại (lần %d/%d).",
                        getattr(e, "status_code", "?"), wait,
                        attempt + 1, settings.LLM_MAX_RETRIES)
            time.sleep(wait)
    raise last_error


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_entities(raw: str) -> List[Dict]:
    """Bóc danh sách thực thể từ câu trả lời của LLM.

    Chấp nhận cả hai dạng đã quan sát được trong thực tế:
      {"entities": [{"text": "...", "label": "PER"}]}      <- dạng đã yêu cầu
      {"PER": ["..."], "LOC": ["..."]}                     <- dạng model tự chế

    Trả về danh sách rỗng khi không đọc được, KHÔNG ném lỗi: một câu trả lời xấu
    của dịch vụ ngoài không được phép làm hỏng cả request.
    """
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    match = _JSON_BLOCK.search(cleaned)
    if not match:
        logger.warning("Câu trả lời LLM không chứa JSON: %s", cleaned[:200])
        return []

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("LLM trả JSON hỏng: %s", match.group(0)[:200])
        return []

    items: List[Dict] = []
    if isinstance(parsed.get("entities"), list):
        for e in parsed["entities"]:
            if isinstance(e, dict):
                items.append({"text": e.get("text") or e.get("word"),
                              "label": e.get("label") or e.get("type")})
    else:
        for label in decoding.VALID_LABELS:                # dạng {"PER": [...]}
            for value in parsed.get(label, []) or []:
                items.append({"text": value, "label": label})

    return [i for i in items
            if i.get("text") and str(i.get("label", "")).upper() in decoding.VALID_LABELS]


class LLMRuntime(Runtime):
    name = "llm"
    kind = "external-api"
    description = ("LLM gọi qua mạng theo giao diện chuẩn OpenAI. "
                   "Không cần huấn luyện, đổi lại chậm hơn và tính tiền theo token.")

    def models(self) -> List[str]:
        """Model LLM để GỢI Ý trong danh sách, không phải danh sách đóng.

        Nhà cung cấp trả về đủ thứ model: nhận dạng giọng nói, tổng hợp tiếng
        nói, phân loại an toàn. Liệt kê hết thì danh sách vừa dài vừa gây hiểu
        nhầm là chúng làm được NER. Nên ở đây chỉ lấy phần giao giữa danh sách
        ưu tiên đã cấu hình và những model nhà cung cấp thực sự đang có.

        Người gọi vẫn truyền được BẤT KỲ tên model nào: định tuyến dựa trên
        "không phải model tự host", không dựa trên danh sách này.
        """
        if not settings.LLM_API_KEY:
            return []
        available = set(list_available_models())
        wanted = list(settings.LLM_MODEL_CANDIDATES)
        if settings.LLM_MODEL:
            wanted.insert(0, settings.LLM_MODEL)
        return [m for m in wanted if m in available]

    def is_ready(self) -> bool:
        return bool(settings.LLM_API_KEY)

    def predict(self, text: str, model: str, options: Dict) -> Prediction:
        if not settings.LLM_API_KEY:
            raise RuntimeError(
                "Chưa cấu hình LLM_API_KEY. Lấy key miễn phí tại "
                "https://console.groq.com/keys, ghi vào file .env rồi chạy lại "
                "`docker compose up -d`.")

        model = resolve_model(model)
        # Thang dự phòng. Bậc nào đã biết chạy được với model này thì thử luôn
        # từ đó, khỏi mất một lượt gọi thừa cho mỗi request.
        ladder = ["json_schema", "json_object", "text"]
        known = _structured_mode.get(model)
        if known:
            ladder = ladder[ladder.index(known):]

        start = time.perf_counter()
        last_error = None
        for mode in ladder:
            try:
                with stage("inference", model):
                    resp = _call(model, text, mode)
            except Exception as e:                                 # noqa: BLE001
                if not _is_unsupported_format_error(e):
                    # Hết quota, sai key, mạng hỏng: hạ bậc không giúp được gì,
                    # chỉ làm mờ nguyên nhân thật. Ném thẳng ra ngoài.
                    raise
                last_error = e
                logger.info("Model %s không hỗ trợ chế độ %s, hạ xuống bậc dưới.", model, mode)
                continue

            with stage("postprocess", model):
                raw = (resp.choices[0].message.content or "").strip()
                items = parse_entities(raw)
            if not items and raw and mode != "text":
                # Gọi được nhưng đọc không ra dữ liệu: có thể do chế độ này chỉ
                # được hỗ trợ nửa vời. Hạ bậc thay vì trả về kết quả rỗng sai sự thật.
                logger.info("Chế độ %s trả nội dung không đọc được, hạ một bậc.", mode)
                last_error = ValueError("Không bóc được JSON: " + raw[:200])
                continue

            _structured_mode[model] = mode
            latency_ms = (time.perf_counter() - start) * 1000
            usage = getattr(resp, "usage", None)
            return Prediction(
                entities=decoding.spans_from_strings(text, items),
                model=model,
                latency_ms=latency_ms,
                meta={"structured_mode": mode,
                      "raw": raw[:2000],
                      "prompt_tokens": getattr(usage, "prompt_tokens", None),
                      "completion_tokens": getattr(usage, "completion_tokens", None)},
            )

        raise RuntimeError(
            f"Không chế độ structured output nào chạy được với model {model}. "
            f"Lỗi cuối: {last_error}")
