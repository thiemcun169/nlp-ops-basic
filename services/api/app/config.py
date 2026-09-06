# -*- coding: utf-8 -*-
"""Cấu hình tập trung, đọc từ biến môi trường.

Nguyên tắc 12-factor: không có giá trị nào chỉ sửa được bằng cách sửa mã nguồn.
Cùng một image chạy được ở máy cá nhân, máy thuê GPU và máy chủ thật, chỉ khác
biến môi trường. Mọi mặc định ở đây phải là mặc định CHẠY ĐƯỢC NGAY, để người
mới chỉ cần `docker compose up` là có hệ thống sống.
"""
import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    # --- Triton -------------------------------------------------------------
    TRITON_URL: str = os.environ.get("TRITON_URL", "triton:8000")
    TRITON_MODEL_GPU: str = os.environ.get("TRITON_MODEL_GPU", "ner_onnx")
    TRITON_MODEL_CPU: str = os.environ.get("TRITON_MODEL_CPU", "ner_onnx_cpu")
    # Số kết nối mỗi thread giữ tới Triton. Xem docs/TROUBLESHOOTING.md muc E03.
    TRITON_CONCURRENCY: int = _int("TRITON_CONCURRENCY", 4)

    # --- Artifact của model -------------------------------------------------
    TOKENIZER_DIR: str = os.environ.get("TOKENIZER_DIR", "/app/tokenizer")
    LABELS_PATH: str = os.environ.get("LABELS_PATH", "/app/labels.json")
    # Độ dài tối đa tính bằng subword. Phải KHỚP giá trị dùng lúc huấn luyện.
    MAX_LEN: int = _int("MAX_LEN", 128)

    # --- ONNX Runtime chạy tại chỗ (dự phòng + mốc so sánh) -----------------
    ONNX_LOCAL_PATH: str = os.environ.get("ONNX_LOCAL_PATH", "/models/ner_onnx/1/model.onnx")
    ONNX_LOCAL_THREADS: int = _int("ONNX_LOCAL_THREADS", 2)

    # --- Tên model tự host --------------------------------------------------
    # Người gọi API chỉ gửi `model`. Tên nằm trong nhóm này thì chạy tại chỗ,
    # mọi tên khác được đẩy sang nhà cung cấp LLM (xem backends/registry.py).
    LOCAL_MODEL_GPU: str = os.environ.get("LOCAL_MODEL_GPU", "phobert-ner-gpu")
    LOCAL_MODEL_CPU: str = os.environ.get("LOCAL_MODEL_CPU", "phobert-ner-cpu")
    LOCAL_MODEL_ONNX: str = os.environ.get("LOCAL_MODEL_ONNX", "phobert-ner-local")

    DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "phobert-ner-gpu")
    # Model dùng khi máy chủ suy luận hỏng. Để rỗng là tắt cơ chế dự phòng.
    FALLBACK_MODEL: str = os.environ.get("FALLBACK_MODEL", "phobert-ner-local")

    # --- LLM (giao diện chuẩn OpenAI) ---------------------------------------
    # Đổi sang nhà cung cấp khác chỉ cần đổi ba biến LLM_* dưới đây.
    LLM_API_KEY: str = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    # Để RỖNG thì hệ thống tự dò model khả dụng theo danh sách ưu tiên bên dưới.
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "")
    LLM_MODEL_CANDIDATES = [
        m.strip() for m in os.environ.get(
            "LLM_MODEL_CANDIDATES",
            # groq/compound đứng đầu: hạn mức miễn phí rộng hơn hẳn.
            "groq/compound,groq/compound-mini,openai/gpt-oss-20b,openai/gpt-oss-120b",
        ).split(",") if m.strip()
    ]
    LLM_TIMEOUT_S: float = float(os.environ.get("LLM_TIMEOUT_S", "60"))
    # Số lần thử lại khi nhà cung cấp trả 429 (quá hạn mức) hoặc 5xx. Hạn mức
    # miễn phí rất dễ chạm khi chạy đo hàng loạt, nên thử lại là bắt buộc chứ
    # không phải tuỳ chọn.
    LLM_MAX_RETRIES: int = _int("LLM_MAX_RETRIES", 4)

    # --- Giới hạn đầu vào ---------------------------------------------------
    MAX_TEXT_CHARS: int = _int("MAX_TEXT_CHARS", 5000)

    # --- Giao diện web ------------------------------------------------------
    STATIC_DIR: str = os.environ.get("STATIC_DIR", "/app/static")


settings = Settings()
