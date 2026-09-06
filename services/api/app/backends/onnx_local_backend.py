# -*- coding: utf-8 -*-
"""Chạy model bằng ONNX Runtime NGAY TRONG tiến trình FastAPI.

Có hai lý do để giữ backend này bên cạnh Triton:

1. **Dự phòng.** Triton chết (hết bộ nhớ GPU, container bị restart, sập mạng
   nội bộ) thì API vẫn trả lời được, chỉ chậm hơn. Suy giảm chức năng còn hơn
   ngừng phục vụ. Bật/tắt bằng biến `FALLBACK_MODEL`.

2. **Điểm tham chiếu khi đo.** Đây là cách chạy model đơn giản nhất có thể:
   một tiến trình, một request một lần, không gộp batch, không hàng đợi. So với
   nó mới thấy rõ máy chủ suy luận đem lại chính xác điều gì.

Đánh đổi phải nói thẳng: cách này nạp nguyên model vào bộ nhớ tiến trình web,
không gộp batch động, không phục vụ nhiều phiên bản model, không xuất metrics.
Chạy demo thì được, làm xương sống cho hệ thống thật thì không.

Model được nạp LƯỜI (chỉ khi thực sự gọi lần đầu), nên nếu bạn không dùng
backend này thì nó không tốn một byte RAM nào.
"""
import logging
import threading
import time
from typing import Dict

import numpy as np

from app.backends.base import Prediction, Runtime
from app.config import settings
from app.core import decoding, encoding, tokenization
from app.metrics import stage

logger = logging.getLogger("uvicorn.error")

_session = None
_session_lock = threading.Lock()


def _get_session():
    """Nạp ONNX Runtime session một lần duy nhất, an toàn khi nhiều thread cùng gọi."""
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            # Giới hạn số luồng: tiến trình này còn phải phục vụ HTTP, để ONNX
            # Runtime chiếm hết CPU thì chính API bị đói tài nguyên.
            opts.intra_op_num_threads = settings.ONNX_LOCAL_THREADS
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(
                settings.ONNX_LOCAL_PATH, sess_options=opts,
                providers=["CPUExecutionProvider"])
            logger.info("Đã nạp ONNX Runtime tại chỗ: %s", settings.ONNX_LOCAL_PATH)
    return _session


class OnnxLocalRuntime(Runtime):
    name = "onnx-local"
    kind = "in-process"
    description = ("ONNX Runtime chạy thẳng trong tiến trình API (CPU). "
                   "Dùng làm phương án dự phòng khi Triton hỏng, và làm mốc so sánh.")

    def models(self):
        return [settings.LOCAL_MODEL_ONNX]

    def is_ready(self) -> bool:
        import os
        return os.path.exists(settings.ONNX_LOCAL_PATH)

    def predict(self, text: str, model: str, options: Dict) -> Prediction:
        words = tokenization.split_words(text)
        if not words:
            raise ValueError("Văn bản không chứa từ nào để phân tích.")

        want_scores = bool(options.get("return_scores"))
        session = _get_session()          # lần đầu sẽ mất vài giây để nạp model
        start = time.perf_counter()

        with stage("preprocess", settings.LOCAL_MODEL_ONNX):
            ids, mask, first_subtoken = encoding.encode(words)

        with stage("inference", settings.LOCAL_MODEL_ONNX):
            logits = session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]

        with stage("postprocess", settings.LOCAL_MODEL_ONNX):
            logits = np.asarray(logits)[0]
            labels, scores = encoding.decode_logits(logits, first_subtoken, want_scores)
            entities = decoding.group_bio(text, words, labels, scores if want_scores else None)

        latency_ms = (time.perf_counter() - start) * 1000
        return Prediction(
            entities=entities,
            model=settings.LOCAL_MODEL_ONNX,
            latency_ms=latency_ms,
            meta={"num_words": len(words),
                  "truncated": any(p < 0 for p in first_subtoken),
                  "max_len": settings.MAX_LEN},
        )
