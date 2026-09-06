# -*- coding: utf-8 -*-
"""Chạy model qua NVIDIA Triton Inference Server.

Cùng một file trọng số được Triton nạp thành hai model: một bản trên GPU, một
bản trên CPU. Nhờ vậy phép so GPU với CPU là so công bằng: cùng trọng số, cùng
runtime, cùng máy chủ, khác biệt đo được chỉ đến từ phần cứng.
"""
import logging
import threading
import time
from typing import Dict, List

import numpy as np
import tritonclient.http as httpclient

from app.backends.base import Prediction, Runtime
from app.config import settings
from app.core import decoding, encoding, tokenization
from app.metrics import stage

logger = logging.getLogger("uvicorn.error")

# tritonclient.http dùng gevent/greenlet bên trong. Một greenlet gắn chặt với
# đúng một OS thread, trong khi FastAPI chạy handler đồng bộ trong threadpool
# nhiều thread thật. Dùng chung một client cho mọi thread sẽ thỉnh thoảng lỗi
# "Cannot switch to a different thread", và lỗi chỉ hiện khi tải cao.
# Xem docs/TROUBLESHOOTING.md muc E03.
_local = threading.local()


def _client() -> httpclient.InferenceServerClient:
    client = getattr(_local, "client", None)
    if client is None:
        client = httpclient.InferenceServerClient(
            url=settings.TRITON_URL, concurrency=settings.TRITON_CONCURRENCY)
        _local.client = client
    return client


class TritonRuntime(Runtime):
    kind = "inference-server"

    def __init__(self, name: str, model_name: str, triton_model: str, description: str):
        self.name = name
        self.model_name = model_name        # tên model người gọi dùng
        self.triton_model = triton_model    # tên model bên trong Triton
        self.description = description

    def models(self) -> List[str]:
        return [self.model_name]

    def is_ready(self) -> bool:
        try:
            return bool(_client().is_model_ready(self.triton_model))
        except Exception as e:                                    # noqa: BLE001
            logger.warning("Triton chưa sẵn sàng cho %s: %s", self.triton_model, e)
            return False

    def predict(self, text: str, model: str, options: Dict) -> Prediction:
        words = tokenization.split_words(text)
        if not words:
            raise ValueError("Văn bản không chứa từ nào để phân tích.")

        want_scores = bool(options.get("return_scores"))
        start = time.perf_counter()

        # Ba chặng đo riêng. Không tách ra thì một hệ thống chậm vì tokenize sẽ
        # trông y hệt một hệ thống chậm vì model, mà hai thứ đó sửa khác nhau.
        with stage("preprocess", self.model_name):
            ids, mask, first_subtoken = encoding.encode(words)
            inp_ids = httpclient.InferInput("input_ids", ids.shape, "INT64")
            inp_ids.set_data_from_numpy(ids)
            inp_mask = httpclient.InferInput("attention_mask", mask.shape, "INT64")
            inp_mask.set_data_from_numpy(mask)

        with stage("inference", self.model_name):
            result = _client().infer(
                self.triton_model,
                inputs=[inp_ids, inp_mask],
                outputs=[httpclient.InferRequestedOutput("logits")],
            )

        with stage("postprocess", self.model_name):
            logits = np.asarray(result.as_numpy("logits"))[0]
            labels, scores = encoding.decode_logits(logits, first_subtoken, want_scores)
            entities = decoding.group_bio(text, words, labels, scores if want_scores else None)

        latency_ms = (time.perf_counter() - start) * 1000

        return Prediction(
            entities=entities,
            model=self.model_name,
            latency_ms=latency_ms,
            meta={"num_words": len(words),
                  "truncated": any(p < 0 for p in first_subtoken),
                  "max_len": settings.MAX_LEN},
        )
