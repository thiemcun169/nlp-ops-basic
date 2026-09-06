# -*- coding: utf-8 -*-
"""Giao diện chung cho mọi nơi chạy được model NER.

Người gọi API chỉ nói tên model, không cần biết nó chạy ở đâu. Việc chọn nơi
chạy là của `registry.py`. Nhờ vậy thêm một nơi chạy mới (TorchServe, vLLM, một
nhà cung cấp LLM khác) chỉ cần viết một lớp Runtime rồi đăng ký, không phải sửa
router hay giao diện.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Prediction:
    """Kết quả một lần chạy, đã chuẩn hoá cho mọi nơi chạy."""
    entities: List[Dict]
    model: str
    latency_ms: float
    meta: Dict = field(default_factory=dict)


class Runtime:
    """Lớp cha. Mỗi nơi chạy con phải khai báo `name` và cài `predict`."""

    name: str = "base"
    kind: str = "unknown"          # "inference-server" | "in-process" | "external-api"
    description: str = ""

    def models(self) -> List[str]:
        """Các model chạy được ở đây. Dùng để dựng bảng định tuyến."""
        return []

    def predict(self, text: str, model: str, options: Dict) -> Prediction:
        raise NotImplementedError

    def is_ready(self) -> bool:
        """Nơi chạy này dùng được ngay bây giờ không.

        Không được ném lỗi: một nơi chạy hỏng phải trả về False chứ không làm
        sập health check.
        """
        return True
