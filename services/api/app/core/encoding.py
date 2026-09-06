# -*- coding: utf-8 -*-
"""Đổi danh sách từ thành tensor cho model, và ngược lại.

Cả ba nơi chạy model PhoBERT (Triton GPU, Triton CPU, ONNX Runtime tại
chỗ) dùng CHUNG module này. Đó là chủ ý: chỉ cần một chỗ duy nhất định nghĩa
"chữ biến thành số như thế nào", nên không thể có chuyện hai nơi cho ra hai
kết quả khác nhau trên cùng một câu.

Quy ước căn nhãn ở đây phải KHỚP HỆT hàm `tokenize_and_align` trong
`notebooks/01_train_ner.ipynb`. Lệch một chi tiết là nhãn trả về lệch theo, mà
lỗi kiểu đó không hề làm chương trình dừng -- nó chỉ âm thầm trả sai.
"""
import json
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from app.config import settings
from app.core.tokenization import Word


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(settings.TOKENIZER_DIR)


@lru_cache(maxsize=1)
def get_id2label() -> Dict[int, str]:
    with open(settings.LABELS_PATH, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def encode(words: List[Word]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Từ -> (input_ids, attention_mask, vị trí subword đầu của mỗi từ).

    PhoBERT dùng tokenizer chậm, không có `.word_ids()`, nên phải tự tokenize
    TỪNG từ rồi ghi lại subword đầu tiên của nó. Nhãn của một từ được lấy tại
    đúng vị trí subword đầu tiên đó -- giống hệt lúc huấn luyện.

    Trả về `first_subtoken[i] = -1` khi từ thứ i bị cắt do vượt MAX_LEN, để
    tầng trên biết mà bỏ qua thay vì đọc nhầm vị trí của từ khác.
    """
    tokenizer = get_tokenizer()
    input_ids = [tokenizer.bos_token_id]
    first_subtoken: List[int] = []
    limit = settings.MAX_LEN - 1  # chừa 1 chỗ cho token kết thúc câu

    truncated = False
    for word in words:
        if truncated:
            first_subtoken.append(-1)
            continue
        sub_ids = tokenizer.encode(word.text, add_special_tokens=False)
        if not sub_ids:
            first_subtoken.append(-1)
            continue
        if len(input_ids) + len(sub_ids) > limit:
            # Hết chỗ. Đánh dấu cắt từ đây TRỞ ĐI, không nhận thêm từ nào nữa.
            # Nếu chỉ `continue`, một từ ngắn đứng sau từ dài vừa bị bỏ vẫn lọt
            # vào, làm chuỗi đưa cho model không còn liên tục -- nhãn trả về sẽ
            # sai một cách khó lần ra.
            truncated = True
            first_subtoken.append(-1)
            continue
        first_subtoken.append(len(input_ids))
        input_ids.extend(sub_ids)

    input_ids.append(tokenizer.eos_token_id)
    ids = np.array([input_ids], dtype=np.int64)
    return ids, np.ones_like(ids), first_subtoken


def decode_logits(logits: np.ndarray, first_subtoken: List[int],
                  want_scores: bool) -> Tuple[List[str], List[float]]:
    """logits (seq_len, num_labels) -> nhãn BIO cho từng từ, kèm độ tin cậy."""
    id2label = get_id2label()
    pred_ids = logits.argmax(-1)

    probs = None
    if want_scores:
        # softmax ổn định số học: trừ max trước khi mũ, tránh tràn số với logit lớn
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)

    labels: List[str] = []
    scores: List[float] = []
    for pos in first_subtoken:
        if pos < 0 or pos >= len(pred_ids):
            labels.append("O")
            scores.append(0.0)
            continue
        labels.append(id2label[int(pred_ids[pos])])
        scores.append(float(probs[pos].max()) if probs is not None else 1.0)
    return labels, scores
