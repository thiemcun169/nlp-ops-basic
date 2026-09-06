# -*- coding: utf-8 -*-
"""Tách văn bản thành từ, GIỮ vị trí ký tự gốc.

Vì sao cần cả một module riêng cho việc "tách từ" tưởng như tầm thường:

Model được huấn luyện trên WikiAnn, nơi dấu câu là MỘT TOKEN RIÊNG:
    ['anh', 'Nguyễn', 'Viết', 'Quang', ',', '40', 'tuổi']

Nếu lúc phục vụ ta dùng `text.split()` thì được:
    ['anh', 'Nguyễn', 'Viết', 'Quang,', '40', 'tuổi']

Token "Quang," không hề tồn tại trong dữ liệu huấn luyện. Model đã đo được là
gán nhãn O cho nó, tức BỎ SÓT chữ cuối của tên người. Đây là lỗi
training/serving skew kinh điển: model không sai, khâu tiền xử lý sai.

Bằng chứng đo trên chính hệ thống này (xem docs/TROUBLESHOOTING.md muc E07):
    text.split()          -> Nguyễn:B-PER Viết:I-PER              (thiếu "Quang")
    tách dấu câu ra riêng -> Nguyễn:B-PER Viết:I-PER Quang:I-PER  (đủ)

Ngoài ra mỗi từ được kèm offset (start, end) trong chuỗi gốc. Nhờ vậy API trả
được span theo vị trí ký tự, và giao diện tô màu bằng offset thay vì so khớp
chuỗi -- so khớp chuỗi chính là lý do web UI cũ không tô được thực thể nào cho
LLM (LLM trả "Quang" trong khi văn bản có "Quang,").
"""
import re
from typing import List, NamedTuple

# Thứ tự các nhánh trong regex có ý nghĩa, khớp từ trái sang:
#   1. Số có dấu phân cách bên trong: 4/9, 3.14, 1,5, 10:30  -> giữ nguyên một khối
#   2. Chuỗi chữ/số liền nhau (\w bao gồm chữ tiếng Việt có dấu)
#   3. Chuỗi dấu câu GIỐNG NHAU liền nhau -> một token: "...", "!!!", "??"
#
# Nhánh 3 dùng `([^\w\s])\1*` chứ không phải `[^\w\s]` là có đo đạc: tách rời
# từng dấu làm "..." thành ba token, lệch với cách tách của bộ dữ liệu huấn luyện
# ở 36,8% số câu. Gom lại còn 16,6%. Xem benchmarks/eval_testset.py, phép đo
# "tách từ lệch so với bộ dữ liệu".
_WORD_RE = re.compile(r"\d+(?:[.,:/]\d+)+|\w+|([^\w\s])\1*", re.UNICODE)


class Word(NamedTuple):
    """Một từ cùng vị trí của nó trong văn bản gốc."""
    text: str
    start: int
    end: int


def split_words(text: str) -> List[Word]:
    """Tách văn bản thành danh sách từ kèm offset ký tự.

    >>> [w.text for w in split_words("anh Quang, 40 tuổi.")]
    ['anh', 'Quang', ',', '40', 'tuổi', '.']
    """
    return [Word(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
