# -*- coding: utf-8 -*-
"""Đo mức khớp nhau giữa hai bộ kết quả NER.

Dùng cho hai việc:
- `/v1/ner/compare`: hai model nói giống nhau tới đâu trên một văn bản.
- `benchmarks/run_benchmark.py`: chấm điểm model so với nhãn người gán.

Cả hai dùng chung một hàm, vì về mặt toán học chúng là một phép tính: so một
tập span với một tập span khác. Chỉ khác ở chỗ tập nào được coi là chân lý.

Khớp theo tiêu chí NGHIÊM (exact match): đúng cả vị trí bắt đầu, vị trí kết thúc
lẫn loại thực thể mới tính là khớp. Đây là tiêu chuẩn của CoNLL và của seqeval,
cũng là tiêu chí dùng khi đánh giá model lúc huấn luyện, nên các con số ở đây so
được trực tiếp với F1 trong notebook.
"""
from typing import Dict, List, Set, Tuple

Span = Tuple[int, int, str]


def to_span_set(entities: List[Dict]) -> Set[Span]:
    """Bỏ qua thực thể không định vị được (start = -1): không có vị trí thì
    không có gì để so."""
    return {(e["start"], e["end"], e["label"]) for e in entities if e.get("start", -1) >= 0}


def _fmt(span: Span, text: str) -> str:
    return f"{text[span[0]:span[1]]} [{span[2]}]"


def compare(reference: List[Dict], candidate: List[Dict], text: str = "") -> Dict:
    """Precision/recall/F1 của `candidate` khi lấy `reference` làm mốc.

    Quy ước khi tập tham chiếu rỗng: nếu candidate cũng rỗng thì coi là khớp
    hoàn toàn (F1 = 1). Hai model cùng nói "câu này không có thực thể nào" là
    đồng ý với nhau, không phải cùng thất bại.
    """
    ref, cand = to_span_set(reference), to_span_set(candidate)
    matched = ref & cand

    precision = len(matched) / len(cand) if cand else (1.0 if not ref else 0.0)
    recall = len(matched) / len(ref) if ref else (1.0 if not cand else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": len(matched),
        "only_in_reference": sorted(_fmt(s, text) for s in ref - cand),
        "only_in_candidate": sorted(_fmt(s, text) for s in cand - ref),
    }
