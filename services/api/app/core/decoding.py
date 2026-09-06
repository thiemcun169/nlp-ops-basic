# -*- coding: utf-8 -*-
"""Gom nhãn BIO ở mức TỪ thành SPAN thực thể ở mức ký tự.

Model trả nhãn cho từng từ theo lược đồ BIO:
    Nguyễn:B-PER  Viết:I-PER  Quang:I-PER  ,:O  40:O  tuổi:O

Nhưng thứ mà ứng dụng gọi API cần là một thực thể trọn vẹn kèm vị trí:
    {"text": "Nguyễn Viết Quang", "label": "PER", "start": 20, "end": 37}

Vì sao trả span theo offset chứ không trả danh sách từ:
- Frontend tô màu được bằng `text.slice(start, end)`, không phải đoán bằng cách
  so khớp chuỗi (cách cũ hỏng ngay khi từ dính dấu câu).
- So sánh giữa hai model khác nhau trở nên khả thi: model tự train trả từng từ,
  LLM trả cả cụm, nhưng quy về span ký tự thì hai bên nói CÙNG một ngôn ngữ.
- Đây cũng là quy ước chung của các thư viện NER phổ biến (spaCy, Hugging Face
  `pipeline("ner", aggregation_strategy=...)`), nên tích hợp về sau đỡ bất ngờ.
"""
from typing import Dict, List, Optional

from app.core.tokenization import Word

VALID_LABELS = ("PER", "ORG", "LOC")


def entity_type(bio_label: str) -> Optional[str]:
    """'B-PER' -> 'PER'; 'O' -> None."""
    if not bio_label or bio_label == "O":
        return None
    return bio_label.split("-")[-1]


def group_bio(text: str, words: List[Word], labels: List[str],
              scores: Optional[List[float]] = None) -> List[Dict]:
    """Gom chuỗi nhãn BIO thành danh sách span.

    Quy ước gom (giống `aggregation_strategy="simple"` của Hugging Face):
    - `B-X` luôn MỞ một span mới.
    - `I-X` nối vào span đang mở nếu span đó cũng loại X, ngược lại mở span mới.
      Nhận `I-X` mồ côi thay vì bỏ đi là có chủ ý: model thật vẫn sinh ra chuỗi
      nhãn không hợp lệ, bỏ đi là âm thầm mất thực thể.
    - Nhãn `O` đóng span đang mở.

    `start`/`end` lấy từ offset của từ ĐẦU và từ CUỐI trong span, nên
    `text[start:end]` luôn là đúng đoạn văn bản gốc (kể cả khoảng trắng ở giữa).
    """
    spans: List[Dict] = []
    current: Optional[Dict] = None

    for i, (word, label) in enumerate(zip(words, labels)):
        etype = entity_type(label)
        score = scores[i] if scores else None

        if etype is None:
            current = None
            continue

        starts_new = label.startswith("B-") or current is None or current["label"] != etype
        if starts_new:
            current = {"label": etype, "start": word.start, "end": word.end,
                       "_scores": [score] if score is not None else []}
            spans.append(current)
        else:
            current["end"] = word.end
            if score is not None:
                current["_scores"].append(score)

    out = []
    for s in spans:
        collected = s.pop("_scores")
        out.append({
            "text": text[s["start"]:s["end"]],
            "label": s["label"],
            "start": s["start"],
            "end": s["end"],
            # Điểm của cả span = điểm THẤP NHẤT trong các từ thành phần. Chọn min
            # (không phải trung bình) vì một span chỉ đáng tin bằng mắt xích yếu
            # nhất của nó: sai một từ là sai cả thực thể.
            "score": round(min(collected), 4) if collected else None,
        })
    return out


def spans_from_strings(text: str, items: List[Dict]) -> List[Dict]:
    """Đổi danh sách thực thể dạng chuỗi (LLM trả về) thành span có offset.

    LLM trả `{"text": "Nguyễn Viết Quang", "label": "PER"}` mà không kèm vị trí,
    nên phải tự dò lại trong văn bản gốc.

    Hai điểm cần cẩn thận, đều đã gặp thật:
    - Cụm DÀI phải được khớp TRƯỚC cụm ngắn. Nếu không, "Quang" sẽ chiếm chỗ và
      "Nguyễn Viết Quang" không còn khớp được nữa.
    - Một cụm có thể xuất hiện nhiều lần trong bài; đánh dấu hết các lần xuất
      hiện chưa bị chiếm chỗ, và không cho hai span chồng lấn nhau.

    Cụm nào không tìm thấy trong văn bản (LLM viết lại chữ, sai dấu) vẫn được
    giữ với `start = -1` và `matched = false` -- đó là tín hiệu chẩn đoán thật,
    giấu đi thì không ai biết LLM đang bịa.
    """
    taken = [False] * len(text)
    out: List[Dict] = []

    for item in sorted(items, key=lambda x: -len(x.get("text", ""))):
        needle = (item.get("text") or "").strip()
        label = str(item.get("label", "")).upper()
        if not needle or label not in VALID_LABELS:
            continue

        found_any = False
        pos = text.find(needle)
        while pos != -1:
            end = pos + len(needle)
            if not any(taken[pos:end]):
                for i in range(pos, end):
                    taken[i] = True
                out.append({"text": text[pos:end], "label": label,
                            "start": pos, "end": end, "score": None, "matched": True})
                found_any = True
            pos = text.find(needle, pos + 1)

        if not found_any:
            out.append({"text": needle, "label": label,
                        "start": -1, "end": -1, "score": None, "matched": False})

    out.sort(key=lambda s: (s["start"] < 0, s["start"]))
    return out
