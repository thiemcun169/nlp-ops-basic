# -*- coding: utf-8 -*-
"""Kiểm thử đơn vị cho phần logic thuần tuý: tách từ, gom nhãn, định vị span.

Chạy được mà KHÔNG cần Docker, không cần model, không cần mạng:

    python3 tests/test_units.py

Đây là tầng test rẻ nhất và chạy nhanh nhất, nên nó phải phủ đúng phần dễ sai
nhất. Ở dự án này, phần dễ sai nhất chính là tách từ: một lỗi ở đó không làm
chương trình dừng, chỉ âm thầm trả kết quả sai. Bài học rút ra từ đúng lỗi đã
xảy ra thật (xem docs/TROUBLESHOOTING.md muc E07).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/api"))

from app.core.decoding import group_bio, spans_from_strings   # noqa: E402
from app.core.tokenization import split_words                 # noqa: E402

passed = failed = 0


def check(name, actual, expected):
    global passed, failed
    if actual == expected:
        passed += 1
        print(f"  ĐẠT   {name}")
    else:
        failed += 1
        print(f"  LỖI   {name}\n        nhận  : {actual!r}\n        mong đợi: {expected!r}")


print("== Tách từ ==")
check("dấu câu tách khỏi từ",
      [w.text for w in split_words("anh Quang, 40 tuổi.")],
      ["anh", "Quang", ",", "40", "tuổi", "."])

check("offset trỏ đúng vào văn bản gốc",
      [(w.text, "Nguyễn Bá Thiêm ở Hà Nội."[w.start:w.end])
       for w in split_words("Nguyễn Bá Thiêm ở Hà Nội.")][:3],
      [("Nguyễn", "Nguyễn"), ("Bá", "Bá"), ("Thiêm", "Thiêm")])

check("ngày tháng và số thập phân không bị xé",
      [w.text for w in split_words("Ngày 4/9, giá tăng 1,5 lần lúc 10:30")],
      ["Ngày", "4/9", ",", "giá", "tăng", "1,5", "lần", "lúc", "10:30"])

check("dấu ngoặc kép và ngoặc đơn",
      [w.text for w in split_words('"Xin chào" (2026)')],
      ['"', "Xin", "chào", '"', "(", "2026", ")"])

check("chuỗi rỗng", [w.text for w in split_words("")], [])
check("chỉ khoảng trắng", [w.text for w in split_words("   \n  ")], [])
check("giữ nguyên dấu tiếng Việt",
      [w.text for w in split_words("Đà Nẵng ượm")], ["Đà", "Nẵng", "ượm"])


print("\n== Gom nhãn BIO thành span ==")
TEXT = "Nguyễn Bá Thiêm làm việc tại Google ở Hà Nội."
WORDS = split_words(TEXT)
LABELS = ["B-PER", "I-PER", "I-PER", "O", "O", "O", "B-ORG", "O", "B-LOC", "I-LOC", "O"]

check("gom đúng ba thực thể",
      [(e["text"], e["label"]) for e in group_bio(TEXT, WORDS, LABELS)],
      [("Nguyễn Bá Thiêm", "PER"), ("Google", "ORG"), ("Hà Nội", "LOC")])

check("text[start:end] khớp với text của thực thể",
      all(TEXT[e["start"]:e["end"]] == e["text"] for e in group_bio(TEXT, WORDS, LABELS)),
      True)

t2 = "Hà Nội Hải Phòng"
check("hai thực thể liền kề không bị dính làm một",
      [e["text"] for e in group_bio(t2, split_words(t2), ["B-LOC", "I-LOC", "B-LOC", "I-LOC"])],
      ["Hà Nội", "Hải Phòng"])

check("nhãn I- mồ côi vẫn mở span mới, không bị bỏ đi",
      [e["text"] for e in group_bio(t2, split_words(t2), ["I-LOC", "I-LOC", "O", "O"])],
      ["Hà Nội"])

check("đổi loại giữa chừng thì tách span",
      [(e["text"], e["label"])
       for e in group_bio(t2, split_words(t2), ["B-PER", "I-LOC", "O", "O"])],
      [("Hà", "PER"), ("Nội", "LOC")])

check("điểm của span lấy theo từ yếu nhất",
      group_bio(TEXT, WORDS, LABELS, [0.9, 0.5, 0.8] + [1.0] * 8)[0]["score"],
      0.5)

check("toàn nhãn O thì không có thực thể nào",
      group_bio(TEXT, WORDS, ["O"] * len(WORDS)), [])


print("\n== Định vị chuỗi do LLM trả về ==")
T = "Anh Nguyễn Viết Quang, 40 tuổi. Anh Quang đã tỉnh."

check("cụm dài được ưu tiên, không bị cụm ngắn chiếm chỗ",
      [(e["text"], e["start"]) for e in
       spans_from_strings(T, [{"text": "Quang", "label": "PER"},
                              {"text": "Nguyễn Viết Quang", "label": "PER"}])],
      [("Nguyễn Viết Quang", 4), ("Quang", 36)])

check("cụm không có trong văn bản bị đánh dấu matched=False",
      [(e["text"], e["start"], e["matched"]) for e in
       spans_from_strings(T, [{"text": "Trần Văn B", "label": "PER"}])],
      [("Trần Văn B", -1, False)])

check("nhãn không hợp lệ bị loại",
      spans_from_strings(T, [{"text": "Quang", "label": "MISC"}]), [])

overlap = [e for e in spans_from_strings(
    T, [{"text": "Nguyễn Viết Quang", "label": "PER"},
        {"text": "Viết Quang", "label": "PER"}]) if e["start"] >= 0]
check("cụm con nằm trong cụm đã lấy thì không lấy lại",
      [e["text"] for e in overlap], ["Nguyễn Viết Quang"])


print("\n== Dashboard Grafana ==")
# Chặn đúng loại lỗi đã gặp thật: một description viết bằng chuỗi nhiều dòng mà
# lỡ để dấu phẩy thừa trong ngoặc sẽ thành tuple -> JSON array -> Grafana báo
# "An unexpected error happened" và panel trắng trơn, không nói rõ vì sao.
import json as _json
_dash = _json.loads((Path(__file__).resolve().parents[1] /
                     "services/monitoring/grafana/dashboards/ner_serving.json")
                    .read_text(encoding="utf-8"))
_bad_desc = [p.get("id") for p in _dash["panels"]
             if not isinstance(p.get("description", ""), str)]
check("mọi description của panel đều là chuỗi", _bad_desc, [])

_bad_title = [p.get("id") for p in _dash["panels"] if not isinstance(p.get("title", ""), str)]
check("mọi title của panel đều là chuỗi", _bad_title, [])

_ids = [p.get("id") for p in _dash["panels"]]
check("mọi panel đều có id và không trùng nhau",
      (None in _ids, len(_ids) != len(set(_ids))), (False, False))

_no_target = [p.get("id") for p in _dash["panels"]
              if p.get("type") != "row" and not p.get("targets")]
check("mọi panel không phải row đều có truy vấn", _no_target, [])


print(f"\n{passed} đạt / {failed} lỗi")
sys.exit(1 if failed else 0)
