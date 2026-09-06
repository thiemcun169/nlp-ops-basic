# -*- coding: utf-8 -*-
"""Sinh file `gold_vi.json` từ phần khai báo bên dưới.

Vì sao không gõ thẳng offset vào JSON: đếm ký tự bằng tay trong câu tiếng Việt
có dấu là việc chắc chắn sẽ sai. Ở đây chỉ khai báo *chuỗi thực thể* và *lần
xuất hiện thứ mấy*, còn offset do máy tính ra rồi tự kiểm tra lại.

Chạy lại sau khi sửa/thêm câu:
    python3 benchmarks/build_gold.py

MỘT LƯU Ý THẲNG THẮN: bộ này được sinh ra trong lúc dựng repo bằng công cụ AI,
không qua kiểm chứng chéo bởi người gán nhãn độc lập, và chỉ có vài chục câu. Nó đủ để soi các tình huống đã chọn
có chủ ý (dấu câu, viết thường, tên tổ chức dài, bẫy danh từ chung) và để so hai
hệ thống với nhau. Nó KHÔNG đủ để công bố một con số F1 như năng lực thật của
model: muốn vậy cần tập test độc lập hàng nghìn câu, nhiều người gán nhãn, và đo
độ đồng thuận giữa họ.
"""
import json
from pathlib import Path

# (nhóm, câu, [(chuỗi thực thể, nhãn, lần xuất hiện thứ mấy tính từ 0)])
SAMPLES = [
    # ---- Nhóm 1: giống dữ liệu huấn luyện (danh từ riêng viết hoa chuẩn) ----
    ("chuẩn", "Nguyễn Bá Thiêm làm việc tại Google ở Hà Nội.",
     [("Nguyễn Bá Thiêm", "PER", 0), ("Google", "ORG", 0), ("Hà Nội", "LOC", 0)]),
    ("chuẩn", "Việt Nam và Hoa Kỳ ký thoả thuận hợp tác về công nghệ.",
     [("Việt Nam", "LOC", 0), ("Hoa Kỳ", "LOC", 0)]),
    ("chuẩn", "Phạm Nhật Vượng sáng lập VinGroup tại Hải Phòng.",
     [("Phạm Nhật Vượng", "PER", 0), ("VinGroup", "ORG", 0), ("Hải Phòng", "LOC", 0)]),
    ("chuẩn", "Trường Đại học Bách khoa Hà Nội tuyển sinh thêm ngành trí tuệ nhân tạo.",
     [("Trường Đại học Bách khoa Hà Nội", "ORG", 0)]),
    ("chuẩn", "Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập tại Quảng trường Ba Đình.",
     [("Hồ Chí Minh", "PER", 0), ("Quảng trường Ba Đình", "LOC", 0)]),

    # ---- Nhóm 2: dấu câu dính vào từ (lỗi tiền xử lý đã sửa) ----
    ("dấu câu", "Anh Nguyễn Viết Quang, 40 tuổi, là nạn nhân duy nhất sống sót.",
     [("Nguyễn Viết Quang", "PER", 0)]),
    ("dấu câu", "Vợ anh là chị Hồ Thị Mai, 36 tuổi, quê ở Bến Tre.",
     [("Hồ Thị Mai", "PER", 0), ("Bến Tre", "LOC", 0)]),
    ("dấu câu", "Theo lời khai, Nguyễn Đức Anh (21 tuổi) đã đến nhà nạn nhân vào rạng sáng 3/9.",
     [("Nguyễn Đức Anh", "PER", 0)]),
    ("dấu câu", "Ông Trần Văn Hùng - giám đốc Sở Y tế Đồng Nai - xác nhận thông tin trên.",
     [("Trần Văn Hùng", "PER", 0), ("Sở Y tế Đồng Nai", "ORG", 0)]),
    ("dấu câu", "\"Chúng tôi sẽ điều tra\", đại diện Công an tỉnh Bình Dương nói.",
     [("Công an tỉnh Bình Dương", "ORG", 0)]),

    # ---- Nhóm 3: tên riêng lặp lại nhiều lần trong đoạn dài ----
    ("đoạn dài",
     "Nạn nhân cuối trong vụ thảm sát ở Đồng Nai đã qua cơn nguy kịch. "
     "Sau ca phẫu thuật, anh Nguyễn Viết Quang, 40 tuổi, đã tỉnh lại. "
     "Ngày 4/9, sức khỏe anh Quang ổn định hơn nhưng vẫn còn mệt.",
     [("Đồng Nai", "LOC", 0), ("Nguyễn Viết Quang", "PER", 0), ("Quang", "PER", 1)]),
    ("đoạn dài",
     "Công ty Cổ phần FPT công bố kết quả kinh doanh quý III. "
     "Đại diện FPT cho biết doanh thu từ thị trường Nhật Bản tăng mạnh, "
     "trong khi mảng viễn thông tại Thành phố Hồ Chí Minh giữ nguyên.",
     [("Công ty Cổ phần FPT", "ORG", 0), ("FPT", "ORG", 1),
      ("Nhật Bản", "LOC", 0), ("Thành phố Hồ Chí Minh", "LOC", 0)]),

    # ---- Nhóm 4: viết thường hết (data drift từ bình luận, chat) ----
    ("viết thường", "nguyễn văn a sống ở đà nẵng và làm cho fpt software.",
     [("nguyễn văn a", "PER", 0), ("đà nẵng", "LOC", 0), ("fpt software", "ORG", 0)]),
    ("viết thường", "hôm qua mình đi hà nội chơi với thằng tuấn.",
     [("hà nội", "LOC", 0), ("tuấn", "PER", 0)]),

    # ---- Nhóm 5: tên nước ngoài xen tiếng Việt ----
    ("nước ngoài", "CEO Tim Cook của Apple đến thăm Thành phố Hồ Chí Minh.",
     [("Tim Cook", "PER", 0), ("Apple", "ORG", 0), ("Thành phố Hồ Chí Minh", "LOC", 0)]),
    ("nước ngoài", "Elon Musk thông báo Tesla sẽ mở nhà máy mới tại Berlin.",
     [("Elon Musk", "PER", 0), ("Tesla", "ORG", 0), ("Berlin", "LOC", 0)]),
    ("nước ngoài", "Nhóm nghiên cứu của Google DeepMind công bố kết quả trên tạp chí Nature.",
     [("Google DeepMind", "ORG", 0), ("Nature", "ORG", 0)]),

    # ---- Nhóm 6: tổ chức (loại yếu nhất của model, F1 thấp nhất khi đánh giá) ----
    ("tổ chức", "Ngân hàng Nhà nước Việt Nam giữ nguyên lãi suất điều hành.",
     [("Ngân hàng Nhà nước Việt Nam", "ORG", 0)]),
    ("tổ chức", "Bộ Giáo dục và Đào tạo công bố phương án thi tốt nghiệp.",
     [("Bộ Giáo dục và Đào tạo", "ORG", 0)]),
    ("tổ chức", "Câu lạc bộ Hoàng Anh Gia Lai thắng Hà Nội FC trên sân Pleiku.",
     [("Câu lạc bộ Hoàng Anh Gia Lai", "ORG", 0), ("Hà Nội FC", "ORG", 0),
      ("Pleiku", "LOC", 0)]),

    # ---- Nhóm 7: bẫy — danh từ chung trông giống tên riêng ----
    ("bẫy", "Anh ấy làm ở một công ty công nghệ tại quận trung tâm.", []),
    ("bẫy", "Giá vàng trong nước tăng 500.000 đồng mỗi lượng trong sáng nay.", []),
    ("bẫy", "Sông Hồng chảy qua nhiều tỉnh phía Bắc trước khi đổ ra Biển Đông.",
     [("Sông Hồng", "LOC", 0), ("Biển Đông", "LOC", 0)]),
    ("bẫy", "Cô ấy học tiếng Anh ở trung tâm gần nhà.", []),

    # ---- Nhóm 8: nhiều thực thể liền kề, dễ dính thành một ----
    ("liền kề", "Hà Nội, Hải Phòng, Đà Nẵng và Cần Thơ là bốn thành phố trực thuộc trung ương.",
     [("Hà Nội", "LOC", 0), ("Hải Phòng", "LOC", 0), ("Đà Nẵng", "LOC", 0),
      ("Cần Thơ", "LOC", 0)]),
    ("liền kề", "Nguyễn An và Trần Bình cùng làm việc tại Viettel.",
     [("Nguyễn An", "PER", 0), ("Trần Bình", "PER", 0), ("Viettel", "ORG", 0)]),
]


def locate(text: str, needle: str, occurrence: int) -> int:
    pos = -1
    for _ in range(occurrence + 1):
        pos = text.find(needle, pos + 1)
        if pos == -1:
            raise ValueError(
                f"Không tìm thấy lần xuất hiện thứ {occurrence} của {needle!r} trong {text!r}")
    return pos


def main() -> None:
    out = []
    for idx, (group, text, entities) in enumerate(SAMPLES):
        spans = []
        for needle, label, occurrence in entities:
            start = locate(text, needle, occurrence)
            end = start + len(needle)
            # Kiểm tra lại ngay: offset sai thì hỏng toàn bộ phép chấm điểm.
            assert text[start:end] == needle, (text, needle)
            spans.append({"text": needle, "label": label, "start": start, "end": end})
        spans.sort(key=lambda s: s["start"])

        # Không cho hai thực thể chồng lấn nhau: đó là lỗi gán nhãn.
        for a, b in zip(spans, spans[1:]):
            assert a["end"] <= b["start"], f"Nhãn chồng lấn trong câu: {text!r}"

        out.append({"id": f"s{idx:03d}", "group": group, "text": text, "entities": spans})

    path = Path(__file__).with_name("gold_vi.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    total = sum(len(s["entities"]) for s in out)
    by_label = {}
    for s in out:
        for e in s["entities"]:
            by_label[e["label"]] = by_label.get(e["label"], 0) + 1
    print(f"Đã ghi {path}")
    print(f"  {len(out)} câu, {total} thực thể: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_label.items())))
    groups = {}
    for s in out:
        groups[s["group"]] = groups.get(s["group"], 0) + 1
    print("  Theo nhóm: " + ", ".join(f"{k}={v}" for k, v in groups.items()))


if __name__ == "__main__":
    main()
