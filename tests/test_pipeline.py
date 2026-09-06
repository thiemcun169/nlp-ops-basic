# -*- coding: utf-8 -*-
"""Kiểm thử end-to-end: gọi API thật của hệ thống đang chạy.

    docker compose up -d
    python3 tests/test_pipeline.py

Khác với `test_units.py` (chỉ kiểm logic thuần), file này kiểm CẢ CHUỖI:
HTTP -> FastAPI -> tokenize -> Triton -> GPU -> giải mã nhãn -> JSON. Đúng những
gì người dùng thật đi qua.

Có hai loại khẳng định trong đây và chúng khác nhau về bản chất:

- Kiểm HỢP ĐỒNG (trường nào phải có, kiểu gì, mã lỗi nào): sai là hỏng thật, phải
  sửa ngay.
- Kiểm CHẤT LƯỢNG model (đoán đúng bao nhiêu thực thể): dùng ngưỡng rộng rãi, vì
  huấn luyện lại là con số sẽ khác. Muốn theo dõi chất lượng nghiêm túc thì dùng
  `benchmarks/run_benchmark.py`, đừng nhét ngưỡng chặt vào test hạ tầng -- test
  đỏ vì model nhích 1% là cách nhanh nhất để cả đội quen với việc bỏ qua test đỏ.

Không cần cài gì thêm: chỉ dùng thư viện chuẩn của Python.
"""
import json
import os
import sys
import urllib.error
import urllib.request

API = os.environ.get("API_URL", "http://localhost:8080")

passed = skipped = failed = 0


def report(status, name, detail=""):
    global passed, skipped, failed
    mark = {"ĐẠT": "ĐẠT ", "BỎ QUA": "BỎ QUA", "LỖI": "LỖI"}[status]
    print(f"{mark:7s} {name}" + (f" -- {detail}" if detail else ""))
    if status == "ĐẠT":
        passed += 1
    elif status == "BỎ QUA":
        skipped += 1
    else:
        failed += 1


def call(path, body=None, timeout=120):
    """Trả về (mã HTTP, dữ liệu). Không ném lỗi khi máy chủ trả 4xx/5xx --
    những mã đó chính là thứ ta cần kiểm tra."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(API + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


SENTENCE = "Nguyễn Bá Thiêm làm việc tại Google ở Hà Nội."
# Câu này CỐ TÌNH có dấu phẩy dính vào tên. Đó chính là lỗi tiền xử lý đã từng
# làm mất chữ cuối của mỗi tên người (xem TROUBLESHOOTING E07), nên nó phải nằm
# trong bộ test vĩnh viễn: lỗi đã sửa mà không có test canh thì sẽ quay lại.
PUNCT_SENTENCE = "Anh Nguyễn Viết Quang, 40 tuổi, sống ở xã Bình Minh, tỉnh Đồng Nai."


print(f"Kiểm thử hệ thống tại {API}\n")

# --- 1. Sức khoẻ hệ thống ---------------------------------------------------
status, health = call("/health")
if status != 200:
    report("LỖI", "GET /health", f"HTTP {status}")
    print("\nHệ thống chưa chạy? Thử: docker compose up -d")
    sys.exit(1)
report("ĐẠT", "GET /health", f"status={health['status']}, "
       f"nơi chạy sẵn sàng={sum(health['runtimes'].values())}/{len(health['runtimes'])}")

status, info = call("/v1/models")
ready = [m["name"] for m in info.get("models", []) if m["ready"]] if status == 200 else []
llm_models = [m["name"] for m in info.get("models", [])
              if m["ready"] and m["runtime"] == "llm"] if status == 200 else []
LOCAL = ["phobert-ner-gpu", "phobert-ner-cpu", "phobert-ner-local"]

# --- 2. Danh sách model -----------------------------------------------------
if status == 200 and info["models"] and "default" in info:
    report("ĐẠT", "GET /v1/models", f"{len(info['models'])} model, "
           f"mặc định={info['default']}")
else:
    report("LỖI", "GET /v1/models", f"HTTP {status}")

# --- 3. Dự đoán trên từng model tự host ------------------------------------
EXPECTED = {("Nguyễn Bá Thiêm", "PER"), ("Google", "ORG"), ("Hà Nội", "LOC")}

for model in LOCAL:
    if model not in ready:
        report("BỎ QUA", f"POST /v1/ner [{model}]", "model chưa sẵn sàng")
        continue
    status, data = call("/v1/ner", {"text": SENTENCE, "model": model,
                                    "options": {"return_scores": True}})
    if status != 200:
        report("LỖI", f"POST /v1/ner [{model}]", f"HTTP {status}: {data}")
        continue

    got = {(e["text"], e["label"]) for e in data["entities"]}
    offsets_ok = all(data["text"][e["start"]:e["end"]] == e["text"] for e in data["entities"])
    scores_ok = all(e.get("score") is not None for e in data["entities"])

    if not offsets_ok:
        report("LỖI", f"POST /v1/ner [{model}]", "offset không trỏ đúng vào văn bản")
    elif not EXPECTED <= got:
        report("LỖI", f"POST /v1/ner [{model}]", f"thiếu {EXPECTED - got}")
    elif not scores_ok:
        report("LỖI", f"POST /v1/ner [{model}]", "thiếu score dù đã yêu cầu")
    else:
        report("ĐẠT", f"POST /v1/ner [{model}]",
               f"{len(got)} thực thể đúng, {data['latency_ms']} ms, chạy ở {data['runtime']}")

# --- 4. Lỗi tiền xử lý đã sửa: dấu câu dính vào từ --------------------------
status, data = call("/v1/ner", {"text": PUNCT_SENTENCE})
if status != 200:
    report("LỖI", "Dấu câu dính vào từ", f"HTTP {status}")
else:
    got = {e["text"] for e in data["entities"]}
    # Từ CUỐI của mỗi tên là chỗ trước đây bị mất vì dính dấu phẩy.
    if "Nguyễn Viết Quang" in got:
        report("ĐẠT", "Dấu câu dính vào từ", "vẫn lấy đủ tên có dấu phẩy phía sau")
    else:
        report("LỖI", "Dấu câu dính vào từ",
               f"tên bị cắt cụt, nhận được: {sorted(got)}")

# --- 5. Ba model tự host phải cho KẾT QUẢ GIỐNG HỆT NHAU --------------------
# Cùng trọng số, cùng cách tokenize thì phải ra cùng kết quả. Lệch nhau nghĩa là
# có một nơi chạy dùng đường tiền xử lý riêng -- loại lỗi âm thầm nguy hiểm nhất.
hosted = [m for m in LOCAL if m in ready]
if len(hosted) >= 2:
    sets = {}
    for model in hosted:
        status, data = call("/v1/ner", {"text": PUNCT_SENTENCE, "model": model})
        sets[model] = {(e["start"], e["end"], e["label"]) for e in data.get("entities", [])}
    if len({frozenset(v) for v in sets.values()}) == 1:
        report("ĐẠT", "Các model tự host cho kết quả giống hệt nhau",
               f"{len(hosted)} model khớp")
    else:
        report("LỖI", "Các model tự host cho kết quả giống hệt nhau", f"lệch nhau: {sets}")
else:
    report("BỎ QUA", "Các model tự host cho kết quả giống hệt nhau", "cần ít nhất 2 model")

# --- 6. Kiểm tra đầu vào ----------------------------------------------------
status, _ = call("/v1/ner", {"text": ""})
report("ĐẠT" if status == 422 else "LỖI", "Chặn văn bản rỗng",
       f"HTTP {status} (mong đợi 422)")

status, _ = call("/v1/ner", {"text": "   "})
report("ĐẠT" if status == 400 else "LỖI", "Chặn văn bản chỉ có khoảng trắng",
       f"HTTP {status} (mong đợi 400)")

# Tên model lạ được coi là model của nhà cung cấp LLM, nên lỗi phải là 502
# (dịch vụ ngoài từ chối), KHÔNG phải 404. Đây là hệ quả trực tiếp của việc
# định tuyến theo tên: hệ thống không có danh sách đóng để mà báo "không tồn tại".
status, _ = call("/v1/ner", {"text": SENTENCE, "model": "khong-ton-tai-xyz"}, timeout=180)
report("ĐẠT" if status == 502 else "LỖI", "Tên model lạ được đẩy sang LLM rồi báo 502",
       f"HTTP {status} (mong đợi 502)")

status, _ = call("/v1/ner", {})
report("ĐẠT" if status == 422 else "LỖI", "Thiếu trường bắt buộc trả 422",
       f"HTTP {status} (mong đợi 422)")

# --- 7. So sánh nhiều model ------------------------------------------------
to_compare = [m for m in ["phobert-ner-gpu", "phobert-ner-cpu"] if m in ready]
if len(to_compare) >= 2:
    status, data = call("/v1/ner/compare", {"text": SENTENCE, "models": to_compare})
    if status == 200 and len(data["results"]) == len(to_compare) and data["agreements"]:
        report("ĐẠT", "POST /v1/ner/compare", f"F1 khớp nhau = {data['agreements'][0]['f1']}")
    else:
        report("LỖI", "POST /v1/ner/compare", f"HTTP {status}")
else:
    report("BỎ QUA", "POST /v1/ner/compare", "cần ít nhất 2 model tự host")

# --- 8. Đối chứng LLM (bỏ qua nếu chưa cấu hình key) ------------------------
if llm_models:
    llm_name = llm_models[0]
    status, data = call("/v1/ner", {"text": SENTENCE, "model": llm_name}, timeout=180)
    if status == 200:
        report("ĐẠT", f"POST /v1/ner [{llm_name}]",
               f"{len(data['entities'])} thực thể, {data['latency_ms']} ms, "
               f"structured={data['meta'].get('structured_mode')}, chạy ở {data['runtime']}")
    elif status == 502:
        # Hết hạn mức là chuyện bình thường với gói miễn phí, không phải lỗi hệ thống.
        report("BỎ QUA", f"POST /v1/ner [{llm_name}]", f"dịch vụ ngoài lỗi: {str(data)[:100]}")
    else:
        report("LỖI", f"POST /v1/ner [{llm_name}]", f"HTTP {status}: {str(data)[:160]}")
else:
    report("BỎ QUA", "POST /v1/ner [llm]", "chưa cấu hình LLM_API_KEY")

# --- 9. Giao diện và tài liệu ----------------------------------------------
status, html = call("/")
report("ĐẠT" if status == 200 and "<html" in str(html).lower() else "LỖI",
       "GET / (trang demo)", f"HTTP {status}")

status, spec = call("/openapi.json")
if status == 200 and "/v1/ner" in spec.get("paths", {}):
    report("ĐẠT", "GET /openapi.json", f"{len(spec['paths'])} đường dẫn có tài liệu")
else:
    report("LỖI", "GET /openapi.json", f"HTTP {status}")

print(f"\n{passed} đạt / {skipped} bỏ qua / {failed} lỗi")
sys.exit(1 if failed else 0)
