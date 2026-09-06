# Hợp đồng API

Dành cho người tích hợp: frontend, backend khác, hoặc chính bạn sáu tháng sau.

Đây là REST API, dữ liệu JSON. Bản đặc tả máy đọc được sinh thẳng từ mã nguồn
(`services/api/app/schemas.py`) nên không bao giờ lạc hậu:

|                                            | Địa chỉ                         |
| ------------------------------------------ | ---------------------------------- |
| Tài liệu tương tác, thử được ngay | http://localhost:8080/docs         |
| Đặc tả OpenAPI                          | http://localhost:8080/openapi.json |

---

## Các endpoint

| Method | Đường dẫn       | Công dụng                            |
| ------ | ------------------- | -------------------------------------- |
| GET    | `/health`         | dịch vụ phục vụ được không     |
| GET    | `/v1/models`      | liệt kê model gọi được           |
| POST   | `/v1/ner`         | nhận diện thực thể                 |
| POST   | `/v1/ner/compare` | chạy cùng văn bản qua nhiều model |

`/v1` là phiên bản của **hợp đồng**, không phải của phần mềm. App lên v3.7 mà
`/v1` vẫn giữ nguyên lời hứa.

---

## `POST /v1/ner`

```json
{
  "text": "Nguyễn Bá Thiêm làm việc tại Google ở Hà Nội.",
  "model": "phobert-ner-gpu",
  "options": { "return_scores": true }
}
```

| Trường                  | Kiểu  | Bắt buộc | Ghi chú                                                         |
| ------------------------- | ------ | ---------- | ---------------------------------------------------------------- |
| `text`                  | string | có        | 1 tới 5000 ký tự                                              |
| `model`                 | string | không     | bỏ trống thì dùng mặc định. Danh sách:`GET /v1/models` |
| `options.return_scores` | bool   | không     | kèm độ tin cậy. Chỉ có ở model tự host                   |

**Chỉ có MỘT trường `model`.** Tên nào là model tự host thì chạy tại chỗ, tên nào
khác thì hệ thống tự đẩy sang nhà cung cấp LLM. Client không cần biết khác biệt đó:

```bash
-d '{"text":"...", "model":"phobert-ner-gpu"}'   # chạy trên GPU
-d '{"text":"...", "model":"phobert-ner-cpu"}'   # chạy trên CPU
-d '{"text":"...", "model":"groq/compound"}'     # tự route sang LLM
```

Đưa `model` vào body chứ không tách đường dẫn riêng (`/v1/ner/gpu`, `/v1/ner/llm`)
vì như vậy client giữ nguyên một URL, và một phép A/B test chỉ khác đúng một chuỗi.

### Kết quả

```json
{
  "text": "Nguyễn Bá Thiêm làm việc tại Google ở Hà Nội.",
  "model": "phobert-ner-gpu",
  "runtime": "triton-gpu",
  "entities": [
    { "text": "Nguyễn Bá Thiêm", "label": "PER", "start": 0,  "end": 15, "score": 0.997 },
    { "text": "Google",          "label": "ORG", "start": 29, "end": 35, "score": 0.994 },
    { "text": "Hà Nội",          "label": "LOC", "start": 38, "end": 44, "score": 0.991 }
  ],
  "latency_ms": 7.8,
  "meta": { "num_words": 11, "truncated": false, "max_len": 128 }
}
```

**Cam kết quan trọng nhất:**

> Với mọi thực thể có `start >= 0`: `text[start:end] === entity.text`

Nhờ vậy frontend tô màu bằng `text.slice(start, end)`, không phải đi tìm chuỗi.
Bản đầu của giao diện tô bằng so khớp chuỗi và không tô được gì khi thực thể đứng
cạnh dấu câu (`"Quang"` không khớp `"Quang,"`).

| Trường               | Ý nghĩa                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| `label`              | `PER`, `ORG` hoặc `LOC`                                                           |
| `score`              | 0 tới 1,`null` nếu không yêu cầu. Điểm span lấy theo từ yếu nhất            |
| `matched`            | chỉ có ở model LLM.`false` = LLM trả cụm không có nguyên văn trong đầu vào |
| `runtime`            | nơi đã chạy:`triton-gpu`, `triton-cpu`, `onnx-local`, `llm`                  |
| `latency_ms`         | thời gian xử lý phía máy chủ, không tính đường mạng                          |
| `meta.truncated`     | văn bản vượt`max_len`, phần đuôi đã bị cắt                                  |
| `meta.fallback_from` | có mặt = model chính hỏng, kết quả này từ đường dự phòng                    |

### Ví dụ

```python
import requests
r = requests.post("http://localhost:8080/v1/ner",
                  json={"text": "Nguyễn Bá Thiêm làm việc tại Google ở Hà Nội."},
                  timeout=30)
r.raise_for_status()
for e in r.json()["entities"]:
    print(e["label"], e["text"])
```

```JavaScript
const { entities } = await (await fetch("/v1/ner", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text }),
})).json();

// Tô màu bằng offset, duyệt từ trái sang, không cần tìm kiếm chuỗi.
let html = "", cursor = 0;
for (const e of entities.filter(e => e.start >= 0).sort((a, b) => a.start - b.start)) {
  html += escapeHTML(text.slice(cursor, e.start));
  html += `<mark class="${e.label}">${escapeHTML(text.slice(e.start, e.end))}</mark>`;
  cursor = e.end;
}
html += escapeHTML(text.slice(cursor));
```

---

## `POST /v1/ner/compare`

```json
{ "text": "...", "models": ["phobert-ner-gpu", "phobert-ner-cpu", "groq/compound"] }
```

Trả về `results` (một phần tử mỗi model) và `agreements` (mức khớp giữa model đầu
và từng model còn lại).

1. **Một model hỏng không làm hỏng cả response.** Lỗi nằm ở `results[i].error`,
   HTTP vẫn `200`. Client phải kiểm `error` trước khi đọc `entities`.
2. **`agreements` là mức KHỚP NHAU, không phải độ chính xác.** Hai model cùng sai
   giống hệt nhau vẫn cho `f1 = 1.0`. Muốn độ chính xác thật thì dùng `benchmarks/`.

---

## Mã lỗi

| Mã     | Nghĩa                                             | Client nên làm gì                                           |
| ------- | -------------------------------------------------- | -------------------------------------------------------------- |
| `400` | đầu vào sai nội dung (chỉ có khoảng trắng) | sửa đầu vào,**đừng thử lại**                     |
| `422` | body sai kiểu hoặc thiếu trường               | sửa code,**đừng thử lại**                           |
| `502` | dịch vụ LLM bên ngoài lỗi                     | thử lại có lùi thời gian, hoặc đổi sang model tự host |
| `503` | model tự host không phục vụ được            | thử lại có lùi thời gian, kiểm`GET /health`            |

Thân lỗi theo dạng chuẩn của FastAPI: `{"detail": "..."}`.

Client dựa vào **mã** để quyết định thử lại hay không, nên đừng so khớp nội dung
thông báo lỗi. Trả `500` cho mọi thứ là ép người tích hợp phải đoán.

---

## Giới hạn và những điều KHÔNG cam kết

|                         | Giá trị                                                                    |
| ----------------------- | ---------------------------------------------------------------------------- |
| Độ dài văn bản     | tối đa 5000 ký tự                                                        |
| Độ dài model xử lý | 128 subword. Dài hơn thì cắt đuôi,`meta.truncated = true`            |
| Giới hạn tần suất   | không có ở dịch vụ này; model`llm` thì có, do nhà cung cấp đặt |
| Xác thực              | **chưa có**                                                          |

> **Chưa có xác thực.** Đây là hệ thống để học, mọi endpoint đều mở. Trước khi đưa
> ra Internet phải thêm ít nhất: xác thực, giới hạn tần suất theo người gọi, và
> CORS khai báo tường minh.

Đừng viết code phụ thuộc vào: thứ tự thực thể trong danh sách, giá trị chính xác
của `score` (huấn luyện lại là khác), các trường trong `meta`, và nội dung chuỗi
`detail`.

---

## Thay đổi hợp đồng

**Được phép bất cứ lúc nào:** thêm trường mới vào kết quả, thêm trường *tuỳ chọn*
vào request, thêm model mới, nới lỏng điều kiện kiểm tra.

**Phải ra `/v2`:** xoá hoặc đổi tên trường, thêm trường *bắt buộc*, đổi kiểu dữ
liệu, hoặc đổi ý nghĩa một trường mà giữ nguyên tên. Loại cuối nguy hiểm nhất:
không ai phát hiện ra cho tới khi có người tính toán sai.

Khi ra `/v2`, giữ `/v1` chạy song song đủ lâu để client chuyển sang.

Sinh client tự động từ đặc tả, đừng gõ tay lại kiểu dữ liệu:

```bash
npx openapi-typescript http://localhost:8080/openapi.json -o src/api-types.ts
```
