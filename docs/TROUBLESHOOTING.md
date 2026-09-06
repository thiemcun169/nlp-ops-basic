# Sổ tay xử lý sự cố

Toàn bộ lỗi dưới đây là lỗi **đã gặp thật** khi dựng hệ thống này. Phần "Bài học"
mới là thứ đáng nhớ: lỗi thì gắn với dự án này, cách nghĩ thì dùng lại được.

**Luôn xem log trước khi đoán:**

```bash
docker compose ps
docker compose logs api --tail 100
docker compose logs triton --tail 100
curl -s localhost:8080/health | python3 -m json.tool
```

| Mã | Lỗi |
|---|---|
| [E01](#e01) | Notebook ghi artifact ra sai thư mục |
| [E02](#e02) | `pip install` trong Docker build treo rất lâu |
| [E03](#e03) | Lỗi 500 ngẫu nhiên chỉ khi tải cao |
| [E04](#e04) | LLM trả 404 `model_not_found` |
| [E05](#e05) | Lỗi dịch vụ ngoài làm sập cả endpoint |
| [E06](#e06) | Triton chạy nhưng model nằm trên CPU |
| [E07](#e07) | Model bỏ sót chữ cuối của tên người |
| [E08](#e08) | Cổng đã bị chiếm |
| [E09](#e09) | Grafana không hiện dữ liệu |
| [E10](#e10) | LLM trả 429, hệ thống báo lỗi sai nguyên nhân |
| [E11](#e11) | Model không nhận ra gì khi văn bản viết thường |
| [E12](#e12) | Clone mới build hỏng: `"/wheels": not found` |

---

<a name="e01"></a>
## E01. Notebook ghi artifact ra sai thư mục

**Triệu chứng.** Notebook báo đã lưu, nhưng `docker compose up` lỗi không tìm thấy
`model.onnx`. Hoặc file 538 MB xuất hiện ở một thư mục lạ.

**Chẩn đoán.** `import os; print(os.getcwd())`

**Nguyên nhân.** Đường dẫn tương đối trong notebook tính theo **thư mục đang chạy
kernel**, không theo vị trí file `.ipynb`.

**Cách sửa.** Neo vào một mốc cố định, và xuất trọng số **thẳng** vào kho model
chứ không qua file tạm:

```python
ROOT = Path.cwd()
while not (ROOT / "docker-compose.yml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
ONNX_PATH = ROOT / "services/triton/model_repository/ner_onnx/1/model.onnx"
```

**Bài học.** Đường dẫn tương đối trong notebook là nguồn lỗi âm thầm. Luôn in
đường dẫn tuyệt đối và `assert` file tồn tại sau khi lưu. Artifact vài trăm MB mà
để lang thang thì sớm muộn cũng có người commit nhầm.

---

<a name="e02"></a>
## E02. `pip install` trong Docker build treo rất lâu

**Triệu chứng.** `docker compose build` đứng im ở `pip install` hơn 10 phút.

**Chẩn đoán.** So tốc độ mạng trong container với trên host:

```bash
docker run --rm python:3.11-slim pip download --no-deps -d /tmp requests
python3 -m pip download --no-deps -d /tmp requests
```

**Nguyên nhân.** Mạng bridge của Docker trên máy bị bóp băng thông rất nặng: đo
được ~20-30 kB/s trong container so với >100 MB/s trên host.

**Cách sửa.** `bash scripts/download_wheels.sh` rồi build lại. Từ treo hơn 15 phút
xuống 8 giây. Không có thư mục `wheels/` thì Dockerfile tự cài từ mạng.

**Bài học.** Khi một bước build chậm bất thường, **đo riêng từng thành phần**
(mạng, đĩa, CPU) thay vì sửa mò cấu hình.

---

<a name="e03"></a>
## E03. Lỗi 500 ngẫu nhiên chỉ khi tải cao

**Triệu chứng.** Một request thì bình thường. 50+ user thì một tỷ lệ nhỏ trả 500,
log không có traceback.

**Chẩn đoán.** Việc đầu tiên là làm cho lỗi **hiện traceback**:

```python
except Exception:
    logger.exception("Gọi Triton thất bại")   # .exception() in cả stack trace
    raise
```

```
Cannot switch to a different thread
    Current:  <greenlet.greenlet object ...>
```

**Nguyên nhân.** `tritonclient.http` dùng gevent/greenlet. Greenlet gắn chặt với
**đúng một OS thread**, trong khi FastAPI chạy handler đồng bộ trong threadpool
nhiều thread. Dùng chung một client là thỉnh thoảng đụng độ.

**Thử sai đã ghi nhận.** Tăng `concurrency` **không sửa được, còn tệ hơn**: 44,87%
lỗi ở 80 user. Bản chất là xung đột thread, không phải thiếu kết nối.

**Cách sửa.** `threading.local()`, mỗi thread một client
(`app/backends/triton_backend.py`). Sau khi sửa: 0% lỗi ở 80 và 150 user.

**Bài học.** Lỗi chỉ xuất hiện khi tải cao gần như **luôn** là vấn đề đồng thời.
Bước đầu tiên là làm cho lỗi hiện traceback, không phải đoán.

---

<a name="e04"></a>
## E04. LLM trả 404 `model_not_found`

**Triệu chứng.** `The model 'llama-3.3-70b-versatile' does not exist`

**Chẩn đoán.** Hỏi thẳng nhà cung cấp key của bạn có model nào:

```python
print(sorted(m.id for m in OpenAI(api_key=KEY, base_url=URL).models.list().data))
```

**Nguyên nhân.** Nhà cung cấp thường xuyên gỡ và đổi tên model. Một tên viết cứng
trong code là quả bom hẹn giờ: hôm nay chạy, tháng sau hỏng mà không ai đụng vào.

**Cách sửa.** Tự dò model khả dụng theo danh sách ưu tiên
(`app/backends/llm_backend.py::resolve_model`). Kiểm bằng
`curl -s localhost:8080/health`. Muốn ép một model thì truyền thẳng trong body:
`{"model": "groq/compound"}`.

**Bài học.** Mọi phụ thuộc bên ngoài đều có thể biến mất. Hãy tự dò khả năng thay
vì giả định, báo lỗi kèm cách sửa, và đừng để nó làm sập phần còn lại.

---

<a name="e05"></a>
## E05. Lỗi dịch vụ ngoài làm sập cả endpoint

**Triệu chứng.** `/v1/ner/compare` trả 500 hoàn toàn, dù model tự host vẫn chạy tốt.

**Nguyên nhân.** Exception từ phụ thuộc **không thiết yếu** lan ra ngoài và giết
luôn cả request.

**Cách sửa.** Phần chính vẫn trả về, phần phụ báo lỗi ở trường riêng
(`results[i].error`), HTTP vẫn 200.

**Bài học.** Phân loại phụ thuộc thành *thiết yếu* và *không thiết yếu* ngay từ khi
thiết kế. Phụ thuộc không thiết yếu hỏng thì hệ thống suy giảm chức năng, không
được sập. Và người gọi phải **nhìn thấy** phần nào đã suy giảm.

---

<a name="e06"></a>
## E06. Triton chạy nhưng model nằm trên CPU

**Triệu chứng.** Máy có GPU, `nvidia-smi` trong container thấy card, nhưng vẫn chậm.

**Chẩn đoán.**

```bash
docker compose logs triton | grep ModelInstanceInitialize
# ner_onnx_0_0 (CPU device 0)   <-- sai
# ner_onnx_0_0 (GPU device 0)   <-- đúng
```

**Nguyên nhân.** Cần **cả hai**, thiếu một là rơi về CPU: `config.pbtxt` khai
`kind: KIND_GPU`, và `docker-compose.yml` cấp GPU cho container.

**Bài học.** "Container thấy GPU" khác với "model chạy trên GPU". Kiểm chứng bằng
log khởi tạo instance, đừng tin cảm giác về tốc độ.

---

<a name="e07"></a>
## E07. Model bỏ sót chữ cuối của tên người

**Triệu chứng.** `"anh Nguyễn Viết Quang, 40 tuổi"` trả về `Nguyễn` và `Viết`
nhưng **thiếu `Quang`**. Không lỗi, không cảnh báo, HTTP vẫn 200.

**Chẩn đoán.** So cách tách từ lúc phục vụ với dữ liệu huấn luyện:

```python
print(text.split())               # ['anh', 'Nguyễn', 'Viết', 'Quang,', ...]
print(raw["train"][0]["tokens"])  # [..., 'Quang', ',', ...]  dấu câu là token RIÊNG
```

**Nguyên nhân.** **Lệch tiền xử lý giữa huấn luyện và phục vụ.** Token `Quang,`
chưa từng xuất hiện lúc huấn luyện nên model gán `O`, và thực thể đứt tại đó.

Đo được trên một đoạn 3 câu: `text.split()` cho 7 từ được gán nhãn, tách riêng dấu
câu cho 11.

**Cách sửa.** Một hàm tách từ duy nhất dùng chung
(`app/core/tokenization.py`), và một test hồi quy canh riêng trong
`tests/test_pipeline.py`. Mục 8 của notebook tái hiện lại toàn bộ.

**Bài học.** Lệch tiền xử lý là lỗi phổ biến nhất trong vận hành mô hình, và là
loại tệ nhất: không exception, không log đỏ, chỉ có kết quả sai. Hai cách phòng
duy nhất đáng tin: **dùng chung một hàm** và **có test hồi quy**.

---

<a name="e08"></a>
## E08. Cổng đã bị chiếm

**Triệu chứng.** `Bind for 0.0.0.0:8080 failed: port is already allocated`

**Chẩn đoán.** `ss -tlnp | grep 8080` (Linux) hoặc `lsof -i :8080` (macOS).

**Cách sửa.** Đổi trong `.env`, **không cần sửa `docker-compose.yml`**:

```bash
API_PORT=18080
GRAFANA_PORT=13001
```

Rồi chạy test với `API_URL=http://localhost:18080 python3 tests/test_pipeline.py`.

**Bài học.** Cổng là tài nguyên dùng chung của cả máy. Mọi cổng nên đọc được từ
biến môi trường ngay từ đầu.

---

<a name="e09"></a>
## E09. Grafana không hiện dữ liệu

**Triệu chứng.** Dashboard mở được nhưng mọi biểu đồ báo "No data".

**Chẩn đoán.** Lần ngược đường đi của số đo:

```bash
curl -s localhost:8002/metrics | head                        # Triton có xuất không
curl -s localhost:9490/api/v1/targets | python3 -m json.tool  # Prometheus thu được không
curl -s localhost:3001/api/datasources                        # Grafana có nguồn chưa
```

**Ba nguyên nhân, theo thứ tự phổ biến.**

1. **Chưa có lưu lượng nào.** Triton chỉ sinh metrics khi có request. Panel trống
   là **đúng** nếu bạn chưa gọi API lần nào.
2. Prometheus không tới được Triton: phải dùng **tên service** (`triton:8002`),
   không phải `localhost:8002`. Mỗi container có `localhost` riêng.
3. Sai cổng: stack này dùng **3001** và **9490**.

**Bài học.** Dashboard trống thì **đừng sửa dashboard trước**. Lần ngược chuỗi
nguồn → thu thập → hiển thị, kiểm từng khâu bằng một lệnh riêng.

---

<a name="e10"></a>
## E10. LLM trả 429, hệ thống báo lỗi sai nguyên nhân

**Triệu chứng.** Chạy chấm điểm hàng loạt thì phần lớn câu lỗi với thông báo
"Không chế độ structured output nào chạy được". Thông báo này **sai**.

**Chẩn đoán.** Đọc mã lỗi cuối trong chuỗi: `429` là quá hạn mức tần suất, không
liên quan gì tới định dạng đầu ra.

**Nguyên nhân.** Đây là **lỗi của chính code này**. Vòng thử các chế độ structured
output bắt `except Exception` rồi hạ bậc, nên gặp 429 nó thử cả ba bậc (đốt thêm
quota) rồi báo một nguyên nhân bịa ra.

**Cách sửa.** Chỉ bắt đúng loại lỗi mình biết cách xử lý, và thử lại có lùi thời
gian theo header `Retry-After`:

```python
except Exception as e:
    if not _is_unsupported_format_error(e):
        raise                   # 429, 401, mạng hỏng: hạ bậc không giúp gì
```

**Bài học.** `except Exception` rồi xử lý mọi thứ như nhau là cách nhanh nhất để
**giấu mất nguyên nhân thật**. Một thông báo lỗi sai còn tệ hơn không có thông báo,
vì nó khiến người đọc đi sửa nhầm chỗ.

---

<a name="e11"></a>
## E11. Model không nhận ra gì khi văn bản viết thường

**Triệu chứng.** `"nguyễn văn a sống ở đà nẵng"` trả về gần như rỗng, trong khi
cùng câu viết hoa chuẩn thì nhận ra hết.

**Chẩn đoán.** `python3 benchmarks/run_benchmark.py` rồi nhìn cột theo nhóm tình
huống: F1 nhóm "viết thường" bằng **0,000**.

**Nguyên nhân.** **Trôi dữ liệu**, không phải lỗi kỹ thuật. WikiAnn lấy từ
Wikipedia nên viết hoa chuẩn; model học chữ hoa là tín hiệu mạnh của tên riêng, và
tín hiệu đó biến mất trong tin nhắn hay bình luận.

**Cách sửa.** Không có bản vá nhanh. Ba hướng theo công sức tăng dần: chuẩn hoá
hoa thường ở tiền xử lý; bổ sung bản viết thường của chính các câu train vào tập
huấn luyện (hiệu quả nhất so với công sức); hoặc chuyển sang model LLM cho nhóm
văn bản này.

**Bài học.** Model chỉ tốt bằng dữ liệu đã thấy. Trước khi triển khai, hãy tự hỏi
**văn bản thật của người dùng khác dữ liệu huấn luyện ở chỗ nào**, rồi dựng bộ
chấm điểm có đúng những tình huống đó.

---

<a name="e12"></a>
## E12. Clone mới build hỏng: `"/wheels": not found`

**Triệu chứng.** Máy đang phát triển thì build tốt, nhưng người khác clone repo về
là hỏng ngay ở bước đầu:

```
ERROR: failed to compute cache key: "/wheels": not found
```

**Chẩn đoán.** Dựng lại đúng những gì Git thực sự có, đừng tin thư mục làm việc:

```bash
mkdir /tmp/fresh && git archive HEAD | tar -x -C /tmp/fresh
cd /tmp/fresh/services/api && docker build .
```

**Nguyên nhân.** Dockerfile có `COPY wheels ./wheels`, nhưng `.gitignore` lại chặn
`services/api/wheels/`. Trên máy phát triển thư mục đó tồn tại nên không ai thấy
gì bất thường; trên clone mới nó không tồn tại và `COPY` hỏng.

Đây là **lỗi chỉ xuất hiện với người khác**, và là loại lỗi dễ lọt nhất: người
viết repo không bao giờ tự gặp.

**Cách sửa.** Giữ thư mục trong Git nhưng bỏ qua nội dung:

```gitignore
services/api/wheels/*
!services/api/wheels/.gitkeep
```

Sửa xong lại lộ ra lỗi thứ hai: điều kiện `[ -n "$(ls -A ./wheels)" ]` thấy chính
file `.gitkeep` nên tưởng có wheel, rồi chạy `pip --no-index` với thư mục không có
gói nào. Điều kiện phải hỏi đúng thứ mình cần:

```dockerfile
RUN if ls ./wheels/*.whl >/dev/null 2>&1; then ... ; else ... ; fi
```

**Bài học chung.** Mọi thứ `.gitignore` chặn mà Dockerfile hoặc script lại cần là
một quả mìn chờ người khác giẫm. Trước khi giao repo, hãy **dựng lại từ đúng nội
dung Git** (`git archive`) và chạy thử, thay vì tin vào thư mục làm việc của mình.
Và khi kiểm tra một điều kiện, hãy hỏi đúng thứ mình cần ("có file .whl không")
chứ không hỏi thứ gần đúng ("thư mục có rỗng không").

---

## Vẫn không xử lý được

```bash
docker compose ps                                    # container nào chết
docker compose logs triton --tail 50
curl -s localhost:8080/health | python3 -m json.tool
python3 tests/test_units.py                          # logic thuần (không cần Docker)
python3 tests/test_pipeline.py                       # khâu nào trong chuỗi bị đứt
nvidia-smi && df -h .
```

Dựng lại sạch: `docker compose down -v && docker compose up --build -d`
