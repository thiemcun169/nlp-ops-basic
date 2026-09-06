# Cài đặt trên Windows và macOS

> Tài liệu chính là [SETUP_LINUX.md](SETUP_LINUX.md): hệ thống được phát triển và kiểm chứng đầy đủ
> trên Ubuntu 24.04. Trang này chỉ ghi những chỗ **khác đi** ở hai hệ điều hành còn lại, dựa trên
> cấu hình đã chạy được trên Linux.

Điểm cần biết trước khi bắt đầu, để khỏi mất thời gian:

| | Windows (WSL 2) | macOS (Apple Silicon) |
|---|---|---|
| Chạy được toàn bộ hệ thống | có | có |
| GPU cho Triton | có, nếu là card NVIDIA | **không**. Triton không hỗ trợ Metal |
| GPU cho huấn luyện | có | một phần (PyTorch dùng được MPS) |
| Cách làm được khuyến nghị | WSL 2 rồi làm y như Linux | chạy CPU tại chỗ, huấn luyện trên Colab |

---

## Windows

### Vì sao phải dùng WSL 2

Docker trên Windows chạy bên trong WSL 2. Muốn container thấy được GPU NVIDIA thì bắt buộc đi qua
WSL 2, không có đường nào khác. Ngoài ra mọi script `.sh` trong repo đều là script bash.

Kết luận thực dụng: **cài WSL 2, clone repo vào bên trong WSL, rồi làm theo đúng
[SETUP_LINUX.md](SETUP_LINUX.md)**. Cách này ít lệch nhất so với môi trường đã kiểm chứng.

### Các bước

**1. Bật WSL 2 với Ubuntu.** Mở PowerShell **quyền Administrator**:

```powershell
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
```

Khởi động lại máy, rồi mở Ubuntu từ menu Start và đặt tên người dùng, mật khẩu.

**2. Cài Docker Desktop.** Tải từ https://www.docker.com/products/docker-desktop

Trong Settings, bật hai mục:
- General > **Use the WSL 2 based engine**
- Resources > WSL Integration > bật cho **Ubuntu-24.04**

**3. GPU NVIDIA (bỏ qua nếu không có).**

Cài driver NVIDIA **cho Windows** (bản thường, có hỗ trợ WSL sẵn từ 2021). **Không** cài driver
Linux bên trong WSL, làm vậy sẽ hỏng. Kiểm tra bên trong Ubuntu của WSL:

```bash
nvidia-smi                                    # phải in ra bảng GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

NVIDIA Container Toolkit thường đã được Docker Desktop lo sẵn. Nếu lệnh thứ hai lỗi thì cài thủ
công theo mục 2 của SETUP_LINUX.md, chạy bên trong Ubuntu của WSL.

**4. Clone repo VÀO BÊN TRONG WSL, không phải vào ổ C.**

```bash
# Trong terminal Ubuntu:
cd ~
git clone <địa-chỉ-repo> nlp-ops-basic
cd nlp-ops-basic
```

> Đây là điều quan trọng nhất trong cả trang này. Để repo ở `/mnt/c/Users/...` khiến Docker phải
> đọc ghi qua lớp chuyển tiếp giữa hai hệ thống tệp, chậm hơn hàng chục lần. Riêng bước
> `docker compose build` có thể từ vài giây thành vài phút. Để trong `~` của WSL là ổ đĩa Linux thật.

**5. Từ đây trở đi làm y hệt SETUP_LINUX.md** từ mục 3.

### Những chỗ khác biệt còn lại trên Windows

| Vấn đề | Cách xử lý |
|---|---|
| Liên kết tượng trưng cho model CPU | Notebook tự phát hiện và **chép hẳn file** thay vì tạo liên kết. Tốn thêm 538 MB đĩa, chạy vẫn đúng. Muốn dùng liên kết thì bật Developer Mode trong Windows Settings |
| Kết thúc dòng CRLF làm hỏng script `.sh` | `git config --global core.autocrlf input` **trước khi** clone |
| Mở giao diện | Dùng `http://localhost:8080` ngay trên trình duyệt Windows, WSL 2 tự chuyển tiếp cổng |
| WSL ăn hết RAM | Tạo file `C:\Users\<tên>\.wslconfig` với nội dung `[wsl2]` và `memory=12GB` |
| Docker Desktop báo hết đĩa | Settings > Resources > tăng dung lượng, hoặc `docker system prune -a` |

---

## macOS

### Điều phải biết trước

**Triton không chạy được trên GPU của Mac.** Không có bản Triton nào hỗ trợ Metal hay Apple
Silicon GPU. Trên Mac, mọi suy luận đều chạy trên CPU. Đây là giới hạn của Triton, không phải của
repo này.

Điều đó vẫn ổn cho việc học: toàn bộ kiến trúc, API, giám sát, đo tải đều chạy đầy đủ. Chỉ có phần
so sánh GPU với CPU là không thực hiện được tại chỗ.

### Các bước

**1. Cài Docker Desktop for Mac** từ https://www.docker.com/products/docker-desktop
(chọn đúng bản Apple Silicon hoặc Intel).

Settings > Resources: cấp ít nhất **8 GB RAM** và **30 GB đĩa**. Mặc định thường quá thấp và
Triton sẽ bị hệ thống kết liễu giữa chừng mà không nói rõ lý do.

**2. Bỏ qua bước dựng image Triton nhẹ.** Script `build_triton_image.sh` cần môi trường Linux
x86_64. Dùng thẳng image chính thức:

```bash
cp .env.example .env
echo "TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.12-py3" >> .env
```

Trên Apple Silicon, Docker sẽ chạy image x86_64 này qua lớp giả lập Rosetta. Nó **chạy được nhưng
chậm**: suy luận có thể mất vài trăm mili giây thay vì vài chục. Đó là chi phí của giả lập, không
phải model có vấn đề.

**3. Cấu hình cho CPU.** Sửa `.env`:

```bash
DEFAULT_MODEL=phobert-ner-cpu
FALLBACK_MODEL=phobert-ner-local
```

Và xoá khối `deploy` của service `triton` trong `docker-compose.yml` (mục 6 của SETUP_LINUX.md).

**4. Lấy model.** Máy Mac không huấn luyện PhoBERT nhanh được, nên dùng Google Colab:
xem [TRAINING.md](TRAINING.md). Tải file `.zip` về, giải nén đè vào repo.

Muốn thử huấn luyện tại chỗ bằng MPS thì được, nhưng chậm và không phải mục tiêu của bài này.

**5. Dựng hệ thống.**

```bash
docker compose up -d
python3 tests/test_pipeline.py
```

Test sẽ tự **bỏ qua** phần `phobert-ner-gpu` vì model đó không sẵn sàng. Đó là kết quả đúng,
không phải lỗi.

### Cách nhanh hơn trên Mac: bỏ Triton đi

Chỉ muốn thử API và giao diện thì dùng `phobert-ner-local`: nó chạy hẳn trên kiến trúc ARM, nhanh
hơn nhiều so với Triton qua giả lập.

```bash
docker compose up -d api
# Triton không sẵn sàng, API tự chuyển sang model dự phòng phobert-ner-local.
curl -X POST localhost:8080/v1/ner \
     -H 'Content-Type: application/json' \
     -d '{"text":"Nguyễn Bá Thiêm làm việc tại Google ở Hà Nội.","model":"phobert-ner-local"}'
```

Đây cũng là cách thấy tận mắt cơ chế dự phòng: `/health` báo `degraded` còn API vẫn trả lời.

---

## Bảng đối chiếu lệnh

| Việc | Linux / WSL / macOS |
|---|---|
| Xem cổng đang bị chiếm | `ss -tlnp \| grep 8080` (Linux) hoặc `lsof -i :8080` (macOS) |
| Xem log | `docker compose logs -f api` |
| Vào bên trong container | `docker compose exec api bash` |
| Xoá sạch để làm lại | `docker compose down -v && docker system prune -a` |

Gặp lỗi thì tra [TROUBLESHOOTING.md](TROUBLESHOOTING.md) trước.
