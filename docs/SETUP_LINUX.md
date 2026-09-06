# Cài đặt trên Linux

Tài liệu **chính**. Hệ thống được phát triển và kiểm chứng trên Ubuntu 24.04,
Docker 29.2.1, Docker Compose v5.1.0, driver NVIDIA 595, NVIDIA Container Toolkit
1.19, Python 3.10.

Yêu cầu tối thiểu thực sự: **Docker Compose v2 trở lên** (lệnh `docker compose`
viết rời, không phải `docker-compose` gạch nối) và driver NVIDIA đủ mới cho CUDA 12.

| Tài nguyên | Cần                                                                        |
| ------------ | --------------------------------------------------------------------------- |
| Đĩa trống | ~25 GB (riêng image Triton đã 19 GB)                                     |
| RAM          | 8 GB, 16 GB nếu chạy cả notebook lẫn hệ thống cùng lúc              |
| GPU          | không bắt buộc (xem mục 5). Có thì**4 tới 8 GB VRAM là đủ** |

Máy khác: [SETUP_WINDOWS_MACOS.md](SETUP_WINDOWS_MACOS.md).
Thuê GPU theo giờ: [SETUP_VASTAI.md](SETUP_VASTAI.md).

---

## 1. Docker và NVIDIA Container Toolkit

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER" && newgrp docker
docker compose version      # PHẢI ra kết quả
```

Driver NVIDIA cài trên máy **không đủ** để container thấy GPU, cần thêm toolkit:

```bash
nvidia-smi     # driver phải sẵn sàng trước đã

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Bước kiểm tra này bắt buộc phải qua trước khi đi tiếp:**

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Lỗi `could not select device driver "" with capabilities: [[gpu]]` nghĩa là toolkit
chưa cấu hình xong.

---

## 2. Image Triton

```bash
git clone <địa-chỉ-repo> nlp-ops-basic && cd nlp-ops-basic
bash scripts/build_triton_image.sh        # MỘT LẦN, 15 tới 30 phút
```

Script dựng image **chỉ chứa backend ONNX Runtime** (19,4 GB) thay vì image đầy đủ
có cả TensorFlow, PyTorch, TensorRT (27,4 GB).

Không muốn chờ thì dùng thẳng image chính thức, hệ thống chạy y hệt:

```bash
echo "TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.12-py3" >> .env
```

---

## 3. Huấn luyện model

Trọng số **không nằm trong Git** (nặng, sinh lại được). Hai cách lấy:

```bash
# Cách A: chạy tại chỗ
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets seqeval onnx onnxruntime accelerate jupyter matplotlib
jupyter notebook notebooks/01_train_ner.ipynb      # ~10 phút trên RTX 3090
```

Cách B, máy không có GPU: huấn luyện trên **Google Colab**, xem
[TRAINING.md](TRAINING.md). Notebook tự nhận biết Colab và đóng gói artifact thành
`.zip` cho bạn tải về.

Kiểm tra:

```bash
ls -la services/triton/model_repository/ner_onnx/1/model.onnx   # ~538 MB
cat services/triton/model_repository/ner_onnx/model_card.json   # F1, seed, dữ liệu
```

---

## 4. Dựng hệ thống

```bash
cp .env.example .env      # tuỳ chọn: điền GROQ_API_KEY
                          # key miễn phí: https://console.groq.com/keys
docker compose up -d
docker compose ps         # cả 4 dịch vụ phải Up
python3 tests/test_pipeline.py
```

|                  | Địa chỉ                 |
| ---------------- | -------------------------- |
| Trang demo       | http://localhost:8080      |
| Tài liệu API   | http://localhost:8080/docs |
| Bảng giám sát | http://localhost:3001      |
| Prometheus       | http://localhost:9490      |

Xác nhận model thực sự chạy trên GPU (đọc log, đừng tin cảm giác):

```bash
docker compose logs triton | grep ModelInstanceInitialize
# mong đợi:  ner_onnx_0_0 (GPU device 0)  và  ner_onnx_cpu_0_0 (CPU device 0)
```

Dừng: `docker compose down`

---

## 5. Máy không có GPU

Vẫn chạy đầy đủ, chỉ chậm hơn. Sửa hai chỗ:

1. Xoá khối `deploy` của service `triton` trong `docker-compose.yml`.
2. Đặt `DEFAULT_MODEL=phobert-ner-cpu` trong `.env`.

Model `ner_onnx` (bản GPU) không nạp được và Triton báo lỗi cho riêng nó, nhưng
`ner_onnx_cpu` vẫn chạy. Muốn log sạch hẳn thì xoá thư mục
`services/triton/model_repository/ner_onnx/`.

---

## 6. Cổng bị chiếm

```bash
ss -tlnp | grep -E '8080|8000|9490|3001'
```

Đổi trong `.env`, không cần sửa `docker-compose.yml`:

```bash
API_PORT=18080
GRAFANA_PORT=13001
PROMETHEUS_PORT=19490
```

Nhớ đổi cả khi chạy test: `API_URL=http://localhost:18080 python3 tests/test_pipeline.py`

---

## 7. Công cụ đo

```bash
pip install locust datasets

python3 benchmarks/run_benchmark.py       # chất lượng trên văn bản giống ngoài đời
python3 benchmarks/eval_testset.py        # chất lượng trên tập test WikiAnn 10.000 câu
bash load_test/run_stress_test.sh         # tìm điểm bão hoà
```

---

## 8. Khi mọi thứ có vẻ hỏng

Chạy lần lượt, dừng ở bước đầu tiên cho kết quả bất thường:

```bash
docker compose ps
docker compose logs triton --tail 50
docker compose logs api --tail 50
curl -s localhost:8080/health | python3 -m json.tool
python3 tests/test_units.py               # logic thuần, không cần Docker
nvidia-smi && df -h .
```

12 lỗi hay gặp kèm cách chẩn đoán: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
