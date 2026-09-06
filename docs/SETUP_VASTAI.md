# Thuê GPU theo giờ

Dành cho người không có GPU. Chi phí thực tế cho một buổi thực hành thường dưới
một đô la.

Ví dụ dùng Vast.ai. RunPod, Lambda Labs hay một máy chủ ảo có GPU bất kỳ đều làm
giống nhau: có SSH và có Docker là chạy được.

---

## 1. Chọn máy

| Tiêu chí | Chọn | Vì sao |
|---|---|---|
| VRAM | **4 tới 8 GB là đủ** (RTX 3060, 2080, T4, A2000) | model chỉ chiếm ~1,5 GB VRAM. Thuê 24 GB là trả tiền cho phần không dùng |
| Disk | **tối thiểu 40 GB** | riêng image Triton đã 19 tới 27 GB |
| CUDA | 12.4 trở lên | khớp với Triton 24.12 |
| Reliability | trên 99% | máy hay rớt thì mất công dựng lại |
| Inet Down | trên 100 Mb/s | phải tải khoảng 20 GB image |

> **Đĩa là chỗ hay bị hụt nhất.** Mặc định Vast.ai chỉ cấp 10 GB, và sau khi thuê
> thì không tăng được nữa. Kéo Disk Space lên ít nhất 40 GB **trước khi** bấm RENT.

Muốn huấn luyện luôn trên máy thuê thì cần khoảng 8 GB VRAM. Chỉ chạy phục vụ và
đo tải thì 4 GB là thoải mái.

---

## 2. Kết nối

Lấy lệnh SSH ở tab Instances, thêm chuyển tiếp cổng:

```bash
ssh -p <PORT> root@<HOST> -L 8080:localhost:8080 \
                          -L 3001:localhost:3001 \
                          -L 9490:localhost:9490
```

Ba tuỳ chọn `-L` làm cho `localhost:8080` trên máy bạn trỏ vào máy thuê, nên mở
trình duyệt là thấy giao diện mà không phải mở cổng ra Internet.

> Đừng bỏ `-L` rồi mở cổng công khai cho tiện. Hệ thống này **chưa có xác thực**,
> mở ra Internet là ai cũng gọi được và bạn trả tiền GPU cho họ.

---

## 3. Dựng hệ thống

```bash
# Kiểm tra môi trường (máy Vast.ai thường đã có sẵn)
nvidia-smi && docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Thiếu Docker thì:  curl -fsSL https://get.docker.com | sh
# Thiếu NVIDIA Container Toolkit thì xem mục 2 của SETUP_LINUX.md

git clone <địa-chỉ-repo> nlp-ops-basic && cd nlp-ops-basic

# Máy thuê tính tiền theo giờ: dùng thẳng image chính thức thay vì bỏ 20 phút
# dựng bản nhẹ. 8 GB đĩa dư ra rẻ hơn 20 phút thuê GPU.
cp .env.example .env
echo "TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.12-py3" >> .env

# Huấn luyện ngay trên máy này, khoảng 10 phút
pip install torch transformers datasets seqeval onnx onnxruntime accelerate jupyter matplotlib
jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3000 notebooks/01_train_ner.ipynb

docker compose up -d
python3 tests/test_pipeline.py
```

Mở http://localhost:8080 và http://localhost:3001 trên trình duyệt máy bạn.

---

## 4. Đo và lấy kết quả về

```bash
python3 benchmarks/run_benchmark.py --out benchmarks/results.json
pip install locust && bash load_test/run_stress_test.sh
```

Chạy trên **máy của bạn**, không phải máy thuê:

```bash
scp -P <PORT> root@<HOST>:~/nlp-ops-basic/benchmarks/results.json .
```

---

## 5. Giữ tiến trình khi mất kết nối

SSH đứt là tiến trình chết theo. Huấn luyện 10 phút mà rớt ở phút thứ 9 là mất trắng.

```bash
tmux new -s train        # Ctrl+B rồi D để thoát, tiến trình vẫn chạy
tmux attach -t train     # quay lại xem
```

---

## 6. Tiết kiệm tiền

**Bấm DESTROY khi xong**, không phải Stop.

| Trạng thái | Còn tính tiền |
|---|---|
| Running | đầy đủ |
| **Stopped** | **vẫn tính tiền đĩa**, vài xu mỗi giờ |
| Destroyed | không |

Nhiều người tưởng Stop là hết chi phí. Không phải. Muốn giữ dữ liệu thì `scp` về
trước rồi Destroy hẳn.

Hai cách giảm chi phí khác:

- Huấn luyện trên **Google Colab miễn phí**, chỉ thuê GPU cho phần phục vụ và đo
  tải. Xem [TRAINING.md](TRAINING.md).
- Chọn máy có Inet Down cao: tải 20 GB image ở 20 Mb/s mất hai tiếng, ở 500 Mb/s
  mất năm phút. Tiền thuê trong lúc chờ tải thường lớn hơn chênh lệch giá máy.

---

## 7. Lỗi hay gặp riêng ở máy thuê

| Triệu chứng | Xử lý |
|---|---|
| `no space left on device` lúc kéo image | Đĩa quá nhỏ, không tăng được sau khi thuê. Destroy và thuê máy khác từ 40 GB |
| `could not select device driver` | Thiếu NVIDIA Container Toolkit, xem SETUP_LINUX.md mục 2 |
| Mở `localhost:8080` không thấy gì | Quên `-L` khi SSH. Thoát ra và SSH lại |
| Huấn luyện dừng giữa chừng, không báo lỗi | SSH đứt. Dùng `tmux` như mục 5 |

Lỗi chung của hệ thống: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
