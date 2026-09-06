# -*- coding: utf-8 -*-
"""Kịch bản đo tải bằng Locust.

Mục tiêu KHÔNG phải "xem hệ thống chịu được bao nhiêu" mà là tìm ĐIỂM BÃO HOÀ:
mức tải mà thêm người dùng nữa thì thông lượng đứng yên còn độ trễ tăng vọt.
Biết điểm đó mới ước lượng được cần bao nhiêu máy cho lượng truy cập dự kiến.

Chạy:
    # có giao diện web, xem đồ thị theo thời gian thực tại http://localhost:8089
    locust -f load_test/locustfile.py --host http://localhost:8080

    # chạy tự động, không giao diện
    locust -f load_test/locustfile.py --host http://localhost:8080 \
           --headless -u 80 -r 20 -t 30s

Chọn model cần đo bằng biến môi trường (mặc định là model mặc định của server):
    NER_MODEL=phobert-ner-cpu locust -f load_test/locustfile.py ...

Ba điều dễ làm sai khi đo tải, đã tránh sẵn ở đây:

1. **Gửi mãi một câu.** Kết quả sẽ đẹp giả tạo vì mọi tầng cache đều trúng, và
   mọi request có cùng độ dài nên không thấy ảnh hưởng của độ dài câu. Ở đây
   xoay vòng nhiều câu dài ngắn khác nhau.
2. **Không có thời gian nghỉ giữa các request.** Người dùng thật đọc kết quả rồi
   mới gõ tiếp. `between(0.1, 0.5)` mô phỏng điều đó; bỏ đi thì đang đo "một
   vòng lặp gọi API" chứ không phải đo tải người dùng.
3. **Chỉ nhìn số trung bình.** Trung bình che mất phần đuôi. Luôn đọc p95/p99 --
   đó mới là thứ người dùng chậm nhất cảm nhận.
"""
import os
import random

from locust import HttpUser, between, task

MODEL = os.environ.get("NER_MODEL", "")        # rỗng = để server tự chọn mặc định

# Câu dài ngắn khác nhau: độ dài đầu vào ảnh hưởng trực tiếp tới thời gian tính
# toán, nên trộn lẫn mới ra bức tranh giống thật.
TEXTS = [
    "Nguyễn Bá Thiêm làm việc tại Google ở Hà Nội.",
    "Việt Nam và Hoa Kỳ ký thoả thuận hợp tác về công nghệ.",
    "Phạm Nhật Vượng sáng lập VinGroup tại Hải Phòng.",
    "CEO Tim Cook của Apple đến thăm Thành phố Hồ Chí Minh.",
    "Ngân hàng Nhà nước Việt Nam giữ nguyên lãi suất điều hành trong quý này.",
    "Câu lạc bộ Hoàng Anh Gia Lai thắng Hà Nội FC trên sân Pleiku chiều qua.",
    ("Nạn nhân cuối trong vụ thảm sát ở Đồng Nai đã qua cơn nguy kịch. Sau ca phẫu thuật, "
     "anh Nguyễn Viết Quang, 40 tuổi, đã tỉnh lại. Vợ anh là chị Hồ Thị Mai, 36 tuổi, "
     "cùng hai con đã tử vong trong vụ việc."),
    ("Công ty Cổ phần FPT công bố kết quả kinh doanh quý III. Đại diện FPT cho biết doanh thu "
     "từ thị trường Nhật Bản tăng mạnh, trong khi mảng viễn thông tại Thành phố Hồ Chí Minh "
     "giữ nguyên so với cùng kỳ."),
]


class NERUser(HttpUser):
    """Một người dùng ảo: gửi câu, chờ kết quả, nghỉ một nhịp, gửi tiếp."""

    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        body = {"text": random.choice(TEXTS)}
        if MODEL:
            body["model"] = MODEL

        # `name` gộp mọi request vào một dòng thống kê thay vì tách theo nội dung.
        with self.client.post("/v1/ner", json=body,
                              name=f"POST /v1/ner [{MODEL or 'mặc định'}]",
                              catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            # Trả 200 nhưng rỗng cũng là hỏng. Không kiểm thì một model nạp lỗi
            # vẫn cho ra biểu đồ đẹp long lanh.
            if "entities" not in response.json():
                response.failure("Response thiếu trường 'entities'")
