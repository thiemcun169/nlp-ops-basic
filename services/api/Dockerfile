# Image của lớp ứng dụng (FastAPI). Cố tình KHÔNG chứa trọng số model:
# model được gắn vào lúc chạy qua volume. Nhờ vậy huấn luyện lại model không
# phải build lại image, và image giữ nguyên kích thước vài trăm MB thay vì
# phình theo từng phiên bản model.
FROM python:3.11-slim

WORKDIR /app

# Cài từ wheel đã tải sẵn trên máy host. Lý do rất thực tế: mạng bên trong
# Docker build ở nhiều máy bị bóp băng thông nặng (đo được ~20-30 kB/s trong
# container so với >100 MB/s trên host) khiến `pip install` gần như treo.
# Xem docs/TROUBLESHOOTING.md muc E02.
# Không có thư mục wheels thì tự động quay về cài từ mạng như bình thường,
# nên máy nào mạng tốt vẫn build được mà không cần bước chuẩn bị nào.
COPY requirements.txt .
COPY wheels ./wheels
# Điều kiện phải hỏi "có file .whl nào không", KHÔNG phải "thư mục có rỗng không".
# Thư mục wheels/ luôn tồn tại trong Git nhờ một file .gitkeep (nếu bỏ nó thì
# `COPY wheels` hỏng ngay ở clone mới), nên `ls -A` LUÔN khác rỗng và nhánh
# offline sẽ chạy với thư mục không có gói nào.
RUN if ls ./wheels/*.whl >/dev/null 2>&1; then \
        echo "Cài từ wheel tải sẵn (offline)"; \
        pip install --no-cache-dir --no-index --find-links=./wheels -r requirements.txt; \
    else \
        echo "Không có wheel tải sẵn, cài từ mạng"; \
        pip install --no-cache-dir -r requirements.txt; \
    fi && rm -rf ./wheels

COPY app ./app
COPY static ./static
COPY tokenizer ./tokenizer
COPY labels.json .

EXPOSE 8080

# Kiểm tra sức khoẻ ngay trong image: orchestrator nào cũng dùng được, không
# phải cấu hình lại ở từng nơi triển khai.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
