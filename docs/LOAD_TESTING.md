# Đo tải

Đo **chất lượng** thì xem [`benchmarks/README.md`](../benchmarks/README.md). Trang
này chỉ nói về sức chịu tải.

Chất lượng cao mà không chịu nổi tải thì không dùng được. Chịu tải tốt mà trả sai
thì càng tệ. Phải đo cả hai.

```bash
pip install locust

bash load_test/run_stress_test.sh                          # 80 user, 30 giây
USERS=150 RUN_TIME=60s bash load_test/run_stress_test.sh
MODELS="phobert-ner-gpu" USERS=200 bash load_test/run_stress_test.sh
```

Xem đồ thị thời gian thực: `locust -f load_test/locustfile.py --host http://localhost:8080`
rồi mở http://localhost:8089.

---

## Mục tiêu: tìm điểm bão hoà

Không phải "xem hệ thống chịu được bao nhiêu", mà là tìm mức tải mà **thêm người
dùng nữa thì throughput đứng yên còn latency tăng vọt**. Biết điểm đó mới ước
lượng được cần bao nhiêu máy.

| Dấu hiệu                                             | Nghĩa là                                                                    |
| ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| Throughput tăng, latency gần như không đổi       | còn dư sức                                                                 |
| Throughput**nằm ngang**, latency p50 tăng dần | **đã bão hoà**                                                      |
| p99 gấp hơn 3 lần p50                               | phần đuôi giãn rộng, thường là đã qua điểm bão hoà              |
| Bắt đầu có request lỗi                            | vượt xa điểm bão hoà, hoặc có lỗi đồng thời (TROUBLESHOOTING E03) |

`summarize_results.py` tự chỉ ra mấy dấu hiệu này.

**Vì sao throughput chạm trần:** mỗi giây GPU chỉ chạy được một số lượt suy luận
nhất định. Request vượt quá con số đó **không biến mất, mà xếp hàng**. Hàng đợi
dài thêm thì mỗi người chờ lâu hơn, còn số người được phục vụ mỗi giây vẫn y nguyên.

Đây cũng là lý do latency và throughput không tối ưu đồng thời được: batch to thì
throughput cao nhưng mỗi người phải chờ đủ batch; batch bằng 1 thì latency thấp
nhất nhưng GPU rảnh phần lớn thời gian.

Locust báo latency theo **phân vị đo phía client** (p50, p90, p99), đã bao gồm cả
đường mạng và thời gian FastAPI xử lý. Dashboard Grafana thì chỉ có **trung bình**
đo phía Triton. Hai nguồn này không so trực tiếp được với nhau.

---

## Ba điều dễ làm sai

Đã tránh sẵn trong `locustfile.py`, nhưng bạn sẽ gặp lại ở dự án khác:

1. **Gửi mãi một câu.** Kết quả đẹp giả tạo vì cache đều trúng, và mọi request
   cùng độ dài nên không thấy ảnh hưởng của độ dài đầu vào. Ở đây xoay vòng tám câu.
2. **Không nghỉ giữa các request.** Người dùng thật đọc kết quả rồi mới gõ tiếp.
   Bỏ thời gian nghỉ là đang đo một vòng lặp gọi API, không phải đo tải người dùng.
3. **Chỉ nhìn số trung bình.** Trung bình che mất phần đuôi. Đọc p95 và p99: đó
   mới là thứ người dùng chậm nhất cảm nhận, và họ là người bỏ đi trước.

---

## Đừng đo tải model LLM

Dịch vụ LLM tính tiền theo lượt gọi và có giới hạn tần suất. Bắn 80 người dùng ảo
vào đó chỉ tạo ra một bảng toàn lỗi 429 và một hoá đơn.

Muốn biết LLM chậm bao nhiêu thì đọc cột p50 trong bảng chấm điểm chất lượng.

---

## Ghép với bảng giám sát

Cách dùng đáng giá nhất là chạy đo tải **trong khi mở Grafana**:

```bash
# cửa sổ 1: mở http://localhost:3001
# cửa sổ 2:
USERS=150 SPAWN_RATE=30 RUN_TIME=60s MODELS="phobert-ner-gpu" bash load_test/run_stress_test.sh
```

Con số trong bảng và hình dạng đường cong trên dashboard nói cùng một câu chuyện
từ hai phía. Chỉ nhìn một thì dễ kết luận sai. Xem [MONITORING.md](MONITORING.md).
