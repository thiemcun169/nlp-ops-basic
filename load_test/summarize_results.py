# -*- coding: utf-8 -*-
"""Đọc các file CSV Locust vừa sinh ra và in một bảng so sánh duy nhất.

Có script này để số liệu đi thẳng từ phép đo vào báo cáo, không qua bước chép
tay -- chép tay là chỗ số liệu bị sai hoặc bị làm đẹp lúc nào không hay.

    python3 load_test/summarize_results.py phobert-ner-gpu phobert-ner-cpu
"""
import csv
import os
import sys


def read_aggregated(prefix: str):
    path = f"results_{prefix}_stats.csv"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return next((r for r in rows if r["Name"] == "Aggregated"), None)


def main() -> None:
    models = sys.argv[1:] or ["phobert-ner-gpu", "phobert-ner-cpu"]
    rows = [(m, read_aggregated(m)) for m in models]

    print("=" * 88)
    print("KẾT QUẢ ĐO TẢI")
    print("=" * 88)
    print(f"{'model':18s} {'request':>8s} {'lỗi':>6s} {'req/s':>8s} "
          f"{'p50':>8s} {'p90':>8s} {'p99':>8s}")
    print("-" * 88)

    measured = []
    for name, row in rows:
        if row is None:
            print(f"{name:18s} {'(chưa chạy)':>8s}")
            continue
        rps = float(row["Requests/s"])
        print(f"{name:18s} {row['Request Count']:>8s} {row['Failure Count']:>6s} "
              f"{rps:8.1f} {row['50%']:>7s}ms {row['90%']:>7s}ms {row['99%']:>7s}ms")
        measured.append((name, rps, float(row["50%"]), float(row["99%"]),
                         int(row["Failure Count"])))

    if len(measured) >= 2:
        fastest = max(measured, key=lambda x: x[1])
        slowest = min(measured, key=lambda x: x[1])
        if slowest[1] > 0:
            print(f"\nThông lượng: {fastest[0]} cao hơn {slowest[0]} "
                  f"{fastest[1] / slowest[1]:.1f} lần.")
        if slowest[2] > 0:
            print(f"Độ trễ p50 : {slowest[0]} chậm hơn {fastest[0]} "
                  f"{slowest[2] / fastest[2]:.1f} lần.")

    for name, _rps, p50, p99, fails in measured:
        if p50 > 0 and p99 / p50 > 3:
            print(f"\n{name}: p99 gấp {p99 / p50:.1f} lần p50. Phần đuôi giãn rộng như vậy "
                  f"thường là dấu hiệu ĐÃ BÃO HOÀ -- request phải xếp hàng chờ.")
        if fails > 0:
            print(f"{name}: có {fails} request lỗi. Xem log của API trước khi tin phần còn lại "
                  f"của bảng này.")

    print("\nCách đọc: thông lượng ngừng tăng trong khi độ trễ tiếp tục tăng chính là điểm")
    print("bão hoà. Thêm người dùng sau điểm đó chỉ làm dài thêm hàng đợi, không phục vụ")
    print("được nhiều hơn. Muốn nhìn thấy trực tiếp thì mở Grafana trong lúc đo.")


if __name__ == "__main__":
    main()
