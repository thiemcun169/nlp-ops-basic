# Huấn luyện model

Toàn bộ nằm trong `notebooks/01_train_ner.ipynb`. Notebook tự nhận biết môi trường
và chạy được ở ba nơi: Google Colab, máy cá nhân có GPU, máy chỉ có CPU.

| Mục | Nội dung |
|---|---|
| 1-2 | Nạp WikiAnn tiếng Việt, khảo sát dữ liệu |
| 3-5 | Tokenize và căn nhãn theo subword, fine-tune PhoBERT, đánh giá bằng seqeval |
| 6-7 | Phân tích lỗi, thử bốn nhóm tình huống lệch phân phối |
| 8 | **Bài học về lệch tiền xử lý**, phần đáng giá nhất |
| 9-10 | Xuất ONNX, kiểm chứng lại, ghi vào kho model |

Kết quả đo được (RTX 3090, seed 0):

```
              precision    recall  f1-score   support
         LOC      0.884     0.920     0.902      3717
         ORG      0.882     0.851     0.866      3704
         PER      0.910     0.933     0.921      3884
   micro avg      0.893     0.902     0.897     11305
```

ORG là loại yếu nhất, và điều này lặp lại ở cả hai bộ chấm điểm độc lập trong
`benchmarks/`, nên không phải ngẫu nhiên: tên tổ chức dài, nhiều biến thể, ranh
giới mơ hồ hơn tên người.

---

## Chạy trên Google Colab

Phù hợp nhất khi máy không có GPU. Colab cho dùng T4 miễn phí, khoảng 15 phút.

1. Tải `notebooks/01_train_ner.ipynb` lên https://colab.research.google.com
2. Runtime > Change runtime type > **T4 GPU**
3. Runtime > Run all
4. Ô cuối tự đóng gói `nlp-ops-artifacts.zip` và mở hộp thoại tải về
5. Giải nén đè lên repo trên máy bạn:

```bash
unzip -o ~/Downloads/nlp-ops-artifacts.zip -d /duong/dan/toi/nlp-ops-basic
ls -la nlp-ops-basic/services/triton/model_repository/ner_onnx/1/model.onnx
```

Notebook tự phân biệt hai chế độ nên **không cần sửa gì**: tìm thấy
`docker-compose.yml` ở thư mục cha thì ghi thẳng vào `services/`, không thì gom vào
thư mục riêng rồi nén lại.

**Không dựng được Docker trên Colab.** Colab chỉ dùng để huấn luyện; phần phục vụ
phải chạy trên máy có Docker.

---

## Chạy trên máy cá nhân

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets seqeval onnx onnxruntime accelerate jupyter matplotlib
jupyter notebook notebooks/01_train_ner.ipynb
```

Hoặc chạy một lượt không cần giao diện:

```bash
jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3000 notebooks/01_train_ner.ipynb
```

Máy chỉ có CPU vẫn chạy nhưng mất hơn một tiếng. Muốn thử nhanh thì giảm
`num_train_epochs` xuống 1.

---

## Mục 8: phần đáng giá nhất

Tái hiện một lỗi thật đã xảy ra trong dự án này, và là loại tệ nhất: **không
exception, không log đỏ, HTTP vẫn 200, chỉ có kết quả sai.**

WikiAnn tách dấu câu thành token riêng. `text.split()` lúc phục vụ lại để dấu câu
dính vào từ, nên token `Quang,` chưa từng xuất hiện lúc huấn luyện. Notebook chạy
cả hai cách trên cùng một câu:

```
A. text.split()           35 token ->  7 từ được gán thực thể
B. tách riêng dấu câu     43 token -> 11 từ được gán thực thể
```

Bốn từ bị mất đều nằm cạnh dấu câu. Model bình thường; khâu tiền xử lý sai.

Ba điều rút ra:

1. **Dùng chung một hàm tiền xử lý cho cả huấn luyện lẫn phục vụ.** Trong repo này
   là `services/api/app/core/tokenization.py`; notebook chép đúng quy ước đó chứ
   không tự viết lại.
2. **Chỉ test hồi quy mới bắt được lỗi loại này.** `tests/test_pipeline.py` có một
   trường hợp canh riêng, để lỗi đã sửa không lặng lẽ quay lại.
3. **Trước khi kết luận model kém, kiểm tra đầu vào lúc phục vụ có giống lúc huấn
   luyện không.**

Cùng ý đó còn đo được ở phía ngược lại: `benchmarks/eval_testset.py` in ra tỷ lệ
câu mà cách tách từ lúc phục vụ khác cách tách của WikiAnn (hiện là 15,8%), và đó
là phần lớn khoảng cách giữa F1 0,879 đo qua API với F1 0,897 notebook báo.

---

## Kho model

Trọng số xuất **thẳng** vào kho model, không qua file tạm:

```
services/triton/model_repository/ner_onnx/
├── config.pbtxt          cách Triton nạp và chạy
├── labels.json           bảng nhãn
├── model_card.json       dữ liệu gì, seed nào, F1 bao nhiêu
└── 1/model.onnx          trọng số, phiên bản 1
```

Thư mục phiên bản chính là cách đánh phiên bản: huấn luyện lại mà muốn giữ bản cũ
thì tạo `2/`, Triton nạp được cả hai cùng lúc.

`model_card.json` hay bị bỏ quên nhất. Trọng số không tự nói được nó đến từ đâu;
không có file này thì ba tháng sau không ai biết model đang chạy đạt bao nhiêu.

---

## Muốn cải thiện model thì làm gì

`benchmarks/` chỉ ra ba điểm yếu cụ thể:

| Điểm yếu | Bằng chứng | Hướng xử lý |
|---|---|---|
| Chữ viết thường | F1 nhóm "viết thường" = **0,000** | bổ sung bản viết thường của chính câu train vào tập huấn luyện, hoặc chuẩn hoá hoa/thường ở tiền xử lý |
| Tên tổ chức dài | ORG F1 thấp nhất ở cả hai bộ | thêm dữ liệu có tên cơ quan tiếng Việt |
| Ranh giới thực thể | `Quảng trường Ba Đình` bị cắt đôi | thêm tầng CRF, hoặc hậu xử lý ràng buộc chuỗi BIO hợp lệ |

Sau mỗi thay đổi, chạy lại **cùng một bộ chấm điểm**. Đổi bộ đánh giá giữa chừng
là mất khả năng so sánh, và đó là cách phổ biến nhất để tự lừa mình.

---

## Vì sao chọn cách làm này

**PhoBERT** được huấn luyện trước trên 20 GB văn bản tiếng Việt, hiểu chính tả và
cách ghép âm tốt hơn hẳn model đa ngôn ngữ cùng kích thước.

**WikiAnn** vì tải được tự do, đủ nhỏ để huấn luyện trong một buổi, có sẵn ba loại
thực thể. Nhược điểm của nó cũng là bài học: nó được xây dựng **tự động** từ liên
kết Wikipedia nên nhãn có nhiễu, và câu rất ngắn, đậm đặc thực thể, viết hoa chuẩn.
Xem [benchmarks/README.md](../benchmarks/README.md).

**ONNX** để môi trường phục vụ không cần PyTorch: image nhẹ hơn nhiều, không phụ
thuộc phiên bản PyTorch, và cùng một file chạy được trên nhiều runtime.

**So logits ONNX với PyTorch** vì xuất mô hình có thể sai âm thầm: một phép toán bị
xấp xỉ, một trục bị hoán vị. Model vẫn chạy, chỉ trả kết quả khác. Phép so ở mục 9
dừng ngay tại đó nếu sai số vượt ngưỡng.
