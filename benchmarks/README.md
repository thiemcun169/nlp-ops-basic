# Chấm điểm chất lượng

Hai bộ dữ liệu, hai câu chuyện khác nhau. **Phải đọc cả hai** mới hiểu đúng model.

| Script | Dữ liệu | Trả lời câu gì |
|---|---|---|
| `eval_testset.py` | WikiAnn test, 10.000 câu | model làm tốt tới đâu trên **đúng loại dữ liệu nó được học** |
| `run_benchmark.py` | 26 câu viết tay, 8 nhóm tình huống | model làm tốt tới đâu trên **văn bản giống ngoài đời** |

```bash
python3 benchmarks/eval_testset.py                   # cần: pip install datasets
python3 benchmarks/run_benchmark.py                  # chỉ dùng thư viện chuẩn
```

---

## Bộ 26 câu: ai gán nhãn?

**Bộ này được sinh ra trong lúc dựng repo, bằng công cụ AI, và không qua kiểm
chứng chéo bởi người gán nhãn độc lập nào.**

Nói rõ vì điều này quyết định con số đáng tin tới đâu:

- Nó **không phải** bộ dữ liệu chuẩn, không có độ đồng thuận giữa nhiều người gán nhãn.
- Nó **chỉ** dùng để soi các tình huống cụ thể đã chọn có chủ ý: dấu câu dính vào
  từ, chữ viết thường, tên nước ngoài, tên tổ chức dài, bẫy danh từ chung.
- Đừng trích con số F1 ở đây như thể đó là năng lực thật của model.

Sửa hoặc thêm câu thì sửa `build_gold.py` rồi chạy lại, đừng sửa tay
`gold_vi.json`: offset ký tự do máy tính ra và tự kiểm lại.

---

## Vì sao WikiAnn cho kết quả ngược hẳn

Đây là phát hiện đáng giá nhất khi chạy cả hai bộ. Trên cùng một hệ thống:

| | WikiAnn (150 câu, seed 0) | Bộ viết tay (26 câu) |
|---|---|---|
| `phobert-ner-gpu` | **0,808** | 0,774 |
| `groq/compound` | **0,436** | **0,962** |

Hai model chấm trên **đúng cùng một tập câu** ở mỗi cột, nên chênh lệch là chênh
lệch thật. Chấm model tự host trên toàn bộ 10.000 câu WikiAnn cho F1 **0,879**;
không chấm LLM trên cả 10.000 câu vì tốn tiền và chạm hạn mức tần suất.

LLM thắng đậm ở bộ này và thua đậm ở bộ kia. Lý do không nằm ở model.

**WikiAnn được xây dựng TỰ ĐỘNG** từ liên kết trong Wikipedia, không phải người
gán nhãn thủ công. Nhãn của nó mang theo quy ước riêng và cả nhiễu:

| Câu | WikiAnn gán | Thực tế |
|---|---|---|
| `Giao thức TCP / IP` | PER | không phải tên người |
| `Ưng ngỗng nâu` | LOC | tên một loài chim |
| `Vùng đô thị` | ORG | danh từ chung |
| `Thiên Hà , Quảng Châu` | một span LOC | hai địa danh |

Nhìn theo nhãn còn rõ hơn: LLM đạt ORG 0,200 và LOC 0,182 trên WikiAnn, trong khi
trên văn bản thật nó đạt ORG 0,909 và LOC 0,976. Cùng một model, cùng một prompt.

Model tự train **học đúng những quy ước đó** vì nó được huấn luyện trên chính bộ
này, nên chấm trên tập test cùng nguồn thì điểm cao. LLM áp dụng ngữ nghĩa đời
thực nên bị chấm là sai, dù câu trả lời của nó hợp lý hơn.

**Kết luận rút ra, dùng được cho mọi dự án:** điểm trên tập test đo *mức khớp với
quy ước gán nhãn của bộ dữ liệu*, không đo *độ đúng*. Hai thứ đó chỉ trùng nhau
khi bộ dữ liệu được gán nhãn cẩn thận bởi người. Trước khi tin một con số F1, hãy
mở vài chục câu của tập test ra đọc.

---

## Model có bền không?

Không. Bộ viết tay chỉ ra bốn điểm yếu cụ thể:

| Nhóm | Tự host | LLM | Vì sao |
|---|---|---|---|
| chữ viết thường | **0,000** | 1,000 | WikiAnn viết hoa chuẩn, model học chữ hoa là tín hiệu chính |
| bẫy danh từ chung | 0,667 | 1,000 | `công ty công nghệ`, `trung tâm` bị nhận nhầm là tên riêng |
| tên tổ chức dài | 0,800 | 0,800 | `Công an tỉnh Bình Dương` bị xé thành `Công an` [ORG] + `tỉnh Bình Dương` [LOC] |
| đoạn dài | 0,769 | 1,000 | tên nhắc lại lần thứ hai hay bị bỏ sót |

F1 bằng 0,000 ở nhóm viết thường là hỏng hoàn toàn, không phải yếu. Dữ liệu huấn
luyện quá "sạch" so với tin nhắn và bình luận ngoài đời.

Đáng chú ý: **tổ chức là nhóm duy nhất LLM cũng chỉ đạt 0,800**, trong khi bảy
nhóm còn lại nó đạt từ 0,857 tới 1,000. Đó là điểm khó của chính bài toán, không
phải của riêng model nào. Tính theo nhãn thì ORG cũng là loại yếu nhất của model
tự host (F1 0,703 so với PER 0,889 và LOC 0,762).

Ba hướng xử lý, theo thứ tự công sức tăng dần: chuẩn hoá hoa thường ở tiền xử lý;
bổ sung bản viết thường của chính các câu train vào tập huấn luyện; hoặc chuyển
sang model LLM cho nhóm văn bản này.

---

## Phép đo "tách từ lệch"

`eval_testset.py` in ra tỷ lệ câu mà cách tách từ lúc phục vụ khác cách tách của
bộ dữ liệu. Con số này giải thích phần lớn khoảng cách giữa F1 ở đây và F1 mà
notebook báo: notebook chấm trên chính token của WikiAnn, còn API nhận văn bản
thô và tự tách lại.

Không đo con số đó thì rất dễ đổ oan cho model một khoản mà thật ra là chi phí
của việc phục vụ văn bản thật.
