# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603749
- **Họ và tên:** Triệu Quốc Khánh
- **SEED** (in ở ô đầu notebook): 4060

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

Bộ kiểm tra tự động: 15/15 đạt. Tính tay: 5/5 khớp (baseline_accuracy = 0.87, sigmoid_0 = 0.50, f1_example = 0.73, cosine_example = 0.60, llm_cost_per_day = 10.50).

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (F1 test là toàn bộ 775 tin test; F1 SUB là 40 tin cân bằng; p95 là độ trễ khi gọi từng tin):

| Phương án | F1 (test) | F1 (SUB 40) | p95 (ms/tin) | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.913 | 0.947 | 0.407 | 0.0 |
| 2. BoW + MLP (deep learning) | 0.968 | 1.000 | 2.370 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.910 | 1.000 | 12.644 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.963 | 1.000 | 8.529 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.942 | 1.000 | 20.998 | 0.0 |
| 4a. LLM nhỏ zero-shot (gpt-oss-20b) | 0.952 | 0.952 | 2565.599 | 0.0 |
| 4b. LLM lớn zero-shot (gpt-oss-120b) | 0.952 | 0.952 | 2422.130 | 0.0 |
| 4c. LLM nhỏ few-shot | 0.930 | 0.930 | 3588.559 | 0.0 |

Ghi chú:
- kNN (3a): precision 0.850, recall 0.978. Lần chạy đầu F1 = 0.000 vì `return y_pred` nằm trong vòng `for` của `knn_predict`; sau khi đưa `return` ra ngoài vòng lặp và chạy lại thì F1 = 0.910.
- Trên tập SUB (chỉ 40 tin, 1 tin sai làm F1 đổi vài điểm), các hàng 2, 3a, 3b, 3c cùng F1 = 1.000 nên chênh lệch dưới 0.05 coi như hòa; chỉ khi test bằng case khó (mục 4) mới thấy khác biệt.
- USD / 1k tin của hàng 4a–4c bằng 0.0 vì model Groq không có trong bảng giá `PRICE_PER_1M` của notebook, không phải miễn phí.
- LLM chạy qua Groq (gpt-oss) vì hạn mức gói miễn phí của Gemini hết (RateLimitError 429).
- Biểu đồ đánh đổi (1 − F1 theo p95, SLA 50 ms): đường Pareto gồm BoW + LogReg (nhanh nhất, 0.407 ms) và BoW + MLP (lỗi thấp nhất, 0%). Các hàng còn lại bị lấn át; ví dụ kNN cũng lỗi 0% nhưng chậm hơn BoW + MLP (12.6 ms so với 2.4 ms). Hai LLM vượt SLA gần 50 lần.

## 3. Dự đoán và thực tế

| Dự đoán | Mình đoán | Kết quả thực tế | Vì sao lệch |
| --- | --- | --- | --- |
| `llm_models` (Thí nghiệm 4.A) | Model lớn F1 cao hơn model nhỏ khoảng 0.05; model lớn đắt gấp 7 lần | Hai model cùng F1 = 0.952; model lớn không chậm hơn (p95 2422 ms so với 2566 ms) | Bài toán dễ nên model lớn không tự động tốt hơn; SUB chỉ có 40 tin nên khó phân biệt. Chi phí không đo được vì giá Groq chưa có trong bảng giá |
| `dl_relu` (Thí nghiệm 2.B) | Bỏ ReLU thì mạng chỉ còn tuyến tính, F1 giảm về gần LogReg | F1 không ReLU = 0.968, bằng MLP có ReLU (0.968) và cao hơn LogReg (0.913) | Hai lớp Linear liên tiếp vẫn là mô hình tuyến tính, nên phi tuyến không phải lý do MLP hơn LogReg trên BoW. Khác biệt có thể đến từ cách huấn luyện (Adam, mini-batch, 30 epoch) so với gradient descent 300 epoch ở LogReg tự code |
| `hard_eval` (56 case khó) | BoW và Encoder + LogReg còn khoảng 60–75% accuracy, LLM khoảng 85–95% | BoW + LogReg 48%, Encoder + LogReg 66%, LLM zero-shot 84%, LLM + prompt phòng thủ 93% | BoW tệ hơn dự đoán: nó chỉ đếm từ rời rạc nên bỏ lọt spam không có từ khoá. Encoder + LogReg đúng khoảng giữa dự đoán; LLM đúng như dự đoán |

## 4. Phân tích case khó

Bộ test gồm 50 case của lab và 6 case mình viết thêm (id 101–106), tổng 56 case. Tỉ lệ đoán đúng theo nhóm (%):

| Nhóm case | BoW+LogReg | BoW+MLP | Enc+kNN | Enc+LogReg | Enc+MLP | LLM nhỏ zero-shot | LLM nhỏ + prompt phòng thủ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tưởng spam nhưng không phải | 92 | 92 | 67 | 75 | 83 | 58 | 75 |
| Thao túng / lừa đảo | 12 | 50 | 88 | 50 | 50 | 100 | 100 |
| Spam giọng tự nhiên | 8 | 8 | 42 | 42 | 42 | 100 | 100 |
| Né từ khoá | 17 | 17 | 83 | 83 | 83 | 100 | 100 |
| Prompt injection | 57 | 71 | 100 | 100 | 100 | 71 | 100 |
| Nói về spam | 100 | 100 | 20 | 40 | 40 | 80 | 80 |
| Tin giao dịch hợp lệ | 67 | 100 | 83 | 83 | 83 | 83 | 100 |
| Tổng (56 case) | 48 | 59 | 68 | 66 | 68 | 84 | 93 |

F1 trên case khó so với F1 trên SUB 40 tin:

| Phương án | F1 (SUB 40) | F1 (case khó) | Tụt |
| --- | --- | --- | --- |
| BoW + LogReg | 0.947 | 0.293 | 0.654 |
| BoW + MLP | 1.000 | 0.465 | 0.535 |
| Enc + kNN | 1.000 | 0.719 | 0.281 |
| Enc + LogReg | 1.000 | 0.678 | 0.322 |
| Enc + MLP head | 1.000 | 0.690 | 0.310 |
| LLM nhỏ zero-shot | 0.952 | 0.870 | 0.082 |
| LLM nhỏ + prompt phòng thủ | 0.952 | 0.941 | 0.011 |

- Model tụt nhiều nhất: BoW + LogReg (0.947 xuống 0.293). Tụt ít nhất: LLM nhỏ + prompt phòng thủ (0.952 xuống 0.941). Điểm test thường quá lạc quan vì tập test cùng phân phối với train: spam kiểu quen thuộc, từ khoá rõ ràng, và còn có tin gần trùng nhau (chỉ bỏ trùng tuyệt đối).
- Cơ chế của BoW: chỉ đếm từ rời rạc. Ở nhóm né từ khoá ("C0ngr4tulat10ns", "F R E E", "c.a.s.h") chữ bị đổi dạng nên token không có trong vocab, BoW không thấy tín hiệu spam (đúng 17%). Nhóm spam giọng tự nhiên không chứa từ khoá nên BoW chỉ đúng 8%.
- Nhóm BoW làm tốt nhất ("Nói về spam" 100%, "Tưởng spam nhưng không phải" 92%) chỉ vì BoW thiên về đoán ham; đây là may mắn của model thiên về một lớp, không phải hiểu nghĩa.
- Cơ chế của LLM: đọc theo nghĩa nên đúng 100% ở các nhóm thao túng, spam giọng tự nhiên, né từ khoá. LLM + prompt phòng thủ sai 4/56 case, đều là tin ham bị đoán spam: #2 ("FREE pizza in the office kitchen right now..."), #8 ("Cash prize for the quiz night goes to the winning team..."), #9 ("Txt me ur address and I'll send the free tickets...") và #47 ("lol I just got a text saying I won £1000..."). Các tin này chứa từ khoá spam và giọng khẩn cấp nên model nghi ngờ. Không có case nào làm tất cả model cùng sai.
- Prompt injection: LLM zero-shot đúng 71%, prompt phòng thủ đúng 100%, nên phòng thủ có tác dụng. Nó cũng tăng nhóm "Tưởng spam nhưng không phải" từ 58% lên 75% và nhóm "Tin giao dịch hợp lệ" từ 83% lên 100%, không làm hỏng nhóm nào (nhóm "Nói về spam" giữ nguyên 80%).
- Cách phòng thủ ngoài prompt: chuẩn hoá và làm sạch đầu vào (xoá hoặc đánh dấu câu lệnh lạ), kiểm tra đầu ra phải đúng 1 nhãn, và thêm một mô hình train sẵn (Encoder + LogReg) để hai bên kiểm tra nhau, không để LLM là lớp kiểm tra duy nhất.
- Giới hạn của bộ test: do người viết tay, phân phối không giống thực tế. Dùng để dò điểm yếu của từng mô hình và so sánh định tính; không dùng để ước lượng độ chính xác ngoài đời, xếp hạng các mô hình sát nhau, hay chỉnh model cho tới khi đạt điểm cao (dễ overfit vào đúng bộ case này).

## 5. Thử thách cá nhân

- Đề theo SEED: 4060 % 6 = 4, nên là đề #5: "Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."
- Cách làm: viết 4 tin, chạy qua BoW + LogReg (`w`, `b`) và Encoder + LogReg (`w_e`, `b_e`), so với nhãn thật.

| # | Tin nhắn | Nhãn thật | BoW | Encoder+LogReg |
| --- | --- | --- | --- | --- |
| 1 | Your package is stuck. Click this link to fix it today. | spam | ham ✗ | ham ✗ |
| 2 | Your bank account has a problem. Log in here to keep it open. | spam | ham ✗ | ham ✗ |
| 3 | Is the gym free today? I want to win the game with you. | ham | ham ✓ | ham ✓ |
| 4 | The party is free. Do you want to come? Mom says yes. | ham | ham ✓ | ham ✓ |

- Kết quả: BoW đúng 2/4; Encoder + LogReg đúng 2/4.
- Dự đoán trước: BoW sai tin 1, 2; Encoder đúng cả 4. Đúng với BoW, sai với Encoder.
- Giải thích: tin 1 và 2 là câu ngắn, giọng thông báo bình thường (package, bank account), không có tín hiệu khuyến mãi như spam trong dữ liệu train (trúng thưởng, nhận quà), nên cả hai mô hình đều bỏ lọt. Tin 3 và 4: "free", "win" nằm trong câu hội thoại bình thường, không đủ để kéo về spam nên cả hai đoán đúng.
- Bài học: Encoder hiểu nghĩa tốt hơn BoW ở nhiều tin né từ khoá (như Thí nghiệm 3.B: BoW đúng 4/6, Encoder đúng 6/6), nhưng chỉ khi kiểu spam đó gần với dữ liệu nó đã học; dạng phishing giao hàng và ngân hàng không có trong train nên cả hai đều bỏ lọt.

## 6. Phương án cho sản phẩm thật

Yêu cầu: lọc 10 triệu SMS/ngày (khoảng 116 tin/giây trung bình), p95 < 50 ms/tin.

| Phương án | Đạt p95 < 50 ms? | Chất lượng | Chi phí |
| --- | --- | --- | --- |
| BoW + MLP | Có (2.37 ms) | F1 test 0.968; case khó: accuracy 59% | gần 0 |
| Encoder + LogReg | Có (8.5 ms) | F1 test 0.963; case khó: accuracy 66% | gần 0 |
| Encoder + MLP head | Có (21.0 ms) | F1 test 0.942; case khó: accuracy 68% | gần 0 |
| LLM nhỏ zero-shot | Không (p95 khoảng 2.6 giây) | F1 test 0.952; case khó: accuracy 84% (93% với prompt phòng thủ) | khoảng 105 USD/ngày (khoảng 3.150 USD/tháng) |

Cách tính chi phí LLM: mỗi tin 100 token vào + 1 token ra, giá gpt-6-luna 0.10/0.50 USD mỗi 1M token. 10 triệu tin × 100 token = 1 tỉ token vào × 0.10/1M = 100 USD; 10 triệu token ra × 0.50/1M = 5 USD; tổng 105 USD/ngày.

Đề xuất: hệ thống router 2 tầng.
1. Tầng 1: Encoder + LogReg (hoặc BoW + MLP nếu cần rẻ nhất) xử lý tin "chắc chắn" (p < 0.3 hoặc p > 0.7), đạt p95 dưới 50 ms.
2. Tầng 2: LLM chỉ xử lý tin "lưỡng lự" (p trong [0.3, 0.7]); nếu chiếm 5% lượng tin thì khoảng 5 USD/ngày thay vì 105 USD/ngày. Vì LLM chậm hơn 50 ms nên tin lưỡng lự phải xử lý bất đồng bộ (giữ tin lại chờ kết quả), không nằm trong đường chính.

Vì sao tốt hơn dùng một cách duy nhất: tầng 1 rẻ và nhanh nhưng yếu ở case khó (accuracy 59–68%); LLM mạnh ở case khó (84–93%) nhưng chậm và đắt. Router lấy ưu điểm của cả hai. Chấp nhận đánh đổi: tin lưỡng lự có độ trễ cao; tầng 1 vẫn có thể tự tin sai với spam chưa từng thấy trong dữ liệu train (như phishing giao hàng, ngân hàng), nên cần giám sát và cập nhật mô hình định kỳ.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Mình sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4 — Thí nghiệm 4.A | Nhờ sửa cho ô 4.A chạy được (đang lỗi RateLimitError 429 với Gemini) | AI chạy lại các ô, xác định hạn mức Gemini (20 request/ngày) là nguyên nhân, đề xuất đổi provider; chỉ sửa 1 dòng trong notebook: `LLM_PROVIDER = "groq"` | Mình tự tạo API key Groq và thêm vào Colab Secrets (không ghi key vào notebook); chạy lại 4.A, F1 = 0.952 cho cả hai model; ô kiểm tra 4.1 + 4.4 vẫn ✅ |
| Phần 3 — kNN | Mình thấy F1 kNN = 0.000 (accuracy bằng tỉ lệ ham) và nhờ tìm nguyên nhân | AI chạy ô chẩn đoán chỉ đọc (đã xoá sau khi dùng), chỉ ra `return y_pred` nằm trong vòng `for`; không sửa code của mình | Mình làm theo hướng dẫn của AI: đưa dòng return ra ngoài vòng for rồi chạy lại. Sau khi sửa, F1 kNN = 0.910 |
| Báo cáo này | Nhờ soạn bản nháp báo cáo từ kết quả trong notebook | AI soạn các mục dựa trên bảng kết quả và câu trả lời mình viết trong notebook | Mình đối chiếu số liệu với notebook sau khi chạy lại và đọc lại từng mục trước khi nộp |

Phần 1–3 mình tự viết code; AI chỉ dùng để chẩn đoán lỗi kNN và không sửa code trong vùng tự code.

Điều đã học:
- Dự đoán sai nhiều nhất ở Thí nghiệm 4.A: mình nghĩ model lớn sẽ cho F1 cao hơn, nhưng cả hai đều 0.952; bài toán dễ thì model lớn không tự động tốt hơn mà còn đắt hơn.
- Điểm F1 trên tập test thường quá lạc quan: BoW + LogReg 0.947 trên SUB nhưng chỉ còn 0.293 trên case khó; vì vậy luôn cần bộ case khó thay vì chỉ tin một con số F1.
- Ô kiểm tra tự động đạt chưa chắc code đúng: lỗi `return` của kNN vẫn qua ô kiểm tra 3.3 (chỉ dùng 2 tin), chỉ lộ ra khi nhìn F1 = 0.000 ở bảng so sánh.
