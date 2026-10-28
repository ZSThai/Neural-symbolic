# 00 — Phân tích email của GS Straccia và định vị nghiên cứu trên CBIS-DDSM

Người soạn: Claude (claude_research), 28/10/2026. Tài liệu được viết lại từ đầu dựa trên email của GS, ghi chú buổi họp, dữ liệu CSV chính thức của CBIS-DDSM và tài liệu khoa học công khai. Không dùng mã nguồn của các phiên bản trước. Kết quả thực nghiệm cũ chỉ được nhắc khi cần định hướng và luôn ghi rõ là "lượt trước".

---

## 1. Email của GS thực sự đặt ra bài toán gì?

Ký hiệu: **X** là ảnh, **Z** là các thuộc tính hoặc khái niệm do chuyên gia mô tả, **Y** là nhãn.

| Ý trong email | Diễn giải khoa học |
|---|---|
| (1) Accuracy A của bộ phân loại chỉ dùng thuộc tính C1 là cận trên. Nếu bộ phân loại chỉ dùng ảnh C2 có B > A thì khó cải thiện bằng luật. Nếu ngang B thì vẫn được lợi là giải thích được. | Hệ neuro-symbolic đi theo chuỗi X → Ẑ → Y. Hiệu năng của nó phụ thuộc vào (i) Z chứa bao nhiêu thông tin về Y và (ii) Ẑ khôi phục Z tốt đến đâu. |
| (2) Cần dữ liệu có B << A, gồm cả thuộc tính phân loại lẫn liên tục. | Đây là **điều kiện tiền đề** của cả bài báo. Phải đo trước khi xây mô hình. |
| (3)–(4) Audio; CBIS-DDSM; nhiều dataset ung thư vú khác. | CBIS-DDSM được chọn: có ảnh, mô tả hình thái BI-RADS và nhãn bệnh lý đã sinh thiết. |
| (5) Làm giàu bằng ontology hoặc knowledge graph theo miền; gán thực thể rồi học luật. | Với nhũ ảnh, **ontology đã có sẵn**: từ vựng BI-RADS (ACR). Đây là cầu nối tự nhiên với chuyên môn của GS (fuzzy OWL 2, fuzzyDL, học fuzzy concept inclusion như PN-OWL). |
| (6) Dự đoán sự kiện tiếp theo trong video. | Ngoài phạm vi bài báo này; giữ làm hướng tương lai. |

### 1.1. Khi nào "A là cận trên" đúng chính xác?

Câu của GS có thể phát biểu chặt chẽ bằng bất đẳng thức xử lý dữ liệu (data processing inequality). Nếu khái niệm dự đoán Ẑ không mang thông tin nào về Y ngoài những gì có trong Z thật, tức Y → Z → Ẑ là một chuỗi Markov, thì

$$I(Y;\hat Z) \le I(Y;Z).$$

Khi đó mọi bộ phân loại chỉ đọc Ẑ đều bị chặn bởi bộ phân loại tốt nhất đọc Z. Điều kiện "Ẑ không mang thông tin ngoài Z" chính là **tính trung thực ngữ nghĩa (semantic faithfulness)** của nút thắt khái niệm. Nếu điều kiện này bị phá, ví dụ Ẑ lén mã hóa thêm đặc trưng ảnh (concept leakage, một hiện tượng đã biết ở concept bottleneck models), thì hệ có thể vượt A, nhưng lời giải thích không còn đáng tin.

Hệ quả cho thiết kế: một hệ neuro-symbolic **trung thực** không thể vượt A. Nó chỉ có thể vượt B khi:

1. **A − B đủ lớn** (khoảng trống thông tin của Z so với ảnh), và
2. **độ khôi phục khái niệm R đủ cao** để phần lớn khoảng trống còn lại sau sai số perception.

Bài báo vì vậy cần đo ba đại lượng chứ không chỉ hai: **A** (trần biểu tượng), **B** (ảnh thuần) và **N(R)** (hiệu năng của lớp luật như một hàm của chất lượng khái niệm). Bước 1 đo A và hàm N(R) chỉ bằng CSV. Bước 3 đo B và R bằng ảnh.

### 1.2. Vì sao "explainable khi ngang B" vẫn là một đóng góp, và cần chứng minh thế nào

Nếu không vượt được B, giá trị còn lại là giải thích. "Giải thích" phải được đo, không chỉ tuyên bố:

- **Faithfulness:** điểm số được phân rã chính xác thành đóng góp của từng luật.
- **Can thiệp khái niệm:** khi một bác sĩ sửa một khái niệm sai, dự đoán thay đổi đúng hướng và hiệu năng tăng. Đây là lợi thế mà mô hình hộp đen không có. Một công trình CBM trên siêu âm vú (Bunnell et al., MICCAI 2024) báo cáo AUROC tăng từ 0,863 lên 0,885 khi sửa khái niệm tối thiểu.
- **Không kém hơn (non-inferiority)** với biên được khai báo trước. Hai điểm ước lượng gần nhau tự nó không chứng minh được tương đương.

---

## 2. Vì sao CBIS-DDSM, và những điểm cần cẩn trọng

**Phù hợp:**

- Ảnh mammography số hóa từ phim, có ROI, mask, mô tả **mass shape** và **mass margins** theo từ vựng BI-RADS, cùng nhãn bệnh lý (Lee et al., *Scientific Data* 2017).
- Từ vựng BI-RADS là một ontology có thứ bậc, có ý nghĩa lâm sàng và có thứ tự nghi ngờ (ví dụ margin spiculated nghi ngờ hơn circumscribed). Đúng tinh thần ý (5) của GS.
- Chú giải chứa ô **đa trị** ("ILL_DEFINED-SPICULATED"): bản thân chú giải đã mơ hồ. Đây là lý do tự nhiên, không gượng ép, để dùng logic mờ.

**Cần cẩn trọng** (các con số đã kiểm tra lại trên CSV chính thức, xem `02_Buoc1_...`):

- **Assessment (BI-RADS 0–5) là rò rỉ.** Trên train, assessment 5 có 293/299 ca ác tính, assessment 2 có 0/77. Đây là kết luận tổng thể của bác sĩ, gần như chính là Y. Không được đưa vào Z.
- **BENIGN_WITHOUT_CALLBACK** (104 dòng train) là nhãn lành không qua sinh thiết. Nên phân tích độ nhạy riêng.
- **Một phần "shape" không phải hình dạng mass theo BI-RADS 5:** architectural distortion, asymmetry, lymph node (145 dòng train chỉ có loại này).
- **Chú giải có nhiễu đáng kể:** cùng một tổn thương, hai view CC và MLO khớp pathology 99,6%, nhưng **shape chỉ khớp 82,3%, margins 83,6%**. Trần độ chính xác khái niệm đạt được từ ảnh vì thế thấp hơn 100% nhiều.
- Mask ROI là bản tinh chỉnh bán tự động (Chan-Vese). Bài gốc báo cáo Dice khoảng 0,79 so với vẽ tay trên một tập nhỏ. Thuộc tính hình học tính từ mask sẽ có sai số.
- CSV **không có thuộc tính liên tục**. Muốn đáp ứng yêu cầu "categorical + continuous" của GS phải tự tính từ mask/ảnh (Bước 2). Thống kê pixel đơn giản như mean/std intensity của crop **không phải khái niệm**: chúng tính trực tiếp được từ ảnh lúc suy luận, nên đưa qua nút thắt không có ý nghĩa ngữ nghĩa.

---

## 3. Nhìn lại hướng trước một cách thẳng thắn (chỉ dựa trên báo cáo, không đọc code)

| Điểm | Nhận xét |
|---|---|
| Đã làm tốt | Pipeline strict (nhãn bắt buộc đi qua luật), chia theo bệnh nhân, bootstrap theo bệnh nhân, giữ test riêng. |
| Tiền đề chưa được kiểm trước | Mô hình được xây trước khi chứng minh B << A và trước khi biết lớp luật cần khái niệm chính xác đến đâu. |
| Thuộc tính liên tục | Là thống kê pixel của crop, không mang ngữ nghĩa lâm sàng và tính trực tiếp được lúc suy luận. |
| Thước đo khái niệm | Macro-F1 trên các giá trị hiếm (ROUND, asymmetry) bị kéo xuống. Thứ quyết định chẩn đoán là sai số **theo mức nghi ngờ**, không phải theo nhãn từ vựng. Phân tích của Bước 1 cho thấy khác biệt này rất lớn. |
| Chưa dùng tri thức | Không có ontology hay thứ tự nghi ngờ BI-RADS. Luật học thuần dữ liệu. |
| Đối chứng còn thiếu | CBM tuyến tính, CNN đa nhiệm có giám sát khái niệm, và luật crisp tối ưu độc lập. |
| Validation | Đã dùng lặp lại qua nhiều phase. Cần khóa protocol trước khi đánh giá test. |

---

## 4. Định vị so với tài liệu đã công bố

| Công trình | Nội dung | Khác biệt cần tạo ra |
|---|---|---|
| Ibrahimi & Straccia (2026), melanoma, preprint | EfficientNet-B3 ước lượng mức thuộc các thuộc tính dermoscopy; luật trích tự động từ chú giải; suy luận bằng toán tử mờ; cấu hình tốt nhất là đa phương thức. | Đây là bài "anh em". Bài mới không được chỉ là "cùng ý tưởng trên mammography". |
| Koh et al. (ICML 2020), Concept Bottleneck Models | Nút thắt khái niệm, can thiệp khái niệm. | Nền tảng; cần một nhánh CBM làm đối chứng. |
| Ma & Gilhuijs (2026), Prior-guided CBM | **Đã dùng CBIS-DDSM mass** với shape, margin, density (18 multi-hot). Diagnosis AUC 0,825 khi có 10% chú giải khái niệm. | "CBM trên CBIS-DDSM với shape/margin" **không còn mới**. Cần đóng góp ở lớp logic mờ, ontology và phân tích tiền đề. |
| Bunnell et al. (MICCAI 2024), BI-RADS CBM cho siêu âm | Khái niệm BI-RADS; can thiệp khái niệm tăng AUROC. | Chứng cứ cho giá trị của can thiệp; dùng làm thiết kế đánh giá. |
| Barnett et al. (2021), IAIA-BL | Mạng prototype có giải thích cho margin mass. | Hướng giải thích theo ca, khác với hướng luật mờ. |
| Cardillo & Straccia, PN-OWL | Học fuzzy concept inclusion từ ontology OWL. | **Công cụ học luật tự nhiên nhất** cho lớp biểu tượng, gắn với chuyên môn của GS. |
| Fumanal-Idocin et al. (2025), fast fuzzy tree | Cây mờ nhanh, luật gọn. | Đối chứng cho bộ học luật. |

### Các đóng góp có thể bảo vệ được

1. **Khung phân tích tiền đề (A, B, R):** khi nào hệ neuro-symbolic trung thực có thể vượt mô hình ảnh. Có điều kiện xử lý dữ liệu rõ ràng và được đo trên CBIS-DDSM. Phát hiện ở Bước 1 về "nhiễu có cấu trúc của con người" là một kết quả mới, kiểm chứng được.
2. **Từ vựng mờ gắn ontology BI-RADS:** ô đa trị trở thành mức thuộc mờ; thuộc tính hình thái **liên tục** tính từ mask chuyên gia được mờ hóa và liên kết với khái niệm từ vựng (ví dụ độ tròn thấp ↔ Irregular).
3. **Học luật bằng fuzzy DL (PN-OWL hoặc tương đương) trên ontology,** so với cây mờ và luật crisp.
4. **Huấn luyện perception có nhận thức ontology:** sai số giữa hai khái niệm cùng mức nghi ngờ bị phạt nhẹ hơn sai số vượt mức. Động cơ trực tiếp đến từ dữ liệu Bước 1.
5. **Đánh giá giải thích có định lượng:** faithfulness, can thiệp khái niệm, và non-inferiority có biên định trước.

---

## 5. Câu hỏi nghiên cứu và giả thuyết có thể bác bỏ

| | Câu hỏi | Giả thuyết | Bác bỏ nếu |
|---|---|---|---|
| RQ1 | Trên CBIS-DDSM mass, Z (shape, margin, cộng thuộc tính liên tục từ mask) chứa nhiều thông tin hơn ảnh không? | A − B ≥ 0,03 AUROC (ngưỡng làm việc, khai báo trước) | Với B đo công bằng ở Bước 3, CI của A − B chứa 0 |
| RQ2 | Một hệ luật mờ nhỏ trên Z có đạt gần mô hình tốt nhất trên Z không? | Luật mờ ≥ LR/GBM − 0,01 AUROC với ≤ 30 luật | Khoảng cách > 0,02 |
| RQ3 | Với ảnh làm đầu vào duy nhất, X → Ẑ → luật có đạt hoặc vượt B không? | Không kém hơn B với biên 0,02 AUROC; vượt B nếu R đủ cao | CI của chênh lệch vượt biên |
| RQ4 | Tri thức ontology (thứ bậc, thứ tự nghi ngờ) có cải thiện perception hoặc luật không? | Loss nhận thức ontology tăng AUROC hoặc giảm sai số vượt mức nghi ngờ | Không cải thiện trên CV train |
| RQ5 | Giải thích có hữu ích đo được không? | Can thiệp k khái niệm sai nhất tăng AUROC rõ rệt | Không tăng |

---

## 6. Nguồn

- Email GS (`Email của GS sau khi họp.txt`) và ghi chú họp (`meeting_saved_new_chat.txt`).
- CBIS-DDSM: Lee R.S. et al., "A curated mammography data set for use in computer-aided detection and diagnosis research", *Scientific Data* 4, 170177 (2017), doi:10.1038/sdata.2017.177; TCIA doi:10.7937/K9/TCIA.2016.7O02S9CY.
- [Ibrahimi & Straccia, A Neuro-Symbolic Fuzzy Rule-based Approach for Melanoma Classification (preprint, 2026)](https://doi.org/10.21203/rs.3.rs-10142900/v1)
- [Ma & Gilhuijs, Prior-Guided Concept Bottleneck Models for Interpretable Cancer Imaging Diagnosis (arXiv 2608.13148)](https://arxiv.org/pdf/2608.13148)
- [Bunnell et al., Learning a Clinically-Relevant Concept Bottleneck for Lesion Detection in Breast Ultrasound (MICCAI 2024)](https://papers.miccai.org/miccai-2024/paper/4008_paper.pdf)
- [Barnett et al., IAIA-BL (arXiv 2103.12308)](https://arxiv.org/abs/2103.12308v1)
- [Cardillo & Straccia, PN-OWL (arXiv 2303.07192)](https://arxiv.org/pdf/2303.07192)
- [Fumanal-Idocin et al., A Fast Interpretable Fuzzy Tree Learner (arXiv 2512.11616)](https://arxiv.org/html/2512.11616)
- [Interpretable mammographic mass classification with fuzzy interpolative reasoning (Knowledge-Based Systems)](https://research.aber.ac.uk/en/publications/interpretable-mammographic-mass-classification-with-fuzzy-interpo/)
- Koh P.W. et al., "Concept Bottleneck Models", ICML 2020.
- ACR BI-RADS Atlas, 5th edition (2013): mass shape (oval, round, irregular), margin (circumscribed, obscured, microlobulated, indistinct, spiculated).
