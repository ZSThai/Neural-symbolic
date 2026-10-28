# 01 — Lộ trình nghiên cứu step by step trên CBIS-DDSM (bản làm lại)

Nguyên tắc chung cho mọi bước:

- **Tiền đề trước, mô hình sau.** Mỗi bước kết thúc bằng một **cổng quyết định** có tiêu chí định lượng khai báo trước.
- **Official test** (378 dòng mass / 201 bệnh nhân) không được nhìn tới cho đến Bước 8. Mọi lựa chọn dùng CV theo bệnh nhân **chỉ trên train chính thức**.
- **Thước đo chính: AUROC.** Accuracy chỉ báo cáo kèm ngưỡng cố định chọn trên CV train. Luôn báo cáo CI bootstrap theo bệnh nhân.
- **Mọi tri thức miền** (ontology, thứ tự nghi ngờ) đều ghi trong file, có phiên bản, và cần GS hoặc bác sĩ xác nhận.
- **Không tái sử dụng mã cũ.** Mã mới nằm trong `claude_research/stepN/`.

```
Bước 1  Tiền đề biểu tượng (chỉ CSV)            → A, dung sai perception, ontology v0      [ĐÃ LÀM]
Bước 2  Dữ liệu ảnh + thuộc tính liên tục        → cohort ảnh/mask đã QC, Z liên tục từ mask
Bước 3  Mốc ảnh B và độ khôi phục khái niệm R    → B công bằng, R theo từng khái niệm
Bước 4  Cổng tiền đề A–B–R                        → chốt claim chính của bài báo
Bước 5  Từ vựng mờ + học luật trên Z              → luật mờ/fuzzy DL, so với crisp
Bước 6  Mô hình neuro-symbolic X → Ẑ → luật → Y   → independent → joint, loss nhận thức ontology
Bước 7  Đánh giá giải thích                       → faithfulness, can thiệp, độ bền
Bước 8  Đánh giá khóa trên official test, viết bài
```

---

## Bước 1 — Tiền đề biểu tượng (đã thực hiện; chi tiết ở `02_Buoc1_...`)

- **Mục tiêu:** Từ CSV chính thức, xác định Z hợp lệ và đo trần biểu tượng A. Đo xem lớp luật chịu được sai số khái niệm đến đâu trước khi thua một mốc ảnh.
- **Không cần GPU hay ảnh.** Chạy được trên máy cá nhân.
- **Cổng 1:** Z hợp lệ (không rò rỉ) cho A > B tham chiếu, và độ chính xác khái niệm cần thiết nằm trong tầm với của con người. **Kết quả: qua cổng, nhưng biên hẹp.** Vì vậy cần Bước 2 (thêm thuộc tính liên tục) để mở rộng A.

## Bước 2 — Dữ liệu ảnh và thuộc tính liên tục có ngữ nghĩa

- **Tải dữ liệu.** Lấy ảnh mass (full mammogram, crop, mask) từ TCIA/NBIA hoặc NCI Imaging Data Commons (gói `idc-index`). Chỉ tải phần mass trước, khoảng vài chục GB. Ghép bằng SeriesInstanceUID trong CSV.
- **QC.** Mask khớp ảnh; đúng chiều; MONOCHROME1/2; crop nằm trong full image; trùng lặp. Xem mẫu ngẫu nhiên có ghi nhận.
- **Thuộc tính liên tục từ mask chuyên gia** (tính trên ảnh gốc theo pixel spacing nếu có), mỗi thuộc tính gắn với một khái niệm BI-RADS:

| Thuộc tính | Định nghĩa | Liên kết ontology |
|---|---|---|
| Diện tích (mm²), trục lớn/nhỏ | Từ mask | Kích thước |
| Circularity | 4π·Area / Perimeter² | Round/Oval ↔ Irregular |
| Solidity | Area / ConvexArea | Irregular, Lobulated |
| Eccentricity / aspect | Từ moments | Round vs Oval |
| Độ nhám đường bờ | Năng lượng tần số cao của hàm khoảng cách tâm–bờ | Microlobulated |
| Chỉ số spiculation | Tỉ lệ gradient hướng tâm trong vành ngoài mask | Spiculated |
| Độ sắc nét bờ | Gradient cường độ vuông góc với bờ | Circumscribed ↔ Indistinct/Obscured |
| Tương phản tổn thương–nền | Cường độ trong mask so với vành ngoài | Mật độ tổn thương |

- **Điểm mấu chốt:** lúc suy luận **không có mask**, nên các thuộc tính này phải được **dự đoán từ ảnh**. Chúng là khái niệm thật, không phải thống kê pixel tầm thường.
- **Chạy lại Bước 1b** với Z hỗn hợp (nominal + liên tục).
- **Cổng 2:** A(Z hỗn hợp) > A(Z_morph). Thuộc tính liên tục đo được ổn định (so CC với MLO của cùng tổn thương).

## Bước 3 — Mốc ảnh B và độ khôi phục khái niệm R

- **B công bằng:** CNN trên ROI với cùng cohort, CV theo bệnh nhân. Hai đến ba backbone (ví dụ ResNet-50, EfficientNet, ConvNeXt-T), ngân sách tìm siêu tham số ngang nhau, có augmentation hợp lý. Không cố làm yếu CNN.
- **R:** bộ dự đoán khái niệm OOF. Đo theo **ba mức**:
  1. AUROC/AP từng giá trị từ vựng.
  2. Độ chính xác theo **mức nghi ngờ** (benign-appearing / indeterminate / suspicious).
  3. MAE từng thuộc tính liên tục.
- **So R với trần con người** (đồng thuận CC↔MLO): 82–84% theo từ vựng, cao hơn nhiều theo mức nghi ngờ.
- **Đối chứng bắt buộc:** CNN đa nhiệm (nhãn + khái niệm) và CBM tuyến tính. Hai nhánh này tách lợi ích của *giám sát khái niệm* khỏi lợi ích của *logic mờ*.

## Bước 4 — Cổng tiền đề A–B–R (quyết định quan trọng nhất)

Đặt điểm (R đo được) lên đường cong dung sai của Bước 1c. Ba kịch bản:

| Kịch bản | Điều kiện | Claim chính của bài |
|---|---|---|
| I | N(R) > B, CI tách khỏi 0 | Neuro-symbolic **vượt** ảnh thuần và giải thích được |
| II | N(R) ≈ B | **Không kém hơn** với biên khai báo trước, cộng lợi ích can thiệp và giải thích |
| III | N(R) << B | Đổi trọng tâm: mô hình lai (luật + residual có kiểm soát), hoặc thêm dataset có A − B lớn hơn (theo ý 2 của GS) |

Kịch bản và biên non-inferiority được ghi vào `preregistration.md` **trước khi** chạy Bước 6.

## Bước 5 — Từ vựng mờ và học luật trên Z

- **Nominal:**
  - Ô đa trị trở thành mức thuộc (so sánh ngữ nghĩa possibilistic và probabilistic).
  - Thứ bậc ontology cho khái niệm cha (SuspiciousMargin = Microlobulated ⊔ Indistinct ⊔ Spiculated), dùng t-conorm.
- **Liên tục:**
  - Phân hoạch mờ K=3 (Low/Medium/High), học trong từng fold train (FCM hoặc quantile, có shoulder).
  - So K=3 với K=5 **trên CV train**.
- **Bộ học luật:** (a) luật lưới làm trơn phân cấp (đã có ở Bước 1), (b) cây mờ, (c) **PN-OWL / fuzzy concept inclusion** trên ontology (nên làm cùng GS), (d) cây crisp và LR làm đối chứng.
- **Ràng buộc tri thức:** đơn điệu theo mức nghi ngờ; số luật ≤ 30; mỗi luật ≤ 3 điều kiện.
- **Cổng 5:** luật mờ ≥ LR − 0,01 AUROC trên Z.

## Bước 6 — Mô hình neuro-symbolic X → Ẑ → luật → Y

1. **Independent:** perception học khái niệm, luật cố định từ Bước 5.
2. **Loss nhận thức ontology:** phạt nhầm vượt mức nghi ngờ nặng hơn nhầm trong cùng mức. Mục tiêu soft từ ô đa trị.
3. **Joint:** gradient đi xuyên qua luật mờ (t-norm product khả vi), có neo ngữ nghĩa để Ẑ không trôi nghĩa. Theo dõi "concept leakage".

Đối chứng như Bước 3. Mọi lựa chọn bằng CV train.

## Bước 7 — Đánh giá giải thích

- **Faithfulness:** phân rã điểm thành đóng góp luật; xóa luật → thay đổi dự đoán.
- **Can thiệp khái niệm:** thay k khái niệm sai nhất bằng chú giải → đường cong AUROC theo k. So với CBM.
- **Hai lớp nhiễu:** nhiễu kiểu con người (thay chú giải bằng view kia) và nhiễu kiểu máy.
- **Ca minh họa** có ảnh, khái niệm, luật kích hoạt và đóng góp từng luật. Nếu có thể, nhờ bác sĩ đọc.

## Bước 8 — Đánh giá khóa và viết bài

- Khóa toàn bộ cấu hình, chạy official test **một lần**.
- Báo cáo đủ: A, B, N; CI; non-inferiority; can thiệp; giới hạn (phim số hóa thập niên 1990, ROI cho sẵn, nhãn BWC).
- Mở rộng có điều kiện: calcification (calc type, distribution) làm miền thứ hai, hoặc audio (theo ý 3 của GS).

---

## Lịch đề xuất (cần điều chỉnh theo lịch họp với GS)

| Thời gian | Việc |
|---|---|
| Tuần này | Bước 1 xong. Gửi GS 1 trang tóm tắt và đề nghị xác nhận ontology (file JSON) |
| Tuần 1–2 tháng 11 | Bước 2 (tải mass, QC, thuộc tính từ mask) |
| Tuần 3 tháng 11 | Bước 3 trên Colab GPU |
| Cuối tháng 11 | Cổng 4 và preregistration |
| Tháng 12 | Bước 5–7 |
| Tháng 1 | Bước 8, bản thảo |
