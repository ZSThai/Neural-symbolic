# claude_research — làm lại nghiên cứu fuzzy neuro-symbolic trên CBIS-DDSM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZSThai/Neural-symbolic/blob/main/step1/Step1_CBIS_DDSM_Colab.ipynb)

Dự án của Thai Le với GS Umberto Straccia. Thư mục do Claude tạo ngày 28/10/2026. Mọi phân tích và mã ở đây được viết mới, độc lập với mã của các phiên bản trước. Dữ liệu đầu vào duy nhất là 4 CSV chính thức của CBIS-DDSM (TCIA, CC BY 3.0) trong `data/official_csv/`.

## Chạy nhanh trên Colab

Bấm nút *Open In Colab* ở trên, rồi chọn *Runtime → Run all*. Chạy khoảng 30 giây, chỉ cần CPU. Hướng dẫn đầy đủ, kể cả repo private, xem `ANTIGRAVITY_COLAB.md`.

## Đọc theo thứ tự

1. `00_Phan_tich_email_GS_va_dinh_vi_nghien_cuu.md`: email của GS nói gì, điều kiện "A là cận trên" khi nào đúng, đóng góp và câu hỏi nghiên cứu.
2. `01_Lo_trinh_nghien_cuu_CBIS_DDSM.md`: lộ trình 8 bước, mỗi bước có cổng quyết định.
3. `02_Buoc1_Phan_tich_chi_tiet.md`: bước đầu tiên, làm lại từ đâu, kết quả và quyết định.
4. `step1/`: notebook Colab (bản chính), script `.py`, ontology BI-RADS v0, kết quả.

## Cấu trúc

```
claude_research/
├── 00_…md, 01_…md, 02_…md        tài liệu phân tích
├── ANTIGRAVITY_COLAB.md          hướng dẫn đẩy GitHub + chạy Colab + kiểm tra kết quả
├── data/official_csv/            4 CSV chính thức (SHA-256 được kiểm tra khi chạy)
├── step1/
│   ├── Step1_CBIS_DDSM_Colab.ipynb   notebook tự chứa (bản chính)
│   ├── ontology/birads_mass_lexicon.json
│   ├── s1_*.py                        bản script
│   └── results/                       kết quả đã chạy
└── tools/
    ├── build_notebook.py         nguồn của notebook (sửa ở đây rồi dựng lại)
    └── push_to_github.ps1        đẩy lên GitHub
```

## Quy ước

- Official test không được dùng cho mô hình hóa cho đến bước cuối.
- Mọi so sánh dùng CV theo bệnh nhân trên train chính thức, kèm CI bootstrap theo bệnh nhân.
- Kết quả của các lượt trước (ví dụ CNN ROI ≈ 0,82 AUROC) chỉ được dùng làm mốc định hướng.
