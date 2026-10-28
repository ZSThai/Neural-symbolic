# Bước 1 — cách chạy

## Cách khuyến nghị: Google Colab (khoảng 30 giây, chỉ cần CPU)

1. Tải cả thư mục `claude_research` lên Google Drive, vào `MyDrive/claude_research`.
2. Mở `step1/Step1_CBIS_DDSM_Colab.ipynb` bằng Colab (chuột phải → *Open with → Google Colaboratory*), hoặc vào colab.research.google.com → *Upload*.
3. Chọn *Runtime → Run all*.
   - Khi được hỏi quyền Google Drive, chọn cho phép.
   - Notebook tự tìm CSV theo thứ tự: thư mục `official_csv/` trong phiên → tải từ TCIA → `MyDrive/claude_research/data/official_csv/` → hộp thoại tải file lên.
   - CSV được kiểm tra SHA-256 trước khi dùng.
4. Kết quả (bảng CSV, `summary.json`) được lưu vào `MyDrive/claude_research/step1_results/` và tải về dạng zip.

Notebook **tự chứa**: không cần các file `.py` bên dưới. Mỗi cell có tiêu đề và có thể thu gọn. Đặt `FAST = True` ở cell 0.1 để chạy thử khoảng 15 giây.

| Phần notebook | Nội dung | Hình |
|---|---|---|
| 0 | Cấu hình, tải và kiểm tra dữ liệu | – |
| 1 | Đọc CSV, ontology BI-RADS | bảng ánh xạ |
| 2 | 1a Audit | tỉ lệ ác tính theo khái niệm; heatmap rò rỉ assessment; đồng thuận và nhầm lẫn CC↔MLO |
| 3 | Luật mờ | bản đồ luật Shape×Margin; phân rã giải thích một ca |
| 4 | 1b Trần biểu tượng A | biểu đồ AUROC + CI; độ nhạy; lượng thông tin |
| 5 | 1c Dung sai perception | đường cong soft vs crisp; nhiễu con người vs ngẫu nhiên |
| 6 | Tổng kết tự sinh, lưu kết quả | – |

## Cách phụ: chạy file .py trên máy

Cần `numpy pandas scikit-learn matplotlib`.

```bash
cd claude_research/step1
python s1_audit.py
python s1_symbolic_ceiling.py      # vài phút (bootstrap không vector hóa)
python s1_perception_tolerance.py
```

| File | Vai trò |
|---|---|
| `Step1_CBIS_DDSM_Colab.ipynb` | **Bản chính**, tự chứa, tối ưu cho Colab |
| `ontology/birads_mass_lexicon.json` | Từ vựng CBIS → BI-RADS 5, thứ bậc, hạng nghi ngờ |
| `s1_common.py`, `s1_fuzzy_rules.py` | Hàm dùng chung, hệ luật mờ |
| `s1_audit.py`, `s1_symbolic_ceiling.py`, `s1_perception_tolerance.py` | Bản script → `results/` |

Mọi mô hình hóa chỉ dùng train chính thức; test chính thức chỉ được đếm số dòng.
