# Nhiệm vụ cho Antigravity: đưa claude_research lên GitHub và chạy Bước 1 trên Colab

Người giao việc: Thai Le. Tài liệu do Claude soạn ngày 28/10/2026. Đây là dự án `claude_research`, độc lập với `Research_Workspace/`. Không áp dụng quy trình bầu vai trò của `Research_Workspace/team/` cho thư mục này.

## Phạm vi được phép

- Tạo repo GitHub, commit và push thư mục này.
- Mở notebook trên Google Colab và chạy *Run all* (CPU là đủ).
- Báo lại kết quả và lỗi nếu có.

**Không được:**

- Sửa logic khoa học: CV, ontology, mô hình, ngưỡng, seed.
- Dùng tập test chính thức để mô hình hóa.
- Sửa trực tiếp `.ipynb`. Nếu cần sửa notebook, sửa `tools/build_notebook.py` rồi dựng lại bằng `python tools/build_notebook.py step1/Step1_CBIS_DDSM_Colab.ipynb`. Mọi thay đổi khác phải báo người dùng trước.

## Bước 1. Đẩy lên GitHub

Repo là chính thư mục `claude_research` (chứa `README.md`, `data/`, `step1/`, `tools/`). Trong PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File ".\tools\push_to_github.ps1" -Repo "ZSThai/Neural-symbolic" -Visibility private
```

Script sẽ:

1. Thay `ZSThai/Neural-symbolic` trong notebook và tài liệu bằng tên repo thật.
2. `git init -b main`, rồi commit.
3. Tạo repo bằng `gh` nếu có, rồi push.

Nếu không có `gh`, cần tạo repo rỗng trên github.com trước (không thêm README).

Dữ liệu `data/official_csv/` (khoảng 2 MB, CC BY 3.0, xem `ATTRIBUTION.txt`) **phải được commit**, vì notebook đọc CSV từ repo.

## Bước 2. Chạy trên Colab

1. Mở `https://colab.research.google.com/github/ZSThai/Neural-symbolic/blob/main/step1/Step1_CBIS_DDSM_Colab.ipynb`.
   - Repo private: Colab hỏi quyền GitHub, chọn cho phép và đánh dấu *include private repos*.
   - Repo private: thêm secret `GITHUB_TOKEN` (token chỉ cần quyền đọc nội dung repo) trong biểu tượng 🔑 bên trái Colab, bật *Notebook access*. Như vậy cell 0.2 mới đọc được CSV.
   - Cách thay thế, không cần token: thêm một cell đầu tiên `!git clone https://github.com/ZSThai/Neural-symbolic.git && %cd REPO/step1`. Notebook sẽ tự tìm `../data/official_csv`.
2. Chọn *Runtime → Run all*. Thời gian khoảng 30 giây. Mặc định không cần quyền Google Drive (`SAVE_TO_DRIVE=False`).
3. Cell cuối in một dòng bắt đầu bằng `STEP1_RESULT_JSON`.

## Bước 3. Kiểm tra kết quả

So `STEP1_RESULT_JSON` với giá trị tham chiếu (đã chạy ngày 28/10/2026, scikit-learn 1.9.1). Sai khác cho phép ±0,005. Riêng `random_same_rate` cho phép ±0,02.

| Khóa | Giá trị tham chiếu |
|---|---:|
| `A_fuzzy` | 0.8698 |
| `A_logreg` | 0.8669 |
| `expert_no_learning` | 0.8525 |
| `with_assessment_leak` | 0.9110 |
| `need_acc_soft` | 0.813 |
| `need_acc_crisp` | 0.843 |
| `cross_view_own` | 0.8791 |
| `cross_view_swap` | 0.8752 |
| `random_same_rate` | 0.7956 |

Cell 0.2 phải in `sha256_khớp = True` cho cả 4 file.

## Bước 4. Báo lại

Gửi người dùng:

- Link repo, link Colab.
- Toàn bộ dòng `STEP1_RESULT_JSON`.
- Bảng so sánh với giá trị tham chiếu.
- Mọi cảnh báo hoặc lỗi kèm tên cell.

Không tự diễn giải lại kết luận khoa học. Phần diễn giải nằm ở `02_Buoc1_Phan_tich_chi_tiet.md`.
