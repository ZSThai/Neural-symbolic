# Cấu hình Phương án A: Tự động cập nhật Notebook, đồng bộ GitHub/Drive và mở trực tiếp trên Google Chrome (Colab).
# Cách dùng:
#   powershell -ExecutionPolicy Bypass -File "claude_research\tools\open_in_colab.ps1"
# Hoặc chỉ định một notebook khác (ví dụ ở Bước 2, Bước 3):
#   powershell -ExecutionPolicy Bypass -File "claude_research\tools\open_in_colab.ps1" -Notebook "step2\Step2_Colab.ipynb"

param(
    [string]$Notebook = "step1\Step1_CBIS_DDSM_Colab.ipynb",
    [switch]$SkipBuild,
    [switch]$SkipPush,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -Path $root

# 1. Tự động build lại notebook Step 1 nếu có chỉnh sửa trong tools/build_notebook.py
if (-not $SkipBuild -and ($Notebook -match "Step1_CBIS_DDSM_Colab\.ipynb$")) {
    Write-Host "[1/3] Đang tạo lại notebook từ tools/build_notebook.py..." -ForegroundColor Cyan
    python tools/build_notebook.py $Notebook
}

# 2. Đồng bộ lên GitHub (để link Colab luôn là bản mới nhất ngay cả khi Google Drive Desktop đang xếp hàng đồng bộ)
if (-not $SkipPush -and (Test-Path (Join-Path $root ".git"))) {
    Write-Host "[2/3] Kiểm tra và đồng bộ thay đổi lên GitHub..." -ForegroundColor Cyan
    git add -A
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Auto-sync notebook before opening in Google Colab" | Out-Null
        git push origin main
        Write-Host "      Đã đẩy bản cập nhật mới nhất lên GitHub." -ForegroundColor Green
    } else {
        Write-Host "      Không có thay đổi mới cần đẩy." -ForegroundColor Gray
    }
}

# 3. Mở Google Chrome trực tiếp tới Google Colab (ưu tiên Drive File ID nếu đã đồng bộ xong, hoặc GitHub)
Write-Host "[3/3] Đang khởi chạy Google Chrome với Google Colab..." -ForegroundColor Cyan
if ($NoBrowser) {
    python tools/open_colab.py $Notebook --no-browser
} else {
    python tools/open_colab.py $Notebook
}
