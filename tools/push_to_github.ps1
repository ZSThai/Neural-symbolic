# Đẩy thư mục claude_research lên GitHub.
# Cách dùng (PowerShell, đứng ở bất kỳ đâu):
#   powershell -ExecutionPolicy Bypass -File "...\claude_research\tools\push_to_github.ps1" -Repo "ten-tai-khoan/ten-repo"
# Tùy chọn: -Visibility public  (mặc định private)
# Cần: git đã cài và đã đăng nhập GitHub (Git Credential Manager), hoặc GitHub CLI `gh auth login`.

param(
    [Parameter(Mandatory = $true)][string]$Repo,
    [ValidateSet("private", "public")][string]$Visibility = "private"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -Path $root
Write-Host "Thư mục repo: $root"

# 1) Điền tên repo vào notebook/README (để Colab tự đọc dữ liệu từ GitHub). Ghi UTF-8 không BOM.
$enc = New-Object System.Text.UTF8Encoding $false
foreach ($f in @("step1\Step1_CBIS_DDSM_Colab.ipynb", "README.md", "ANTIGRAVITY_COLAB.md")) {
    $p = Join-Path $root $f
    $t = [IO.File]::ReadAllText($p, $enc)
    [IO.File]::WriteAllText($p, $t.Replace("OWNER/REPO", $Repo), $enc)
}

# 2) Khởi tạo git và commit
if (-not (Test-Path (Join-Path $root ".git"))) { git init -b main | Out-Null }
git add -A
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { git commit -m "claude_research: Step 1 CBIS-DDSM symbolic premise + Colab notebook" | Out-Null }

# 3) Tạo repo trên GitHub (nếu có gh) rồi push
$hasGh = [bool](Get-Command gh -ErrorAction SilentlyContinue)
if ($hasGh) {
    gh repo view $Repo *> $null
    if ($LASTEXITCODE -ne 0) {
        gh repo create $Repo "--$Visibility" --source . --remote origin --push
        Write-Host "Đã tạo và đẩy https://github.com/$Repo"
        exit 0
    }
}
if (-not (git remote) -or -not ((git remote) -contains "origin")) {
    git remote add origin "https://github.com/$Repo.git"
}
git push -u origin main
Write-Host "Xong: https://github.com/$Repo"
Write-Host "Mở Colab: https://colab.research.google.com/github/$Repo/blob/main/step1/Step1_CBIS_DDSM_Colab.ipynb"
