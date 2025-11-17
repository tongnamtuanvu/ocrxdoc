# Hướng Dẫn Publish ocrxdoc lên PyPI

## Bước 1: Tạo PyPI Account và API Token

1. Đăng ký tài khoản tại https://pypi.org/account/register/
2. Đăng nhập và vào Settings → API tokens
3. Tạo token mới với scope: **Entire account** hoặc **Project: ocrxdoc**
4. Copy token (chỉ hiển thị 1 lần, lưu lại cẩn thận)

## Bước 2: Cấu Hình GitHub Secrets

1. Vào repository: https://github.com/tongnamtuanvu/ocrxdoc
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Paste token từ PyPI
6. Click "Add secret"

## Bước 3: Tạo GitHub Environment (Optional nhưng khuyến nghị)

1. Vào repository Settings → Environments
2. Click "New environment"
3. Name: `pypi`
4. (Optional) Thêm protection rules nếu cần
5. Click "Configure environment"

## Bước 4: Publish Package

### Cách 1: Tự động qua GitHub Release (Khuyến nghị)

1. Tạo tag mới:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. Tạo Release trên GitHub:
   - Vào repository → Releases → "Create a new release"
   - Chọn tag vừa tạo
   - Điền release notes
   - Click "Publish release"
   - GitHub Actions sẽ tự động publish lên PyPI

### Cách 2: Manual Trigger

1. Vào repository → Actions
2. Chọn workflow "Publish to PyPI"
3. Click "Run workflow"
4. Chọn branch và click "Run workflow"

## Bước 5: Kiểm Tra

Sau khi workflow chạy thành công:
- Package sẽ có trên PyPI: https://pypi.org/project/ocrxdoc/
- Có thể cài đặt: `pip install ocrxdoc`

## Lưu Ý

- **Version**: Mỗi lần publish cần tăng version trong `setup.py` và `pyproject.toml`
- **Test trước**: Nên test với TestPyPI trước khi publish lên PyPI chính thức
- **Changelog**: Nên cập nhật CHANGELOG.md khi có version mới

## Test với TestPyPI

1. Tạo account tại https://test.pypi.org/
2. Tạo token tương tự
3. Thêm secret `TEST_PYPI_API_TOKEN` vào GitHub
4. Tạo workflow riêng cho TestPyPI hoặc modify workflow hiện tại

## Troubleshooting

- **Lỗi authentication**: Kiểm tra lại PYPI_API_TOKEN trong Secrets
- **Lỗi version đã tồn tại**: Cần tăng version trong setup.py
- **Lỗi build**: Kiểm tra setup.py và pyproject.toml có đúng format không

