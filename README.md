# FreshRetailNet-50K Forecast Dashboard

Dashboard Streamlit để xem kết quả dự báo phân vị trên FreshRetailNet-50K.

Ứng dụng trình bày hai lượt thực nghiệm riêng:

- LightGBM trên 50.000 chuỗi và 350.000 quan sát eval.
- So sánh LightGBM, TFT và Chronos trên panel chung 100 chuỗi, tối đa 700 quan sát.

Dashboard đọc các bảng CSV và hình đã được tạo trong quá trình thực nghiệm. Ứng dụng không huấn luyện lại mô hình khi mở trang, không tải Parquet gốc và không ước lượng nhu cầu tiềm ẩn khi hết hàng.

## Cấu trúc repository

    app.py                         Ứng dụng Streamlit
    requirements.txt               Thư viện cho dashboard
    requirements-modeling.txt      Thư viện cho notebook và chạy mô hình
    src/                           Các script Python của nghiên cứu
    notebooks/                     Notebook Google Colab
    results/                       Chỉ số, hình và dự báo panel 100 chuỗi

Các tệp Parquet gốc và dự báo từng dòng LightGBM trên toàn bộ 350.000 quan sát không nằm trong repository. Nếu cần chạy lại thực nghiệm, dùng notebook hoặc script tải dữ liệu từ nguồn chính thức.

## Chạy trên máy cá nhân

1. Cài Python 3.10 trở lên.
2. Mở terminal tại thư mục repository.
3. Cài thư viện bằng lệnh:

    pip install -r requirements.txt

4. Khởi động dashboard:

    streamlit run app.py

## Đưa mã lên GitHub

Tạo repository mới, ví dụ freshretailnet50k-forecast-dashboard, rồi tải nội dung thư mục này lên nhánh main. Giữ app.py và requirements.txt ở thư mục gốc. Không tải dữ liệu Parquet gốc hoặc token lên repository.

## Triển khai trên Streamlit Community Cloud

1. Đăng nhập Streamlit Community Cloud bằng tài khoản GitHub có quyền quản trị repository.
2. Chọn tạo ứng dụng mới.
3. Chọn repository, nhánh main và tệp chạy app.py.
4. Triển khai ứng dụng. Sau khi hoàn tất, Streamlit cấp URL để truy cập dashboard.

Hướng dẫn chính thức:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Phạm vi diễn giải

sale_amount là lượng bán quan sát đã chuẩn hóa toàn cục, không phải số lượng vật lý. Nhãn stockout chỉ dùng để phân tầng sai số; chênh lệch giữa các nhóm không chứng minh quan hệ nhân quả và không cho biết lượng cầu bị mất.

Kết quả so sánh ba mô hình chỉ áp dụng cho panel 100 chuỗi được chọn với seed 17; không suy rộng trực tiếp cho toàn bộ 50.000 chuỗi. Kết quả LightGBM trên toàn bộ dữ liệu được trình bày riêng.
