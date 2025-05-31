# Chương trình nhận dạng cảm xúc văn bản sử dụng CafeBERT và VnEmoLex

Chương trình này sử dụng mô hình CafeBERT kết hợp với từ điển cảm xúc tiếng Việt VnEmoLex để nhận dạng cảm xúc trong văn bản tiếng Việt.

## Cấu trúc dự án

```
├── main.py                # File chính để huấn luyện và đánh giá mô hình
├── predict.py             # File dự đoán cảm xúc cho văn bản mới
├── evaluate.py            # File đánh giá chi tiết mô hình
├── config.py              # File cấu hình tham số
├── train.csv              # Dữ liệu huấn luyện
├── valid.csv              # Dữ liệu kiểm định
├── test.csv               # Dữ liệu kiểm tra
├── VnEmoLex.csv           # Từ điển cảm xúc tiếng Việt
├── requirements.txt       # Các thư viện cần thiết
├── README.md              # Hướng dẫn sử dụng
├── PREDICT_GUIDE.md       # Hướng dẫn sử dụng công cụ dự đoán
├── MODEL_ARCHITECTURE.md  # Mô tả kiến trúc mô hình
├── RESULTS_GUIDE.md       # Hướng dẫn phân tích kết quả
├── models/                # Thư mục lưu mô hình
├── logs/                  # Thư mục lưu thông tin tiền xử lý và huấn luyện
└── results/               # Thư mục lưu kết quả đánh giá
```

## Các thư viện cần thiết

Chương trình sử dụng các thư viện sau:

- torch
- transformers
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- imbalanced-learn

Bạn có thể cài đặt các thư viện này bằng lệnh:

```bash
pip install -r requirements.txt
```

## Cách sử dụng

### Huấn luyện và đánh giá mô hình

```bash
python main.py --imbalance_method [method]
```

Trong đó `[method]` là phương pháp xử lý dữ liệu mất cân bằng, có thể là một trong các giá trị sau:
- `none`: Không áp dụng phương pháp xử lý dữ liệu mất cân bằng
- `class_weight`: Sử dụng trọng số lớp (mặc định)
- `weighted_sampler`: Sử dụng weighted sampler
- `oversampling`: Sử dụng oversampling

### Đánh giá chi tiết mô hình

```bash
python evaluate.py --model [model_path] --analyze_errors
```

Trong đó:
- `--model [model_path]`: Đường dẫn đến file mô hình đã huấn luyện (mặc định: models/best_model.pt)
- `--analyze_errors`: Thêm tùy chọn này để phân tích lỗi

### Dự đoán cảm xúc cho văn bản mới

```bash
python predict.py --text "Văn bản cần dự đoán"
```

hoặc

```bash
python predict.py --file path/to/file.txt
```

Xem thêm chi tiết trong file [PREDICT_GUIDE.md](PREDICT_GUIDE.md).

## Các tham số có thể điều chỉnh

Bạn có thể điều chỉnh các tham số trong file `config.py`:

- `DATA_CONFIG`: Cấu hình dữ liệu (đường dẫn, độ dài tối đa)
- `TRAINING_CONFIG`: Cấu hình huấn luyện (batch size, epochs, learning rate, ...)
- `MODEL_CONFIG`: Cấu hình mô hình (tên mô hình, kích thước lớp ẩn, ...)
- `PATH_CONFIG`: Cấu hình đường dẫn (thư mục lưu mô hình, logs, kết quả)

## Kết quả

Sau khi chạy chương trình, các kết quả sẽ được lưu trong các thư mục sau:

- `models/`: Lưu mô hình tốt nhất
- `logs/`: Lưu thông tin tiền xử lý, phân phối dữ liệu, lịch sử huấn luyện
- `results/`: Lưu kết quả đánh giá, biểu đồ, ma trận nhầm lẫn

## Các chỉ số đánh giá

Chương trình sử dụng các chỉ số đánh giá sau:

- Accuracy: Độ chính xác tổng thể
- Macro F1-score: F1-score trung bình của tất cả các lớp (không tính đến sự mất cân bằng)
- Weighted F1-score: F1-score trung bình có trọng số của tất cả các lớp (có tính đến sự mất cân bằng)

## Xử lý dữ liệu mất cân bằng

Chương trình hỗ trợ các phương pháp xử lý dữ liệu mất cân bằng sau:

### 1. Class Weighting

Phương pháp này gán trọng số cao hơn cho các lớp có ít mẫu trong hàm mất mát. Trọng số được tính dựa trên nghịch đảo của tần suất lớp.

```bash
python main.py --imbalance_method class_weight
```

### 2. Weighted Sampler

Phương pháp này tạo một sampler để lấy mẫu với xác suất cao hơn từ các lớp có ít mẫu trong quá trình huấn luyện.

```bash
python main.py --imbalance_method weighted_sampler
```

### 3. Oversampling

Phương pháp này tạo thêm các mẫu cho các lớp có ít mẫu bằng cách sao chép ngẫu nhiên để cân bằng phân phối lớp.

```bash
python main.py --imbalance_method oversampling
```

## Tiền xử lý văn bản

Chương trình sử dụng thư viện underthesea để chuẩn hóa văn bản tiếng Việt. Quy trình tiền xử lý bao gồm:

1. Chuẩn hóa văn bản tiếng Việt sử dụng underthesea (text_normalize) - sửa lỗi chính tả, chuyển về dạng từ gốc
2. Chuyển về chữ thường
3. Xóa khoảng trắng thừa
4. Giữ nguyên emoji và ký tự đặc biệt

Việc sử dụng underthesea giúp cải thiện chất lượng chuẩn hóa văn bản tiếng Việt. Việc tách từ được thực hiện bởi mô hình cafeBERT.

## Kiến trúc mô hình

Mô hình kết hợp đặc trưng từ mô hình CafeBERT và đặc trưng từ từ điển cảm xúc VnEmoLex. Xem thêm chi tiết trong file [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

## Phân tích kết quả

Xem hướng dẫn phân tích kết quả trong file [RESULTS_GUIDE.md](RESULTS_GUIDE.md).