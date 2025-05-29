# Nhận dạng Cảm xúc Văn bản Tiếng Việt với CafeBERT

Dự án này xây dựng một hệ thống nhận dạng cảm xúc văn bản tiếng Việt sử dụng mô hình CafeBERT, kết hợp với từ điển cảm xúc VnEmoLex để nâng cao hiệu suất phân loại.

## Cấu trúc dự án

```
├── main.py                # Mã nguồn chính của dự án
├── train.csv              # Dữ liệu huấn luyện
├── valid.csv              # Dữ liệu kiểm định
├── test.csv               # Dữ liệu kiểm tra
├── VnEmoLex.csv           # Từ điển cảm xúc tiếng Việt
├── A.PY                   # Ví dụ cách sử dụng CafeBERT
├── models/                # Thư mục lưu mô hình đã huấn luyện
├── logs/                  # Thư mục lưu nhật ký và thông tin tiền xử lý
└── results/               # Thư mục lưu kết quả đánh giá và biểu đồ
```

## Yêu cầu

```
python >= 3.6
torch
transformers
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

Cài đặt các thư viện cần thiết:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn tqdm
```

## Cách sử dụng

### Huấn luyện và đánh giá mô hình

```bash
python main.py
```

Chương trình sẽ tự động:
1. Đọc và tiền xử lý dữ liệu từ các file CSV
2. Tải mô hình CafeBERT và tích hợp với từ điển VnEmoLex
3. Huấn luyện mô hình trên tập huấn luyện
4. Đánh giá mô hình trên tập kiểm tra
5. Lưu các kết quả, biểu đồ và thông tin vào các thư mục tương ứng

### Cấu hình tham số

Các tham số chính có thể điều chỉnh trong file `main.py`:

```python
MAX_LEN = 128         # Độ dài tối đa của văn bản
BATCH_SIZE = 32       # Kích thước batch
EPOCHS = 10           # Số epoch huấn luyện
LEARNING_RATE = 2e-5  # Tốc độ học
WARMUP_STEPS = 0      # Số bước khởi động
WEIGHT_DECAY = 0.01   # Hệ số suy giảm trọng số
```

## Kết quả

Sau khi chạy chương trình, các kết quả sẽ được lưu trong các thư mục:

- **models/**: Lưu mô hình tốt nhất
- **logs/**: Lưu thông tin tiền xử lý, phân phối dữ liệu, lịch sử huấn luyện
- **results/**: Lưu kết quả đánh giá, biểu đồ mất mát, độ chính xác, F1-score và ma trận nhầm lẫn

## Đánh giá mô hình

Mô hình được đánh giá dựa trên các chỉ số:

1. **Accuracy**: Tỷ lệ dự đoán đúng trên tổng số mẫu
2. **Macro F1-Score**: Trung bình của F1-score cho mỗi lớp, coi mỗi lớp có tầm quan trọng như nhau
3. **Weighted F1-Score**: Trung bình có trọng số của F1-score cho mỗi lớp, trọng số là số lượng mẫu của mỗi lớp

## Cấu trúc mô hình

Mô hình kết hợp đặc trưng từ CafeBERT và đặc trưng từ từ điển cảm xúc VnEmoLex:

1. **CafeBERT**: Mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, cung cấp biểu diễn ngữ nghĩa phong phú
2. **VnEmoLex**: Từ điển cảm xúc tiếng Việt, cung cấp thông tin về mối liên hệ giữa từ và cảm xúc
3. **Kết hợp đặc trưng**: Đặc trưng từ CafeBERT và VnEmoLex được kết hợp để tạo ra biểu diễn phong phú hơn
4. **Phân loại**: Sử dụng mạng neural để phân loại văn bản vào 7 loại cảm xúc: Anger, Disgust, Fear, Enjoyment, Sadness, Surprise, Other

## Tác giả

Dự án được phát triển theo yêu cầu của người dùng, sử dụng mô hình CafeBERT từ UIT-NLP.