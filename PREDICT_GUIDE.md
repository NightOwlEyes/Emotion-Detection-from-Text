# Hướng dẫn sử dụng công cụ dự đoán cảm xúc văn bản

File `predict.py` cung cấp một công cụ dòng lệnh để dự đoán cảm xúc cho văn bản tiếng Việt sử dụng mô hình đã được huấn luyện.

## Chuẩn bị

1. Đảm bảo bạn đã huấn luyện mô hình bằng cách chạy `main.py` hoặc đã có sẵn file mô hình đã huấn luyện.
2. Mô hình đã huấn luyện nên được lưu trong thư mục `models/` với tên `best_model.pt`.

## Cách sử dụng

### Dự đoán cảm xúc cho một văn bản

```bash
python predict.py --text "Tôi rất vui khi gặp lại bạn sau nhiều năm."
```

### Dự đoán cảm xúc cho nhiều văn bản từ file

```bash
python predict.py --file "path/to/texts.txt"
```

File văn bản nên có mỗi câu trên một dòng.

### Sử dụng mô hình từ đường dẫn khác

```bash
python predict.py --text "Tôi rất buồn khi nghe tin này." --model "path/to/custom_model.pt"
```

## Kết quả

### Khi dự đoán cho một văn bản

Kết quả sẽ được hiển thị trực tiếp trên màn hình, bao gồm:
- Văn bản gốc
- Cảm xúc được dự đoán
- Xác suất cho mỗi loại cảm xúc

Ví dụ:

```
Văn bản: Tôi rất vui khi gặp lại bạn sau nhiều năm.
Cảm xúc dự đoán: Enjoyment

Xác suất cho mỗi cảm xúc:
Enjoyment: 0.8523
Surprise: 0.0854
Other: 0.0321
Sadness: 0.0156
Fear: 0.0082
Anger: 0.0042
Disgust: 0.0022
```

### Khi dự đoán cho nhiều văn bản từ file

Kết quả sẽ được lưu vào một file CSV với tên `[tên_file_gốc]_predictions.csv`, bao gồm các cột:
- `text`: Văn bản gốc
- `emotion`: Cảm xúc được dự đoán
- `probabilities`: Xác suất cho mỗi loại cảm xúc

## Các loại cảm xúc

Mô hình có thể dự đoán 7 loại cảm xúc:
1. `Anger` (Giận dữ)
2. `Disgust` (Ghê tởm)
3. `Fear` (Sợ hãi)
4. `Enjoyment` (Vui vẻ)
5. `Sadness` (Buồn bã)
6. `Surprise` (Ngạc nhiên)
7. `Other` (Khác)

## Lưu ý

- Mô hình hoạt động tốt nhất với văn bản tiếng Việt có độ dài vừa phải (dưới 128 từ).
- Kết quả dự đoán có thể bị ảnh hưởng bởi chất lượng của mô hình đã huấn luyện.
- Đối với văn bản dài, mô hình sẽ chỉ xử lý 128 token đầu tiên.