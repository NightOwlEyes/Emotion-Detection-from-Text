# Hướng dẫn phân tích kết quả và biểu đồ

Sau khi chạy chương trình `main.py`, nhiều kết quả và biểu đồ sẽ được tạo ra để giúp bạn đánh giá hiệu suất của mô hình. Tài liệu này sẽ hướng dẫn bạn cách hiểu và phân tích các kết quả này.

## Cấu trúc thư mục kết quả

Kết quả được lưu trong ba thư mục chính:

```
├── models/                # Chứa mô hình đã huấn luyện
├── logs/                  # Chứa nhật ký và thông tin tiền xử lý
└── results/               # Chứa kết quả đánh giá và biểu đồ
```

## 1. Thư mục models/

### best_model.pt

Đây là file lưu trạng thái của mô hình có hiệu suất tốt nhất trên tập kiểm định (dựa trên chỉ số Macro F1-score). File này được sử dụng khi bạn muốn dự đoán cảm xúc cho văn bản mới thông qua `predict.py`.

## 2. Thư mục logs/

### preprocessing_info.txt

Mô tả các bước tiền xử lý văn bản được áp dụng trong quá trình huấn luyện và dự đoán.

### preprocessed_train.csv, preprocessed_valid.csv, preprocessed_test.csv

Dữ liệu đã qua tiền xử lý, được sử dụng để huấn luyện, kiểm định và kiểm tra mô hình.

### data_distribution.txt

Thông tin chi tiết về phân phối nhãn trong các tập dữ liệu, bao gồm:
- Tổng số mẫu trong mỗi tập
- Số lượng và tỷ lệ phần trăm của mỗi loại cảm xúc trong từng tập

### vnemolex_info.txt

Thông tin về từ điển cảm xúc VnEmoLex, bao gồm:
- Tổng số từ trong từ điển
- Số lượng từ cho mỗi loại cảm xúc

### training_history.json

Lịch sử huấn luyện dưới dạng JSON, bao gồm các chỉ số sau cho mỗi epoch:
- `train_loss`: Mất mát trên tập huấn luyện
- `val_loss`: Mất mát trên tập kiểm định
- `val_accuracy`: Độ chính xác trên tập kiểm định
- `val_macro_f1`: Macro F1-score trên tập kiểm định
- `val_weighted_f1`: Weighted F1-score trên tập kiểm định

### execution_time.txt

Thời gian thực thi tổng cộng của chương trình.

## 3. Thư mục results/

### data_distribution.png

Biểu đồ cột thể hiện phân phối nhãn trong ba tập dữ liệu: huấn luyện, kiểm định và kiểm tra.

**Cách phân tích**:
- Kiểm tra xem phân phối nhãn có cân bằng không
- Xác định các lớp có ít mẫu, có thể gây khó khăn cho mô hình
- So sánh phân phối giữa các tập để đảm bảo tính nhất quán

### training_history.png

Biểu đồ thể hiện quá trình huấn luyện qua các epoch, bao gồm:

1. **Mất mát qua các epoch**: So sánh mất mát trên tập huấn luyện và tập kiểm định
   - Nếu mất mát trên tập huấn luyện giảm nhưng mất mát trên tập kiểm định tăng, có thể mô hình đang bị overfitting
   - Nếu cả hai đều giảm chậm, có thể cần tăng learning rate hoặc số epoch

2. **Độ chính xác qua các epoch**: Thể hiện độ chính xác trên tập kiểm định
   - Theo dõi xu hướng tăng/giảm để đánh giá quá trình học

3. **F1-score qua các epoch**: So sánh Macro F1 và Weighted F1 trên tập kiểm định
   - Macro F1 thấp hơn nhiều so với Weighted F1 có thể cho thấy mô hình hoạt động kém trên các lớp thiểu số

### confusion_matrix.png

Ma trận nhầm lẫn thể hiện số lượng mẫu được phân loại đúng và sai cho mỗi lớp trên tập kiểm tra.

**Cách phân tích**:
- Các giá trị trên đường chéo chính thể hiện số lượng mẫu được phân loại đúng
- Các giá trị ngoài đường chéo chính thể hiện số lượng mẫu bị phân loại sai
- Hàng thể hiện nhãn thực tế, cột thể hiện nhãn dự đoán
- Xác định các cặp lớp thường bị nhầm lẫn với nhau
- Tìm hiểu nguyên nhân và cải thiện mô hình cho các lớp có tỷ lệ nhầm lẫn cao

### evaluation_results.txt

Kết quả đánh giá cuối cùng trên tập kiểm tra, bao gồm:
- **Accuracy**: Tỷ lệ dự đoán đúng trên tổng số mẫu
- **Macro F1-score**: Trung bình của F1-score cho mỗi lớp, coi mỗi lớp có tầm quan trọng như nhau
- **Weighted F1-score**: Trung bình có trọng số của F1-score cho mỗi lớp, trọng số là số lượng mẫu của mỗi lớp

**Cách phân tích**:
- Accuracy cao cho thấy mô hình dự đoán đúng phần lớn các mẫu
- Macro F1 thấp hơn nhiều so với Weighted F1 cho thấy mô hình hoạt động kém trên các lớp thiểu số
- So sánh các chỉ số này với các mô hình khác để đánh giá hiệu suất tương đối

## Cải thiện mô hình dựa trên kết quả

Dựa trên phân tích kết quả, bạn có thể cải thiện mô hình bằng cách:

1. **Xử lý mất cân bằng dữ liệu**:
   - Oversampling các lớp thiểu số
   - Undersampling các lớp đa số
   - Sử dụng kỹ thuật tạo dữ liệu tổng hợp như SMOTE

2. **Điều chỉnh tham số mô hình**:
   - Thay đổi learning rate
   - Điều chỉnh số lượng epoch
   - Thử nghiệm các giá trị dropout khác nhau
   - Thay đổi kích thước batch

3. **Cải thiện đặc trưng**:
   - Thử nghiệm các phương pháp tiền xử lý văn bản khác nhau
   - Tích hợp thêm các nguồn thông tin hoặc từ điển khác
   - Điều chỉnh cách kết hợp đặc trưng từ CafeBERT và VnEmoLex

4. **Thay đổi kiến trúc mô hình**:
   - Thử nghiệm các mô hình ngôn ngữ nền tảng khác
   - Thay đổi cấu trúc mạng neural
   - Sử dụng các kỹ thuật như attention mechanism

5. **Phân tích lỗi**:
   - Xem xét các mẫu bị phân loại sai
   - Tìm hiểu nguyên nhân và đặc điểm chung của các lỗi
   - Điều chỉnh mô hình để giải quyết các lỗi phổ biến