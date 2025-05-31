# Kiến trúc Mô hình Nhận dạng Cảm xúc Văn bản

Tài liệu này mô tả chi tiết về kiến trúc mô hình nhận dạng cảm xúc văn bản tiếng Việt được sử dụng trong dự án này.

## Tổng quan

Mô hình kết hợp sức mạnh của mô hình ngôn ngữ tiền huấn luyện CafeBERT với thông tin từ từ điển cảm xúc VnEmoLex để tạo ra một hệ thống phân loại cảm xúc hiệu quả cho văn bản tiếng Việt.

## Các thành phần chính

### 1. CafeBERT

CafeBERT là một mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, được phát triển bởi UIT-NLP. Mô hình này dựa trên kiến trúc XLM-RoBERTaBER và được huấn luyện trên một lượng lớn dữ liệu tiếng Việt.

Trong dự án này, chúng ta sử dụng CafeBERT để trích xuất biểu diễn ngữ nghĩa phong phú từ văn bản tiếng Việt. Đầu ra của CafeBERT là một vector đặc trưng có kích thước 768 (hoặc tùy thuộc vào cấu hình cụ thể của mô hình).

### 2. Từ điển cảm xúc VnEmoLex

VnEmoLex là một từ điển cảm xúc tiếng Việt, cung cấp thông tin về mối liên hệ giữa các từ và 6 loại cảm xúc cơ bản: Anger (Giận dữ), Disgust (Ghê tởm), Fear (Sợ hãi), Enjoyment (Vui vẻ), Sadness (Buồn bã), và Surprise (Ngạc nhiên).

Trong dự án này, chúng ta sử dụng VnEmoLex để trích xuất đặc trưng cảm xúc từ văn bản. Cụ thể, với mỗi văn bản, chúng ta:

1. Tách văn bản thành các từ
2. Đếm số lượng từ liên quan đến mỗi loại cảm xúc dựa trên từ điển VnEmoLex
3. Chuẩn hóa các giá trị đếm để tạo ra một vector đặc trưng 6 chiều

### 3. Kết hợp đặc trưng

Mô hình kết hợp đặc trưng từ CafeBERT và đặc trưng từ VnEmoLex để tạo ra một biểu diễn phong phú hơn cho văn bản. Cụ thể:

1. Đặc trưng từ CafeBERT (768 chiều) và đặc trưng từ VnEmoLex (6 chiều) được nối lại với nhau tạo thành một vector 774 chiều.
2. Vector kết hợp này được đưa qua một lớp tuyến tính để giảm kích thước xuống còn 512 chiều.
3. Áp dụng hàm kích hoạt ReLU và dropout để tránh overfitting.

### 4. Phân loại

Vector đặc trưng 512 chiều được đưa qua một lớp tuyến tính cuối cùng để phân loại văn bản vào 6 loại cảm xúc cơ bản: Anger, Disgust, Fear, Enjoyment, Sadness, và Surprise.

## Sơ đồ kiến trúc

```
Văn bản tiếng Việt
       |
       |
       v
+-------------+    +----------------+
|  CafeBERT   |    |   VnEmoLex    |
|  Encoder    |    |   Features    |
+-------------+    +----------------+
       |                  |
       |                  |
       v                  v
  [768 chiều]        [6 chiều]
       |                  |
       +--------+--------+
                |
                v
         [774 chiều]
                |
                v
     +---------------------+
     | Linear(774 -> 512) |
     +---------------------+
                |
                v
     +---------------------+
     |        ReLU        |
     +---------------------+
                |
                v
     +---------------------+
     |       Dropout      |
     +---------------------+
                |
                v
     +---------------------+
     | Linear(512 -> 7)   |
     +---------------------+
                |
                v
          Dự đoán cảm xúc
```

## Quá trình huấn luyện

Mô hình được huấn luyện bằng cách sử dụng hàm mất mát Cross-Entropy và tối ưu hóa AdamW. Quá trình huấn luyện bao gồm các bước sau:

1. Tiền xử lý dữ liệu:
   - Chuẩn hóa văn bản tiếng Việt sử dụng underthesea (text_normalize) - sửa lỗi chính tả, chuyển về dạng từ gốc
   - Chuyển về chữ thường và xóa khoảng trắng thừa
   - Trích xuất đặc trưng từ từ điển cảm xúc
   - Việc tách từ được thực hiện bởi mô hình cafeBERT
2. Huấn luyện mô hình: Cập nhật tham số của mô hình để tối thiểu hóa hàm mất mát trên tập huấn luyện.
3. Đánh giá mô hình: Đánh giá hiệu suất của mô hình trên tập kiểm định sau mỗi epoch và lưu mô hình tốt nhất.

## Đánh giá mô hình

Mô hình được đánh giá dựa trên các chỉ số:

1. **Accuracy**: Tỷ lệ dự đoán đúng trên tổng số mẫu.
2. **Macro F1-Score**: Trung bình của F1-score cho mỗi lớp, coi mỗi lớp có tầm quan trọng như nhau.
3. **Weighted F1-Score**: Trung bình có trọng số của F1-score cho mỗi lớp, trọng số là số lượng mẫu của mỗi lớp.

## Ưu điểm của kiến trúc

1. **Kết hợp thông tin ngữ nghĩa và từ điển**: Mô hình kết hợp biểu diễn ngữ nghĩa phong phú từ CafeBERT với thông tin cảm xúc từ từ điển VnEmoLex, giúp nắm bắt tốt hơn các sắc thái cảm xúc trong văn bản tiếng Việt.
2. **Tận dụng mô hình tiền huấn luyện**: Sử dụng CafeBERT, một mô hình đã được tiền huấn luyện trên dữ liệu tiếng Việt lớn, giúp mô hình có khả năng hiểu ngữ cảnh và ngữ nghĩa tốt hơn.
3. **Tích hợp kiến thức từ từ điển**: Sử dụng từ điển cảm xúc VnEmoLex giúp mô hình có thêm thông tin về mối liên hệ giữa từ và cảm xúc, đặc biệt hữu ích cho các trường hợp dữ liệu huấn luyện hạn chế.
4. **Kiến trúc linh hoạt**: Kiến trúc mô hình cho phép dễ dàng điều chỉnh và mở rộng, ví dụ như thay đổi mô hình ngôn ngữ nền tảng hoặc tích hợp thêm các nguồn thông tin khác.
