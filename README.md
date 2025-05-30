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

Bởi Đỗ Huy Trúc trong môn Đồ án cơ sở, sử dụng mô hình CafeBERT từ UIT-NLP và từ điển cảm xúc tiếng Việt EmoLex bởi VnEmoLex và Viet WordNet.

@inproceedings{do-etal-2024-vlue,
    title = "{VLUE}: A New Benchmark and Multi-task Knowledge Transfer Learning for {V}ietnamese Natural Language Understanding",
    author = "Do, Phong  and
      Tran, Son  and
      Hoang, Phu  and
      Nguyen, Kiet  and
      Nguyen, Ngan",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.15",
    pages = "211--222",
    abstract = "The success of Natural Language Understanding (NLU) benchmarks in various languages, such as GLUE for English, CLUE for Chinese, KLUE for Korean, and IndoNLU for Indonesian, has facilitated the evaluation of new NLU models across a wide range of tasks. To establish a standardized set of benchmarks for Vietnamese NLU, we introduce the first Vietnamese Language Understanding Evaluation (VLUE) benchmark. The VLUE benchmark encompasses five datasets covering different NLU tasks, including text classification, span extraction, and natural language understanding. To provide an insightful overview of the current state of Vietnamese NLU, we then evaluate seven state-of-the-art pre-trained models, including both multilingual and Vietnamese monolingual models, on our proposed VLUE benchmark. Furthermore, we present CafeBERT, a new state-of-the-art pre-trained model that achieves superior results across all tasks in the VLUE benchmark. Our model combines the proficiency of a multilingual pre-trained model with Vietnamese linguistic knowledge. CafeBERT is developed based on the XLM-RoBERTa model, with an additional pretraining step utilizing a significant amount of Vietnamese textual data to enhance its adaptation to the Vietnamese language. For the purpose of future research, CafeBERT is made publicly available for research purposes.",
}
