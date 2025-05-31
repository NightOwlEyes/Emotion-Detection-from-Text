# Cấu hình tham số cho mô hình nhận dạng cảm xúc văn bản

# Cấu hình dữ liệu
DATA_CONFIG = {
    'train_path': 'train.csv',
    'valid_path': 'valid.csv',
    'test_path': 'test.csv',
    'vnemolex_path': 'VnEmoLex.csv',
    'max_len': 128,  # Độ dài tối đa của văn bản
}

# Cấu hình huấn luyện
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 2e-5,
    'warmup_steps': 0,
    'weight_decay': 0.01,
    'dropout_rate': 0.3,
    'early_stopping_patience': 3,  # Số epoch chờ đợi trước khi dừng sớm
}

# Cấu hình mô hình
MODEL_CONFIG = {
    'bert_model_name': 'uitnlp/CafeBERT',
    'hidden_size': 512,  # Kích thước lớp ẩn
    'num_classes': 6,  # Số lượng lớp cảm xúc (đã loại bỏ nhãn Other)
}

# Cấu hình đường dẫn
PATH_CONFIG = {
    'model_dir': 'models',
    'best_model_path': 'models/best_model.pt',
    'logs_dir': 'logs',
    'results_dir': 'results',
}

# Ánh xạ nhãn cảm xúc
EMOTION_MAPPING = {
    'Anger': 0,
    'Disgust': 1,
    'Fear': 2,
    'Enjoyment': 3,
    'Sadness': 4,
    'Surprise': 5
}

# Ánh xạ ngược lại từ số sang nhãn cảm xúc
REVERSE_EMOTION_MAPPING = {v: k for k, v in EMOTION_MAPPING.items()}

# Danh sách cảm xúc trong từ điển VnEmoLex
VNEMOLEX_EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Enjoyment', 'Sadness', 'Surprise']