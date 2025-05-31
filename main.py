import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import emoji
import json
import time
import argparse
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Import cấu hình từ config.py
from config import DATA_CONFIG, TRAINING_CONFIG, MODEL_CONFIG, PATH_CONFIG, EMOTION_MAPPING, REVERSE_EMOTION_MAPPING, VNEMOLEX_EMOTIONS

# Tạo thư mục để lưu kết quả
os.makedirs(PATH_CONFIG['results_dir'], exist_ok=True)
os.makedirs(PATH_CONFIG['model_dir'], exist_ok=True)
os.makedirs(PATH_CONFIG['logs_dir'], exist_ok=True)

# Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.strip().lower()                      # Chuyển về chữ thường
    text = re.sub(r'\s+', ' ', text)                 # Xóa khoảng trắng thừa
    text = re.sub(r'\d+', '', text)                  # Loại bỏ số
    # Giữ lại emoji
    emojis = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    # Loại ký tự đặc biệt, giữ lại chữ cái, khoảng trắng, dấu câu nhẹ
    text = re.sub(r'[^\w\s\.,!?]', '', text)
    # Ghép lại với emoji nếu cần
    return text.strip() + ' ' + emojis if emojis else text.strip()

# Lưu thông tin tiền xử lý
def save_preprocessing_info():
    with open(os.path.join(PATH_CONFIG['logs_dir'], 'preprocessing_info.txt'), 'w', encoding='utf-8') as f:
        f.write("Quy trình tiền xử lý văn bản:\n")
        f.write("1. Chuẩn hóa văn bản tiếng Việt sử dụng underthesea\n")
        f.write("2. Chuyển về chữ thường\n")
        f.write("3. Xóa khoảng trắng thừa\n")
        f.write("4. Giữ nguyên emoji và ký tự đặc biệt\n")

# Đọc dữ liệu
def load_data():
    train_df = pd.read_csv(DATA_CONFIG['train_path'])
    valid_df = pd.read_csv(DATA_CONFIG['valid_path'])
    test_df = pd.read_csv(DATA_CONFIG['test_path'])
    
    # Xóa hàng có giá trị NaN
    train_df = train_df.dropna()
    valid_df = valid_df.dropna()
    test_df = test_df.dropna()
    
    # Loại bỏ các mẫu có nhãn 'Other'
    print(f"Số lượng mẫu trong tập huấn luyện trước khi loại bỏ nhãn 'Other': {len(train_df)}")
    print(f"Số lượng mẫu trong tập kiểm định trước khi loại bỏ nhãn 'Other': {len(valid_df)}")
    print(f"Số lượng mẫu trong tập kiểm tra trước khi loại bỏ nhãn 'Other': {len(test_df)}")
    
    train_df = train_df[train_df['Emotion'] != 'Other']
    valid_df = valid_df[valid_df['Emotion'] != 'Other']
    test_df = test_df[test_df['Emotion'] != 'Other']
    
    print(f"Số lượng mẫu trong tập huấn luyện sau khi loại bỏ nhãn 'Other': {len(train_df)}")
    print(f"Số lượng mẫu trong tập kiểm định sau khi loại bỏ nhãn 'Other': {len(valid_df)}")
    print(f"Số lượng mẫu trong tập kiểm tra sau khi loại bỏ nhãn 'Other': {len(test_df)}")
    
    # Tiền xử lý văn bản
    train_df['Sentence'] = train_df['Sentence'].apply(preprocess_text)
    valid_df['Sentence'] = valid_df['Sentence'].apply(preprocess_text)
    test_df['Sentence'] = test_df['Sentence'].apply(preprocess_text)
    
    # Lưu dữ liệu đã tiền xử lý
    train_df.to_csv(os.path.join(PATH_CONFIG['logs_dir'], 'preprocessed_train.csv'), index=False)
    valid_df.to_csv(os.path.join(PATH_CONFIG['logs_dir'], 'preprocessed_valid.csv'), index=False)
    test_df.to_csv(os.path.join(PATH_CONFIG['logs_dir'], 'preprocessed_test.csv'), index=False)
    
    # Phân tích phân phối nhãn
    analyze_data_distribution(train_df, valid_df, test_df)
    
    return train_df, valid_df, test_df

# Phân tích phân phối dữ liệu
def analyze_data_distribution(train_df, valid_df, test_df):
    # Đếm số lượng mẫu cho mỗi cảm xúc
    train_counts = train_df['Emotion'].value_counts()
    valid_counts = valid_df['Emotion'].value_counts()
    test_counts = test_df['Emotion'].value_counts()
    
    # Lưu thông tin phân phối
    with open(os.path.join(PATH_CONFIG['logs_dir'], 'data_distribution.txt'), 'w', encoding='utf-8') as f:
        f.write("Phân phối dữ liệu:\n")
        f.write(f"Tổng số mẫu huấn luyện: {len(train_df)}\n")
        f.write(f"Tổng số mẫu kiểm định: {len(valid_df)}\n")
        f.write(f"Tổng số mẫu kiểm tra: {len(test_df)}\n\n")
        
        f.write("Phân phối nhãn trong tập huấn luyện:\n")
        for emotion, count in train_counts.items():
            f.write(f"{emotion}: {count} ({count/len(train_df)*100:.2f}%)\n")
        
        f.write("\nPhân phối nhãn trong tập kiểm định:\n")
        for emotion, count in valid_counts.items():
            f.write(f"{emotion}: {count} ({count/len(valid_df)*100:.2f}%)\n")
        
        f.write("\nPhân phối nhãn trong tập kiểm tra:\n")
        for emotion, count in test_counts.items():
            f.write(f"{emotion}: {count} ({count/len(test_df)*100:.2f}%)\n")
    
    # Vẽ biểu đồ phân phối
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    train_counts.plot(kind='bar', color='blue')
    plt.title('Phân phối nhãn - Tập huấn luyện')
    plt.ylabel('Số lượng mẫu')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    valid_counts.plot(kind='bar', color='green')
    plt.title('Phân phối nhãn - Tập kiểm định')
    plt.ylabel('Số lượng mẫu')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    test_counts.plot(kind='bar', color='red')
    plt.title('Phân phối nhãn - Tập kiểm tra')
    plt.ylabel('Số lượng mẫu')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_CONFIG['results_dir'], 'data_distribution.png'))
    plt.close()
    
    return train_counts

# Đọc từ điển cảm xúc VnEmoLex
def load_vnemolex():
    vnemolex_df = pd.read_csv(DATA_CONFIG['vnemolex_path'])
    
    # Tạo từ điển cảm xúc
    emotion_dict = {}
    for _, row in vnemolex_df.iterrows():
        word = row['Vietnamese']
        emotions = {}
        for emotion in VNEMOLEX_EMOTIONS:
            if emotion in row and row[emotion] == 1:
                emotions[emotion] = 1
        if emotions:
            emotion_dict[word] = emotions
    
    # Lưu thông tin từ điển
    with open(os.path.join(PATH_CONFIG['logs_dir'], 'vnemolex_info.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Tổng số từ trong từ điển VnEmoLex: {len(emotion_dict)}\n")
        emotion_counts = {emotion: 0 for emotion in VNEMOLEX_EMOTIONS}
        for word, emotions in emotion_dict.items():
            for emotion in emotions:
                emotion_counts[emotion] += 1
        f.write("Số lượng từ cho mỗi cảm xúc:\n")
        for emotion, count in emotion_counts.items():
            f.write(f"{emotion}: {count}\n")
    
    return emotion_dict

# Tạo đặc trưng từ từ điển cảm xúc
def extract_lexicon_features(text, emotion_dict):
    words = text.split()
    features = {emotion: 0 for emotion in VNEMOLEX_EMOTIONS}
    
    for word in words:
        if word in emotion_dict:
            for emotion, value in emotion_dict[word].items():
                features[emotion] += value
    
    # Chuẩn hóa đặc trưng
    total = sum(features.values())
    if total > 0:
        for emotion in features:
            features[emotion] /= total
    
    return list(features.values())

# Tạo dataset PyTorch
class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, emotion_dict):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.emotion_dict = emotion_dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.iloc[index]['Sentence']
        emotion = self.data.iloc[index]['Emotion']
        
        # Tokenize văn bản
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Trích xuất đặc trưng từ từ điển cảm xúc
        lexicon_features = extract_lexicon_features(text, self.emotion_dict)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'lexicon_features': torch.tensor(lexicon_features, dtype=torch.float),
            'label': torch.tensor(EMOTION_MAPPING[emotion], dtype=torch.long)
        }

# Mô hình phân loại cảm xúc
class EmotionClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=MODEL_CONFIG['num_classes']):
        super(EmotionClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(TRAINING_CONFIG['dropout_rate'])
        
        # Kích thước đầu ra của mô hình BERT
        self.bert_output_dim = self.bert.config.hidden_size
        
        # Số đặc trưng từ từ điển cảm xúc
        self.lexicon_features_dim = len(VNEMOLEX_EMOTIONS)
        
        # Lớp kết hợp đặc trưng BERT và đặc trưng từ điển
        self.feature_combiner = nn.Linear(self.bert_output_dim + self.lexicon_features_dim, MODEL_CONFIG['hidden_size'])
        
        # Lớp phân loại
        self.classifier = nn.Linear(MODEL_CONFIG['hidden_size'], num_classes)
        
        # Hàm kích hoạt
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, lexicon_features):
        # Đầu ra từ mô hình BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Kết hợp đặc trưng BERT và đặc trưng từ điển
        combined_features = torch.cat((pooled_output, lexicon_features), dim=1)
        combined_features = self.feature_combiner(combined_features)
        combined_features = self.relu(combined_features)
        combined_features = self.dropout(combined_features)
        
        # Phân loại
        logits = self.classifier(combined_features)
        
        return logits

# Tính trọng số cho từng lớp dựa trên tần suất xuất hiện
def calculate_class_weights(train_df):
    class_counts = train_df['Emotion'].value_counts().to_dict()
    total_samples = len(train_df)
    
    # Tính trọng số nghịch đảo tần suất lớp
    class_weights = {emotion: total_samples / (len(class_counts) * count) 
                    for emotion, count in class_counts.items()}
    
    # Chuyển đổi thành tensor
    weights = torch.FloatTensor([class_weights[emotion] for emotion in REVERSE_EMOTION_MAPPING.values()])
    
    # Lưu thông tin trọng số
    with open(os.path.join(PATH_CONFIG['logs_dir'], 'class_weights.txt'), 'w', encoding='utf-8') as f:
        f.write("Trọng số cho từng lớp cảm xúc:\n")
        for emotion, weight in class_weights.items():
            f.write(f"{emotion}: {weight:.4f}\n")
    
    return weights

# Tạo sampler cho dữ liệu mất cân bằng
def create_weighted_sampler(train_df):
    # Lấy nhãn
    train_labels = [EMOTION_MAPPING[emotion] for emotion in train_df['Emotion']]
    
    # Đếm số lượng mẫu cho mỗi lớp
    class_counts = Counter(train_labels)
    
    # Tính trọng số cho từng mẫu
    weights = [1.0 / class_counts[label] for label in train_labels]
    
    # Tạo sampler
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    return sampler

# Áp dụng oversampling cho dữ liệu mất cân bằng
def apply_oversampling(train_df):
    # Tách features và labels
    X = train_df.index.values.reshape(-1, 1)  # Sử dụng chỉ số làm đặc trưng
    y = train_df['Emotion'].values
    
    # Áp dụng RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Tạo DataFrame mới với dữ liệu đã được oversampling
    resampled_indices = X_resampled.flatten()
    oversampled_df = train_df.iloc[resampled_indices].copy()
    oversampled_df['Emotion'] = y_resampled
    
    # Lưu thông tin oversampling
    with open(os.path.join(PATH_CONFIG['logs_dir'], 'oversampling_info.txt'), 'w', encoding='utf-8') as f:
        f.write("Thông tin oversampling:\n")
        f.write(f"Số lượng mẫu trước khi oversampling: {len(train_df)}\n")
        f.write(f"Số lượng mẫu sau khi oversampling: {len(oversampled_df)}\n\n")
        
        original_counts = train_df['Emotion'].value_counts()
        resampled_counts = oversampled_df['Emotion'].value_counts()
        
        f.write("Phân phối nhãn trước khi oversampling:\n")
        for emotion, count in original_counts.items():
            f.write(f"{emotion}: {count} ({count/len(train_df)*100:.2f}%)\n")
        
        f.write("\nPhân phối nhãn sau khi oversampling:\n")
        for emotion, count in resampled_counts.items():
            f.write(f"{emotion}: {count} ({count/len(oversampled_df)*100:.2f}%)\n")
    
    # Vẽ biểu đồ so sánh phân phối trước và sau khi oversampling
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    original_counts.plot(kind='bar', color='blue')
    plt.title('Phân phối nhãn trước khi oversampling')
    plt.ylabel('Số lượng mẫu')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    resampled_counts.plot(kind='bar', color='green')
    plt.title('Phân phối nhãn sau khi oversampling')
    plt.ylabel('Số lượng mẫu')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_CONFIG['results_dir'], 'oversampling_distribution.png'))
    plt.close()
    
    return oversampled_df

# Hàm huấn luyện mô hình
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs, class_weights=None):
    # Hàm mất mát với trọng số lớp (nếu có)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Sử dụng trọng số lớp cho hàm mất mát")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Lưu lịch sử huấn luyện
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_macro_f1': [],
        'val_weighted_f1': [],
        'val_class_f1': []
    }
    
    # Lưu mô hình tốt nhất
    best_val_f1 = 0
    patience_counter = 0
    
    # Bắt đầu huấn luyện
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)
        
        # ===== Huấn luyện =====
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            # Đưa dữ liệu lên thiết bị
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lexicon_features = batch['lexicon_features'].to(device)
            labels = batch['label'].to(device)
            
            # Xóa gradient
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask, lexicon_features)
            
            # Tính mất mát
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Cập nhật tham số
            optimizer.step()
            scheduler.step()
            
            # Cập nhật thanh tiến trình
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Tính mất mát trung bình trên tập huấn luyện
        avg_train_loss = train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        # ===== Đánh giá =====
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, desc="Validation")
            for batch in progress_bar:
                # Đưa dữ liệu lên thiết bị
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                lexicon_features = batch['lexicon_features'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask, lexicon_features)
                
                # Tính mất mát
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Lấy dự đoán
                _, preds = torch.max(outputs, dim=1)
                
                # Lưu dự đoán và nhãn
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
                
                # Cập nhật thanh tiến trình
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Tính mất mát trung bình trên tập kiểm định
        avg_val_loss = val_loss / len(val_dataloader)
        history['val_loss'].append(avg_val_loss)
        
        # Tính các chỉ số đánh giá (loại bỏ Accuracy vì bộ dữ liệu mất cân bằng)
        val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
        val_weighted_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_class_f1 = f1_score(val_labels, val_preds, average=None)
        
        history['val_macro_f1'].append(val_macro_f1)
        history['val_weighted_f1'].append(val_weighted_f1)
        history['val_class_f1'].append(val_class_f1.tolist())
        
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Macro F1: {val_macro_f1:.4f}')
        print(f'Val Weighted F1: {val_weighted_f1:.4f}')
        print('\nF1-score cho từng lớp:')
        class_names = [REVERSE_EMOTION_MAPPING[i] for i in range(len(REVERSE_EMOTION_MAPPING))]
        for i, class_name in enumerate(class_names):
            print(f'{class_name}: {val_class_f1[i]:.4f}')
        
        # Lưu mô hình tốt nhất dựa trên Macro F1
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), os.path.join(PATH_CONFIG['model_dir'], 'best_model.pt'))
            print("Đã lưu mô hình tốt nhất!")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Dừng sớm nếu không cải thiện sau một số epoch
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"Dừng sớm sau {epoch+1} epochs vì không cải thiện!")
            break
    
    # Vẽ biểu đồ lịch sử huấn luyện
    plot_training_history(history)
    
    # Lưu lịch sử huấn luyện
    with open(os.path.join(PATH_CONFIG['logs_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    return history

# Vẽ biểu đồ lịch sử huấn luyện
def plot_training_history(history):
    plt.figure(figsize=(15, 10))
    
    # Vẽ biểu đồ mất mát
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Mất mát qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Vẽ biểu đồ F1-score cho từng lớp
    plt.subplot(2, 2, 2)
    class_names = [REVERSE_EMOTION_MAPPING[i] for i in range(len(REVERSE_EMOTION_MAPPING))]
    for i, class_name in enumerate(class_names):
        class_f1_values = [epoch_f1[i] for epoch_f1 in history['val_class_f1']]
        plt.plot(class_f1_values, label=class_name)
    plt.title('F1-score cho từng lớp qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    
    # Vẽ biểu đồ F1-score
    plt.subplot(2, 2, 3)
    plt.plot(history['val_macro_f1'], label='Macro F1')
    plt.plot(history['val_weighted_f1'], label='Weighted F1')
    plt.title('F1-score qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_CONFIG['results_dir'], 'training_history.png'))
    plt.close()

# Đánh giá mô hình trên tập kiểm tra
def evaluate_model(model, test_dataloader, device):
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    
    # Lưu dự đoán và nhãn
    all_preds = []
    all_labels = []
    
    # Không tính gradient
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Testing")
        for batch in progress_bar:
            # Đưa dữ liệu lên thiết bị
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lexicon_features = batch['lexicon_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, lexicon_features)
            
            # Lấy dự đoán
            _, preds = torch.max(outputs, dim=1)
            
            # Lưu dự đoán và nhãn
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Tính các chỉ số đánh giá (loại bỏ Accuracy vì bộ dữ liệu mất cân bằng)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    class_f1 = f1_score(all_labels, all_preds, average=None)
    
    # Tính ma trận nhầm lẫn
    cm = confusion_matrix(all_labels, all_preds)
    
    # Lưu kết quả đánh giá
    with open(os.path.join(PATH_CONFIG['results_dir'], 'evaluation_results.txt'), 'w') as f:
        f.write(f"Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.4f}\n\n")
        f.write("F1-score cho từng lớp:\n")
        class_names = [REVERSE_EMOTION_MAPPING[i] for i in range(len(REVERSE_EMOTION_MAPPING))]
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}: {class_f1[i]:.4f}\n")
    
    # Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(cm)
    
    return macro_f1, weighted_f1, class_f1, cm

# Vẽ ma trận nhầm lẫn
def plot_confusion_matrix(cm):
    # Danh sách nhãn cảm xúc
    labels = list(EMOTION_MAPPING.keys())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Ma trận nhầm lẫn')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_CONFIG['results_dir'], 'confusion_matrix.png'))
    plt.close()

# Hàm dự đoán cảm xúc cho văn bản mới
def predict_emotion(text, model, tokenizer, emotion_dict, device):
    # Tiền xử lý văn bản sử dụng underthesea
    text = preprocess_text(text)
    
    # Tokenize văn bản
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=DATA_CONFIG['max_len'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Trích xuất đặc trưng từ từ điển cảm xúc
    lexicon_features = extract_lexicon_features(text, emotion_dict)
    
    # Đưa dữ liệu lên thiết bị
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    lexicon_features = torch.tensor([lexicon_features], dtype=torch.float).to(device)
    
    # Dự đoán
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, lexicon_features)
        _, preds = torch.max(outputs, dim=1)
    
    return REVERSE_EMOTION_MAPPING[preds.item()]

# Hàm chính
def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình nhận dạng cảm xúc văn bản')
    parser.add_argument('--imbalance_method', type=str, default='weighted_sampler', 
                        choices=['none', 'class_weight', 'weighted_sampler', 'oversampling'],
                        help='Phương pháp xử lý dữ liệu mất cân bằng')
    args = parser.parse_args()
    
    # Bắt đầu đo thời gian
    start_time = time.time()
    
    # Lưu thông tin tiền xử lý
    save_preprocessing_info()
    
    # Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    train_df, valid_df, test_df = load_data()
    
    # Xử lý dữ liệu mất cân bằng
    class_weights = None
    sampler = None
    
    if args.imbalance_method == 'class_weight':
        print("Áp dụng trọng số lớp cho dữ liệu mất cân bằng...")
        class_weights = calculate_class_weights(train_df)
    elif args.imbalance_method == 'weighted_sampler':
        print("Áp dụng weighted sampler cho dữ liệu mất cân bằng...")
        sampler = create_weighted_sampler(train_df)
    elif args.imbalance_method == 'oversampling':
        print("Áp dụng oversampling cho dữ liệu mất cân bằng...")
        train_df = apply_oversampling(train_df)
    else:
        print("Không áp dụng phương pháp xử lý dữ liệu mất cân bằng")
    
    # Đọc từ điển cảm xúc
    print("Đang đọc từ điển cảm xúc...")
    emotion_dict = load_vnemolex()
    
    # Tải mô hình và tokenizer
    print("Đang tải mô hình CafeBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['bert_model_name'])
    bert_model = AutoModel.from_pretrained(MODEL_CONFIG['bert_model_name'])
    
    # Tạo dataset
    print("Đang tạo dataset...")
    train_dataset = EmotionDataset(train_df, tokenizer, DATA_CONFIG['max_len'], emotion_dict)
    valid_dataset = EmotionDataset(valid_df, tokenizer, DATA_CONFIG['max_len'], emotion_dict)
    test_dataset = EmotionDataset(test_df, tokenizer, DATA_CONFIG['max_len'], emotion_dict)
    
    # Tạo dataloader
    if args.imbalance_method == 'weighted_sampler' and sampler is not None:
        train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], sampler=sampler)
        print("Sử dụng weighted sampler cho train dataloader")
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=TRAINING_CONFIG['batch_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'])
    
    # Tạo mô hình
    print("Đang khởi tạo mô hình...")
    model = EmotionClassifier(bert_model)
    model.to(device)
    
    # Tạo optimizer và scheduler
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'], 
                           weight_decay=TRAINING_CONFIG['weight_decay'])
    total_steps = len(train_dataloader) * TRAINING_CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAINING_CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình...")
    history = train_model(model, train_dataloader, valid_dataloader, optimizer, scheduler, 
                         device, TRAINING_CONFIG['epochs'], class_weights)
    
    # Tải mô hình tốt nhất
    print("Đang tải mô hình tốt nhất...")
    model.load_state_dict(torch.load(os.path.join(PATH_CONFIG['model_dir'], 'best_model.pt')))
    
    # Đánh giá mô hình trên tập kiểm tra
    print("Đang đánh giá mô hình...")
    macro_f1, weighted_f1, class_f1, cm = evaluate_model(model, test_dataloader, device)
    
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print("\nF1-score cho từng lớp:")
    class_names = [REVERSE_EMOTION_MAPPING[i] for i in range(len(REVERSE_EMOTION_MAPPING))]
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_f1[i]:.4f}")
    
    # Kết thúc đo thời gian
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Thời gian thực thi: {elapsed_time:.2f} giây")
    
    # Lưu thông tin thời gian
    with open(os.path.join(PATH_CONFIG['logs_dir'], 'execution_time.txt'), 'w') as f:
        f.write(f"Thời gian thực thi: {elapsed_time:.2f} giây\n")
        f.write(f"Phương pháp xử lý dữ liệu mất cân bằng: {args.imbalance_method}\n")

# Chạy chương trình
if __name__ == "__main__":
    main()