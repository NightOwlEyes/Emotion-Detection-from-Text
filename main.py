import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import json
import time
from tqdm import tqdm

# Tạo thư mục để lưu kết quả
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# Cấu hình tham số
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Chuẩn hóa văn bản
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    text = text.strip().lower()  # Chuyển về chữ thường
    
    # Lưu lại các emoji và ký tự đặc biệt
    # Có thể mở rộng thêm các quy tắc tiền xử lý khác
    
    return text

# Lưu thông tin tiền xử lý
with open('logs/preprocessing_info.txt', 'w', encoding='utf-8') as f:
    f.write("Quy trình tiền xử lý văn bản:\n")
    f.write("1. Xóa khoảng trắng thừa\n")
    f.write("2. Chuyển về chữ thường\n")
    f.write("3. Giữ nguyên emoji và ký tự đặc biệt\n")

# Đọc dữ liệu
def load_data():
    train_df = pd.read_csv('train.csv')
    valid_df = pd.read_csv('valid.csv')
    test_df = pd.read_csv('test.csv')
    
    # Xóa hàng có giá trị NaN
    train_df = train_df.dropna()
    valid_df = valid_df.dropna()
    test_df = test_df.dropna()
    
    # Tiền xử lý văn bản
    train_df['Sentence'] = train_df['Sentence'].apply(preprocess_text)
    valid_df['Sentence'] = valid_df['Sentence'].apply(preprocess_text)
    test_df['Sentence'] = test_df['Sentence'].apply(preprocess_text)
    
    # Lưu dữ liệu đã tiền xử lý
    train_df.to_csv('logs/preprocessed_train.csv', index=False)
    valid_df.to_csv('logs/preprocessed_valid.csv', index=False)
    test_df.to_csv('logs/preprocessed_test.csv', index=False)
    
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
    with open('logs/data_distribution.txt', 'w', encoding='utf-8') as f:
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
    plt.savefig('results/data_distribution.png')
    plt.close()

# Đọc từ điển cảm xúc VnEmoLex
def load_vnemolex():
    vnemolex_df = pd.read_csv('VnEmoLex.csv')
    
    # Tạo từ điển cảm xúc
    emotion_dict = {}
    for _, row in vnemolex_df.iterrows():
        word = row['Vietnamese']
        emotions = {}
        for emotion in ['Anger', 'Disgust', 'Fear', 'Enjoyment', 'Sadness', 'Surprise']:
            if emotion in row and row[emotion] == 1:
                emotions[emotion] = 1
        if emotions:
            emotion_dict[word] = emotions
    
    # Lưu thông tin từ điển
    with open('logs/vnemolex_info.txt', 'w', encoding='utf-8') as f:
        f.write(f"Tổng số từ trong từ điển VnEmoLex: {len(emotion_dict)}\n")
        emotion_counts = {emotion: 0 for emotion in ['Anger', 'Disgust', 'Fear', 'Enjoyment', 'Sadness', 'Surprise']}
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
    features = {emotion: 0 for emotion in ['Anger', 'Disgust', 'Fear', 'Enjoyment', 'Sadness', 'Surprise']}
    
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
        
        # Ánh xạ nhãn cảm xúc sang số
        self.emotion_map = {
            'Anger': 0,
            'Disgust': 1,
            'Fear': 2,
            'Enjoyment': 3,
            'Sadness': 4,
            'Surprise': 5,
            'Other': 6
        }
    
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
            'label': torch.tensor(self.emotion_map[emotion], dtype=torch.long)
        }

# Mô hình phân loại cảm xúc
class EmotionClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=7):
        super(EmotionClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        
        # Kích thước đầu ra của mô hình BERT
        self.bert_output_dim = self.bert.config.hidden_size
        
        # Số đặc trưng từ từ điển cảm xúc
        self.lexicon_features_dim = 6
        
        # Lớp kết hợp đặc trưng BERT và đặc trưng từ điển
        self.feature_combiner = nn.Linear(self.bert_output_dim + self.lexicon_features_dim, 512)
        
        # Lớp phân loại
        self.classifier = nn.Linear(512, num_classes)
        
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

# Hàm huấn luyện mô hình
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs):
    # Hàm mất mát
    criterion = nn.CrossEntropyLoss()
    
    # Lưu lịch sử huấn luyện
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_macro_f1': [],
        'val_weighted_f1': []
    }
    
    # Lưu mô hình tốt nhất
    best_val_f1 = 0
    
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
        
        # Tính các chỉ số đánh giá
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
        val_weighted_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        history['val_accuracy'].append(val_accuracy)
        history['val_macro_f1'].append(val_macro_f1)
        history['val_weighted_f1'].append(val_weighted_f1)
        
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}')
        print(f'Val Macro F1: {val_macro_f1:.4f}')
        print(f'Val Weighted F1: {val_weighted_f1:.4f}')
        
        # Lưu mô hình tốt nhất
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), 'models/best_model.pt')
            print("Đã lưu mô hình tốt nhất!")
    
    # Vẽ biểu đồ lịch sử huấn luyện
    plot_training_history(history)
    
    # Lưu lịch sử huấn luyện
    with open('logs/training_history.json', 'w') as f:
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
    
    # Vẽ biểu đồ độ chính xác
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.title('Độ chính xác qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
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
    plt.savefig('results/training_history.png')
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
    
    # Tính các chỉ số đánh giá
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Tính ma trận nhầm lẫn
    cm = confusion_matrix(all_labels, all_preds)
    
    # Lưu kết quả đánh giá
    with open('results/evaluation_results.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.4f}\n")
    
    # Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(cm)
    
    return accuracy, macro_f1, weighted_f1, cm

# Vẽ ma trận nhầm lẫn
def plot_confusion_matrix(cm):
    # Danh sách nhãn cảm xúc
    labels = ['Anger', 'Disgust', 'Fear', 'Enjoyment', 'Sadness', 'Surprise', 'Other']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Ma trận nhầm lẫn')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()

# Hàm dự đoán cảm xúc cho văn bản mới
def predict_emotion(text, model, tokenizer, emotion_dict, device):
    # Tiền xử lý văn bản
    text = preprocess_text(text)
    
    # Tokenize văn bản
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
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
    
    # Ánh xạ số sang nhãn cảm xúc
    emotion_map = {
        0: 'Anger',
        1: 'Disgust',
        2: 'Fear',
        3: 'Enjoyment',
        4: 'Sadness',
        5: 'Surprise',
        6: 'Other'
    }
    
    return emotion_map[preds.item()]

# Hàm chính
def main():
    # Bắt đầu đo thời gian
    start_time = time.time()
    
    # Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    train_df, valid_df, test_df = load_data()
    
    # Đọc từ điển cảm xúc
    print("Đang đọc từ điển cảm xúc...")
    emotion_dict = load_vnemolex()
    
    # Tải mô hình và tokenizer
    print("Đang tải mô hình CafeBERT...")
    tokenizer = AutoTokenizer.from_pretrained('uitnlp/CafeBERT')
    bert_model = AutoModel.from_pretrained('uitnlp/CafeBERT')
    
    # Tạo dataset
    print("Đang tạo dataset...")
    train_dataset = EmotionDataset(train_df, tokenizer, MAX_LEN, emotion_dict)
    valid_dataset = EmotionDataset(valid_df, tokenizer, MAX_LEN, emotion_dict)
    test_dataset = EmotionDataset(test_df, tokenizer, MAX_LEN, emotion_dict)
    
    # Tạo dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Tạo mô hình
    print("Đang khởi tạo mô hình...")
    model = EmotionClassifier(bert_model)
    model.to(device)
    
    # Tạo optimizer và scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình...")
    history = train_model(model, train_dataloader, valid_dataloader, optimizer, scheduler, device, EPOCHS)
    
    # Tải mô hình tốt nhất
    print("Đang tải mô hình tốt nhất...")
    model.load_state_dict(torch.load('models/best_model.pt'))
    
    # Đánh giá mô hình trên tập kiểm tra
    print("Đang đánh giá mô hình...")
    accuracy, macro_f1, weighted_f1, cm = evaluate_model(model, test_dataloader, device)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    
    # Kết thúc đo thời gian
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Thời gian thực thi: {elapsed_time:.2f} giây")
    
    # Lưu thông tin thời gian
    with open('logs/execution_time.txt', 'w') as f:
        f.write(f"Thời gian thực thi: {elapsed_time:.2f} giây\n")

# Chạy chương trình
if __name__ == "__main__":
    main()