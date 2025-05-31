import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import re
import argparse
import emoji
# Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cấu hình tham số
MAX_LEN = 128

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

# Mô hình phân loại cảm xúc
class EmotionClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=6):
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

# Hàm dự đoán cảm xúc cho văn bản mới
def predict_emotion(text, model, tokenizer, emotion_dict):
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
        5: 'Surprise'
    }
    
    # Lấy xác suất cho mỗi cảm xúc
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    probs_dict = {emotion_map[i]: float(probabilities[0][i]) for i in range(6)}
    
    return emotion_map[preds.item()], probs_dict

# Hàm tương tác với người dùng
def get_user_input():
    print("\nChọn phương thức nhập dữ liệu:")
    print("1. Nhập văn bản trực tiếp")
    print("2. Nhập đường dẫn đến file văn bản")
    print("3. Thoát")
    
    choice = input("Nhập lựa chọn của bạn (1-3): ")
    
    if choice == '1':
        text = input("\nNhập văn bản cần dự đoán cảm xúc: ")
        return {'type': 'text', 'content': text}
    elif choice == '2':
        file_path = input("\nNhập đường dẫn đến file văn bản: ")
        return {'type': 'file', 'content': file_path}
    elif choice == '3':
        return {'type': 'exit'}
    else:
        print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")
        return get_user_input()

# Hàm xử lý dự đoán cho văn bản
def process_text(text, model, tokenizer, emotion_dict):
    emotion, probs = predict_emotion(text, model, tokenizer, emotion_dict)
    print(f"\nVăn bản: {text}")
    print(f"Cảm xúc dự đoán: {emotion}")
    print("\nXác suất cho mỗi cảm xúc:")
    for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {prob:.4f}")

# Hàm xử lý dự đoán cho file
def process_file(file_path, model, tokenizer, emotion_dict):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        print(f"\nĐã đọc {len(texts)} dòng từ file {file_path}")
        
        results = []
        for i, text in enumerate(texts):
            text = text.strip()
            if text:  # Bỏ qua dòng trống
                emotion, probs = predict_emotion(text, model, tokenizer, emotion_dict)
                results.append({
                    'text': text,
                    'emotion': emotion,
                    'probabilities': probs
                })
                print(f"Dòng {i+1}: {emotion}")
        
        # Lưu kết quả vào file
        output_file = file_path.rsplit('.', 1)[0] + '_predictions.csv'
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nĐã lưu kết quả dự đoán vào file {output_file}")
    
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")

# Hàm chính
def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Dự đoán cảm xúc cho văn bản tiếng Việt')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='Đường dẫn đến file mô hình đã huấn luyện')
    args = parser.parse_args()
    
    # Đọc từ điển cảm xúc
    print("Đang đọc từ điển cảm xúc...")
    emotion_dict = load_vnemolex()
    
    # Tải mô hình và tokenizer
    print("Đang tải mô hình CafeBERT...")
    tokenizer = AutoTokenizer.from_pretrained('uitnlp/CafeBERT')
    bert_model = AutoModel.from_pretrained('uitnlp/CafeBERT')
    
    # Tạo mô hình
    print("Đang khởi tạo mô hình...")
    model = EmotionClassifier(bert_model)
    model.to(device)
    
    # Tải trọng số mô hình đã huấn luyện
    print(f"Đang tải mô hình đã huấn luyện từ {args.model}...")
    model.load_state_dict(torch.load(args.model, map_location=device))
    print("Mô hình đã được tải thành công!")
    
    # Vòng lặp tương tác với người dùng
    while True:
        user_input = get_user_input()
        
        if user_input['type'] == 'exit':
            print("Tạm biệt!")
            break
        elif user_input['type'] == 'text':
            process_text(user_input['content'], model, tokenizer, emotion_dict)
        elif user_input['type'] == 'file':
            process_file(user_input['content'], model, tokenizer, emotion_dict)

# Chạy chương trình
if __name__ == "__main__":
    main()