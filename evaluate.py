import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import argparse
import os
import json

# Import các module từ dự án
from main import EmotionDataset, EmotionClassifier, preprocess_text, load_vnemolex
from config import DATA_CONFIG, MODEL_CONFIG, PATH_CONFIG, REVERSE_EMOTION_MAPPING

# Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hàm đánh giá mô hình
def evaluate_model(model, dataloader, device, output_dir='results'):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    
    # Lưu dự đoán và nhãn
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Không tính gradient
    with torch.no_grad():
        for batch in dataloader:
            # Đưa dữ liệu lên thiết bị
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lexicon_features = batch['lexicon_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, lexicon_features)
            
            # Lấy dự đoán và xác suất
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            # Lưu dự đoán, xác suất và nhãn
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    # Tính các chỉ số đánh giá (loại bỏ Accuracy vì bộ dữ liệu mất cân bằng)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    # Tính thêm các chỉ số đánh giá khác
    class_f1 = f1_score(all_labels, all_preds, average=None)
    
    # Tính ma trận nhầm lẫn
    cm = confusion_matrix(all_labels, all_preds)
    
    # Tạo báo cáo phân loại
    class_names = [REVERSE_EMOTION_MAPPING[i] for i in range(len(REVERSE_EMOTION_MAPPING))]
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Lưu kết quả đánh giá
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.4f}\n\n")
        f.write("F1-score cho từng lớp:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}: {class_f1[i]:.4f}\n")
        f.write("\nBáo cáo phân loại chi tiết:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Lưu báo cáo phân loại dưới dạng JSON
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Vẽ biểu đồ ROC và Precision-Recall nếu có đủ dữ liệu
    if len(all_probs) > 0:
        plot_roc_curves(all_labels, all_probs, class_names, os.path.join(output_dir, 'roc_curves.png'))
        plot_precision_recall_curves(all_labels, all_probs, class_names, os.path.join(output_dir, 'precision_recall_curves.png'))
    
    return macro_f1, weighted_f1, class_f1, cm, report

# Vẽ ma trận nhầm lẫn
def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Ma trận nhầm lẫn')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Vẽ biểu đồ ROC
def plot_roc_curves(labels, probs, class_names, output_path):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Chuyển đổi nhãn thành dạng one-hot
    n_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))
    
    # Tính toán đường cong ROC và AUC cho mỗi lớp
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], [p[i] for p in probs])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Vẽ biểu đồ ROC
    plt.figure(figsize=(12, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Đường cong ROC đa lớp')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Vẽ biểu đồ Precision-Recall
def plot_precision_recall_curves(labels, probs, class_names, output_path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    
    # Chuyển đổi nhãn thành dạng one-hot
    n_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))
    
    # Tính toán đường cong Precision-Recall và AP cho mỗi lớp
    precision = {}
    recall = {}
    avg_precision = {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels_bin[:, i], [p[i] for p in probs])
        avg_precision[i] = average_precision_score(labels_bin[:, i], [p[i] for p in probs])
    
    # Vẽ biểu đồ Precision-Recall
    plt.figure(figsize=(12, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Đường cong Precision-Recall đa lớp')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Phân tích lỗi
def analyze_errors(model, dataloader, device, output_dir='results'):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    
    # Lưu dự đoán, nhãn và văn bản
    all_preds = []
    all_labels = []
    all_texts = []
    all_probs = []
    
    # Không tính gradient
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Đưa dữ liệu lên thiết bị
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lexicon_features = batch['lexicon_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, lexicon_features)
            
            # Lấy dự đoán và xác suất
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            # Lưu dự đoán, xác suất, nhãn và văn bản
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            
            # Lấy văn bản từ dataloader
            texts = dataloader.dataset.data.iloc[batch_idx * dataloader.batch_size:(batch_idx + 1) * dataloader.batch_size]['Sentence'].tolist()
            all_texts.extend(texts[:len(preds)])  # Đảm bảo số lượng văn bản khớp với số lượng dự đoán
    
    # Tạo DataFrame chứa kết quả
    results_df = pd.DataFrame({
        'text': all_texts,
        'true_label': [REVERSE_EMOTION_MAPPING[label] for label in all_labels],
        'predicted_label': [REVERSE_EMOTION_MAPPING[pred] for pred in all_preds],
        'correct': [pred == label for pred, label in zip(all_preds, all_labels)]
    })
    
    # Thêm xác suất cho mỗi lớp
    for i, emotion in REVERSE_EMOTION_MAPPING.items():
        results_df[f'prob_{emotion}'] = [probs[i] for probs in all_probs]
    
    # Lưu tất cả kết quả
    results_df.to_csv(os.path.join(output_dir, 'all_predictions.csv'), index=False)
    
    # Lọc ra các dự đoán sai
    errors_df = results_df[~results_df['correct']]
    errors_df.to_csv(os.path.join(output_dir, 'error_predictions.csv'), index=False)
    
    # Phân tích lỗi theo loại cảm xúc
    error_analysis = pd.DataFrame(columns=['true_label', 'predicted_label', 'count'])
    for true_label in REVERSE_EMOTION_MAPPING.values():
        for pred_label in REVERSE_EMOTION_MAPPING.values():
            if true_label != pred_label:
                count = len(errors_df[(errors_df['true_label'] == true_label) & (errors_df['predicted_label'] == pred_label)])
                if count > 0:
                    error_analysis = error_analysis.append({
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'count': count
                    }, ignore_index=True)
    
    # Sắp xếp theo số lượng lỗi giảm dần
    error_analysis = error_analysis.sort_values('count', ascending=False)
    error_analysis.to_csv(os.path.join(output_dir, 'error_analysis.csv'), index=False)
    
    # Vẽ biểu đồ phân tích lỗi
    plt.figure(figsize=(12, 8))
    sns.barplot(x='true_label', y='count', hue='predicted_label', data=error_analysis)
    plt.title('Phân tích lỗi theo loại cảm xúc')
    plt.xlabel('Nhãn thực tế')
    plt.ylabel('Số lượng lỗi')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'))
    plt.close()
    
    return results_df, errors_df, error_analysis

# Hàm chính
def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Đánh giá mô hình nhận dạng cảm xúc văn bản')
    parser.add_argument('--model', type=str, default=PATH_CONFIG['best_model_path'], help='Đường dẫn đến file mô hình đã huấn luyện')
    parser.add_argument('--data', type=str, default=DATA_CONFIG['test_path'], help='Đường dẫn đến file dữ liệu kiểm tra')
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    parser.add_argument('--output_dir', type=str, default='results', help='Thư mục lưu kết quả đánh giá')
    parser.add_argument('--analyze_errors', action='store_true', help='Phân tích lỗi')
    args = parser.parse_args()
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Đọc dữ liệu kiểm tra
    print(f"Đang đọc dữ liệu từ {args.data}...")
    test_df = pd.read_csv(args.data)
    test_df = test_df.dropna()  # Xóa hàng có giá trị NaN
    
    # Loại bỏ các mẫu có nhãn 'Other'
    print(f"Số lượng mẫu trong tập kiểm tra trước khi loại bỏ nhãn 'Other': {len(test_df)}")
    test_df = test_df[test_df['Emotion'] != 'Other']
    print(f"Số lượng mẫu trong tập kiểm tra sau khi loại bỏ nhãn 'Other': {len(test_df)}")
    
    test_df['Sentence'] = test_df['Sentence'].apply(preprocess_text)  # Tiền xử lý văn bản
    
    # Đọc từ điển cảm xúc
    print("Đang đọc từ điển cảm xúc...")
    emotion_dict = load_vnemolex()
    
    # Tải mô hình và tokenizer
    print("Đang tải mô hình CafeBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['bert_model_name'])
    bert_model = AutoModel.from_pretrained(MODEL_CONFIG['bert_model_name'])
    
    # Tạo dataset
    print("Đang tạo dataset...")
    test_dataset = EmotionDataset(test_df, tokenizer, DATA_CONFIG['max_len'], emotion_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Tạo mô hình
    print("Đang khởi tạo mô hình...")
    model = EmotionClassifier(bert_model, num_classes=6)
    model.to(device)
    
    # Tải trọng số mô hình đã huấn luyện
    print(f"Đang tải mô hình đã huấn luyện từ {args.model}...")
    model.load_state_dict(torch.load(args.model, map_location=device))
    
    # Đánh giá mô hình
    print("Đang đánh giá mô hình...")
    macro_f1, weighted_f1, class_f1, cm, report = evaluate_model(model, test_dataloader, device, args.output_dir)
    
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print("\nF1-score cho từng lớp:")
    class_names = [REVERSE_EMOTION_MAPPING[i] for i in range(len(REVERSE_EMOTION_MAPPING))]
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_f1[i]:.4f}")
    
    # Phân tích lỗi nếu được yêu cầu
    if args.analyze_errors:
        print("Đang phân tích lỗi...")
        results_df, errors_df, error_analysis = analyze_errors(model, test_dataloader, device, args.output_dir)
        print(f"Số lượng dự đoán đúng: {len(results_df[results_df['correct']])}")
        print(f"Số lượng dự đoán sai: {len(errors_df)}")
        print("\nTop 5 loại lỗi phổ biến nhất:")
        print(error_analysis.head(5))

# Chạy chương trình
if __name__ == "__main__":
    main()