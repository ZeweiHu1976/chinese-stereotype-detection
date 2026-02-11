# RBT6 Full Dataset Prediction
# 使用训练好的RBT6模型对COLD完整数据集进行预测
# 输出: fullresult.csv

import pandas as pd
import numpy as np
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm

# ==================== 环境配置 ====================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 基础目录配置 - 请根据实际情况修改
BASE_DIR = Path.cwd()

# 模型配置
MODEL_CONFIG = {
    'model_dir': 'model_output_rbt6/rbt6_cold',  # 训练后保存的RBT6模型路径
    'model_name': 'RBT6'
}


def load_full_data(csv_file_path, labelling_criteria='stereotype'):
    """
    加载完整数据集（不做划分）
    
    Args:
        csv_file_path: CSV文件路径
        labelling_criteria: 正类标签
    
    Returns:
        data: 完整数据集DataFrame
    """
    print(f"Loading full dataset from: {csv_file_path}")
    
    # 读取数据
    data = pd.read_csv(csv_file_path, usecols=['text', 'label', 'group'])
    print(f"Total data size: {len(data)}")
    
    # 保存原始标签
    data['original_label'] = data['label']
    
    # 标签二值化
    label2id = {label: (1 if label == labelling_criteria else 0) for label in data['label'].unique()}
    data['label'] = data['label'].map(label2id)
    
    print(f"Label mapping: {label2id}")
    print(f"Label distribution:\n{data['label'].value_counts()}")
    
    return data


def predict_batch(model, tokenizer, texts, device, batch_size=32):
    """
    批量预测
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        texts: 文本列表
        device: 设备
        batch_size: 批次大小
    
    Returns:
        predictions: 预测标签
        probabilities: 预测概率
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Predicting"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx].tolist()
            
            # 分词
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 预测
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 计算概率
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)


def main():
    # 配置
    CONFIG = {
        'csv_file_path': BASE_DIR / 'cold.csv',  # 数据集路径，请根据实际修改
        'labelling_criteria': 'stereotype',
        'batch_size': 32,
        'output_file': 'fullresult.csv'
    }
    
    print("="*70)
    print("RBT6 Full Dataset Prediction")
    print("="*70)
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    # 1. 加载完整数据
    print("\n[Step 1/3] Loading full dataset...")
    data = load_full_data(
        csv_file_path=CONFIG['csv_file_path'],
        labelling_criteria=CONFIG['labelling_criteria']
    )
    
    # 2. 加载模型
    print("\n[Step 2/3] Loading RBT6 model...")
    print(f"Model path: {MODEL_CONFIG['model_dir']}")
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CONFIG['model_dir'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_dir'])
    model.to(device)
    print("Model loaded successfully!")
    
    # 3. 预测
    print("\n[Step 3/3] Running predictions...")
    predictions, probabilities = predict_batch(
        model=model,
        tokenizer=tokenizer,
        texts=data['text'],
        device=device,
        batch_size=CONFIG['batch_size']
    )
    
    # 4. 构建结果DataFrame
    print("\nBuilding result DataFrame...")
    
    result_df = data.copy()
    result_df['predicted_label'] = predictions
    result_df['prob_non_stereotype'] = probabilities[:, 0]  # 类别0的概率
    result_df['prob_stereotype'] = probabilities[:, 1]      # 类别1的概率
    result_df['correct'] = (result_df['predicted_label'] == result_df['label']).astype(int)
    
    # 添加预测标签的文字描述
    result_df['predicted_label_text'] = result_df['predicted_label'].map({0: 'non-stereotype', 1: 'stereotype'})
    
    # 重新排列列顺序
    result_df = result_df[[
        'text', 
        'group',
        'original_label',
        'label',
        'predicted_label',
        'predicted_label_text',
        'prob_non_stereotype',
        'prob_stereotype',
        'correct'
    ]]
    
    # 5. 保存结果
    output_path = CONFIG['output_file']
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved to: {output_path}")
    
    # 6. 打印统计信息
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    
    total = len(result_df)
    correct = result_df['correct'].sum()
    accuracy = correct / total
    
    print(f"\nOverall Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\nPrediction Distribution:")
    print(f"  Predicted as stereotype: {(result_df['predicted_label'] == 1).sum()}")
    print(f"  Predicted as non-stereotype: {(result_df['predicted_label'] == 0).sum()}")
    
    print(f"\nActual Distribution:")
    print(f"  Actual stereotype: {(result_df['label'] == 1).sum()}")
    print(f"  Actual non-stereotype: {(result_df['label'] == 0).sum()}")
    
    # 按group统计
    print(f"\nAccuracy by Group:")
    group_stats = result_df.groupby('group').agg({
        'correct': ['sum', 'count', 'mean']
    }).round(4)
    group_stats.columns = ['Correct', 'Total', 'Accuracy']
    print(group_stats)
    
    # 错误样本分析
    errors = result_df[result_df['correct'] == 0]
    print(f"\nError Analysis:")
    print(f"  Total errors: {len(errors)}")
    
    # False Positives (预测为stereotype但实际不是)
    fp = errors[errors['predicted_label'] == 1]
    print(f"  False Positives (predicted stereotype, actual non-stereotype): {len(fp)}")
    
    # False Negatives (预测为non-stereotype但实际是)
    fn = errors[errors['predicted_label'] == 0]
    print(f"  False Negatives (predicted non-stereotype, actual stereotype): {len(fn)}")
    
    print("\n" + "="*70)
    print("✅ Full dataset prediction completed!")
    print("="*70)
    
    return result_df


if __name__ == "__main__":
    result_df = main()
