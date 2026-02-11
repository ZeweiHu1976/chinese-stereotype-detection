# Chinese BERT Models Fine-Tuning for Stereotype Detection
# 中文刻板印象检测模型微调
# 基于 HEARTS 框架复现

# ==================== [FIX 0] 依赖版本修复 ====================
# transformers v5.1.0 要求 accelerate >= 1.5.0，否则 Trainer 会崩溃
# 在 Colab 顶部 cell 运行:
#   !pip install --upgrade accelerate>=1.5.0
# 或取消下面两行的注释在脚本内自动升级:
# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "accelerate>=1.5.0"])

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, balanced_accuracy_score
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, pipeline
from codecarbon import EmissionsTracker
import torch
import transformers

# ==================== 环境配置 ====================
# 设置 Hugging Face 镜像地址（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HUGGINGFACE_TRAINER_ENABLE_PROGRESS_BAR"] = "1"

# 设置日志
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

# 基础目录配置
BASE_DIR = Path.cwd()

# ==================== 中文模型配置 ====================
# 定义中文预训练模型路径（对标原论文的英文模型）
CHINESE_MODELS = {
    'bert': 'bert-base-chinese',                          # 对应 BERT-base-uncased
    'albert': 'uer/albert-base-chinese-cluecorpussmall',  # 对应 ALBERT-V2
    'rbt6': 'hfl/rbt6',                                   # 对应 DistilBERT (6层RoBERTa)
    'macbert': 'hfl/chinese-macbert-base',                # 额外对比：MacBERT
}

# ==================== 数据加载函数 ====================
def data_loader(csv_file_path, labelling_criteria, dataset_name, sample_size=1000000, num_examples=5, test_size=0.2, random_state=42):
    """
    加载并预处理数据集
    
    Args:
        csv_file_path: CSV文件路径
        labelling_criteria: 正类标签（将被映射为1）
        dataset_name: 数据集名称
        sample_size: 采样大小（如果数据量超过此值则采样）
        num_examples: 打印的示例数量
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        train_data, test_data: 训练集和测试集DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # 读取数据
    combined_data = pd.read_csv(csv_file_path, usecols=['text', 'label', 'group'])
    print(f"Original data size: {len(combined_data)}")
    print(f"Label distribution:\n{combined_data['label'].value_counts()}")
    
    # 标签二值化
    label2id = {label: (1 if label == labelling_criteria else 0) for label in combined_data['label'].unique()}
    combined_data['label'] = combined_data['label'].map(label2id)
    print(f"Label mapping: {label2id}")
    
    # 添加数据集名称标识
    combined_data['data_name'] = dataset_name
    
    # 采样（如果需要）
    if sample_size < len(combined_data):
        sample_proportion = sample_size / len(combined_data)
        sampled_data, _ = train_test_split(
            combined_data, 
            train_size=sample_proportion, 
            stratify=combined_data['label'],
            random_state=random_state
        )
        print(f"Sampled data size: {len(sampled_data)}")
    else:
        sampled_data = combined_data
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(
        sampled_data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=sampled_data['label']
    )
    
    print(f"\nTrain data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    print(f"Train label distribution:\n{train_data['label'].value_counts()}")
    print(f"Test label distribution:\n{test_data['label'].value_counts()}")
    
    print(f"\nFirst {num_examples} examples from training data:")
    print(train_data.head(num_examples))
    
    return train_data, test_data


# ==================== 模型训练函数 ====================
def train_model(train_data, model_path, model_name, batch_size=32, epochs=6, learning_rate=2e-5, 
                model_output_base_dir='model_output', dataset_name='cold', seed=42):
    """
    微调预训练模型
    
    Args:
        train_data: 训练数据DataFrame
        model_path: 预训练模型路径
        model_name: 模型名称（用于输出目录命名）
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        model_output_base_dir: 模型输出基础目录
        dataset_name: 数据集名称
        seed: 随机种子
    
    Returns:
        model_output_dir: 模型保存路径
        training_emissions: 训练阶段碳排放量
    """
    print(f"\n{'='*60}")
    print(f"Training model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_labels = len(train_data['label'].unique())
    print(f"Number of unique labels: {num_labels}")
    
    # 启动碳排放追踪
    tracker = EmissionsTracker()
    tracker.start()
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=num_labels, 
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 分词函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding=True, 
            truncation=True, 
            max_length=512
        )
    
    # 划分训练集和验证集
    train_split, val_split = train_test_split(
        train_data, 
        test_size=0.2, 
        random_state=seed,
        stratify=train_data['label']
    )
    print(f"Training split size: {len(train_split)}")
    print(f"Validation split size: {len(val_split)}")
    
    # [FIX 1] reset_index 避免 __index_level_0__ 残留列
    tokenized_train = Dataset.from_pandas(train_split.reset_index(drop=True)).map(
        tokenize_function, batched=True
    )
    tokenized_train = tokenized_train.rename_column('label', 'labels')  # [FIX 2] 用 rename_column 替代 map

    tokenized_val = Dataset.from_pandas(val_split.reset_index(drop=True)).map(
        tokenize_function, batched=True
    )
    tokenized_val = tokenized_val.rename_column('label', 'labels')  # [FIX 2]

    # [FIX 3] 移除不需要的列，避免 Trainer 警告
    cols_to_remove = [c for c in tokenized_train.column_names if c not in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']]
    tokenized_train = tokenized_train.remove_columns(cols_to_remove)
    tokenized_val = tokenized_val.remove_columns(cols_to_remove)
    
    print(f"Sample tokenized input: {tokenized_train[0]}")
    
    # 评估指标
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return {
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "balanced_accuracy": balanced_acc
        }
    
    # 创建输出目录
    model_output_dir = os.path.join(model_output_base_dir, f"{model_name}_{dataset_name}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        eval_strategy="epoch",  # 新版transformers使用eval_strategy
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        logging_dir=os.path.join(model_output_dir, 'logs'),
        logging_steps=100,
        metric_for_best_model='f1',
        greater_is_better=True,
        seed=seed,
    )
    
    # [FIX 4] 兼容 transformers v4.x 和 v5.x
    # v5.0+ 使用 processing_class, v4.x 使用 tokenizer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )
    if int(transformers.__version__.split('.')[0]) >= 5:
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    
    # 训练
    print("Starting training...")
    trainer.train()
    
    # 保存最佳模型
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to: {model_output_dir}")
    
    # 停止碳排放追踪
    training_emissions = tracker.stop()
    print(f"Training emissions: {training_emissions:.6f} kg CO2")
    
    # 保存碳排放记录
    emissions_file = os.path.join(model_output_dir, 'training_emissions.txt')
    with open(emissions_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training emissions: {training_emissions:.6f} kg CO2\n")
    
    return model_output_dir, training_emissions


# ==================== 模型评估函数 ====================
def evaluate_model(test_data, model_output_dir, model_name, result_output_base_dir='result_output', 
                   dataset_name='cold', seed=42):
    """
    在测试集上评估模型
    
    Args:
        test_data: 测试数据DataFrame
        model_output_dir: 已训练模型的路径
        model_name: 模型名称
        result_output_base_dir: 结果输出基础目录
        dataset_name: 数据集名称
        seed: 随机种子
    
    Returns:
        df_report: 分类报告DataFrame
        eval_emissions: 评估阶段碳排放量
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"Model path: {model_output_dir}")
    print(f"Test dataset: {dataset_name}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    num_labels = len(test_data['label'].unique())
    print(f"Number of unique labels: {num_labels}")
    print(f"Test data size: {len(test_data)}")
    
    # 启动碳排放追踪（评估阶段也追踪 - 改进原论文代码）
    tracker = EmissionsTracker()
    tracker.start()
    
    # [FIX 5] 加载自己训练好的模型时不应使用 ignore_mismatched_sizes
    print("Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir, 
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
    
    # 创建结果输出目录
    result_output_dir = os.path.join(result_output_base_dir, f"{model_name}_{dataset_name}")
    os.makedirs(result_output_dir, exist_ok=True)
    
    # 自动选择设备：优先GPU（改进原论文代码）
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"Using device: {device_name}")
    
    # 创建推理pipeline
    pipe = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=device
    )
    
    # [FIX 6] 推理时必须加 truncation + batch_size，否则长文本会报错/OOM
    print("Running predictions...")
    predictions = pipe(
        test_data['text'].tolist(), 
        top_k=None, 
        truncation=True, 
        max_length=512,
        batch_size=32,
    )
    
    # 提取预测结果 (兼容新版本格式)
    def extract_prediction(pred):
        """从预测结果中提取标签和概率"""
        if isinstance(pred, list):
            # 多分类结果: [{'label': 'LABEL_0', 'score': 0.8}, {'label': 'LABEL_1', 'score': 0.2}]
            best = max(pred, key=lambda x: x['score'])
        else:
            # 单个结果: {'label': 'LABEL_1', 'score': 0.9}
            best = pred
        
        label_str = best['label']
        # 处理不同的标签格式: "LABEL_0", "LABEL_1" 或直接 "0", "1"
        if 'LABEL_' in label_str:
            label = int(label_str.split('_')[-1])
        else:
            label = int(label_str)
        return label, best['score']
    
    pred_labels = []
    pred_probs = []
    for pred in predictions:
        label, prob = extract_prediction(pred)
        pred_labels.append(label)
        pred_probs.append(prob)
    
    y_true = test_data['label'].tolist()
    
    # 停止碳排放追踪
    eval_emissions = tracker.stop()
    print(f"Evaluation emissions: {eval_emissions:.6f} kg CO2")
    
    # 保存完整结果
    results_df = pd.DataFrame({
        'text': test_data['text'].tolist(),
        'predicted_label': pred_labels,
        'predicted_probability': pred_probs,
        'actual_label': y_true,
        'group': test_data['group'].tolist(),
        'dataset_name': test_data['data_name'].tolist()
    })
    
    results_file_path = os.path.join(result_output_dir, "full_results.csv")
    results_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')
    print(f"Full results saved to: {results_file_path}")
    
    # 生成分类报告
    report = classification_report(y_true, pred_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    report_file_path = os.path.join(result_output_dir, "classification_report.csv")
    df_report.to_csv(report_file_path, encoding='utf-8-sig')
    print(f"Classification report saved to: {report_file_path}")
    
    # 打印报告
    print("\nClassification Report:")
    print(classification_report(y_true, pred_labels))
    
    # 计算额外指标
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred_labels, average='macro')
    balanced_acc = balanced_accuracy_score(y_true, pred_labels)
    
    print(f"\nMacro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # 保存碳排放记录
    emissions_file = os.path.join(result_output_dir, 'evaluation_emissions.txt')
    with open(emissions_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation emissions: {eval_emissions:.6f} kg CO2\n")
    
    return df_report, eval_emissions


# ==================== 完整实验流程函数 ====================
def run_experiment(model_key, model_path, train_data, test_data, 
                   batch_size=32, epochs=6, learning_rate=2e-5, seed=42):
    """
    运行单个模型的完整训练和评估流程
    
    Args:
        model_key: 模型键名
        model_path: 模型路径
        train_data: 训练数据
        test_data: 测试数据
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        seed: 随机种子
    
    Returns:
        results: 包含模型输出路径和评估报告的字典
    """
    print(f"\n{'#'*70}")
    print(f"# Running experiment for: {model_key}")
    print(f"# Model: {model_path}")
    print(f"{'#'*70}")
    
    # 训练模型
    model_output_dir, training_emissions = train_model(
        train_data=train_data,
        model_path=model_path,
        model_name=model_key,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        model_output_base_dir=f'model_output_{model_key}',
        dataset_name='cold',
        seed=seed
    )
    
    # 评估模型
    report, eval_emissions = evaluate_model(
        test_data=test_data,
        model_output_dir=model_output_dir,
        model_name=model_key,
        result_output_base_dir=f'result_output_{model_key}',
        dataset_name='cold',
        seed=seed
    )
    
    return {
        'model_key': model_key,
        'model_path': model_path,
        'model_output_dir': model_output_dir,
        'report': report,
        'training_emissions': training_emissions,
        'eval_emissions': eval_emissions,
        'total_emissions': training_emissions + eval_emissions
    }


# ==================== 主程序 ====================
if __name__ == "__main__":
    
    # 实验配置
    CONFIG = {
        'csv_file_path': BASE_DIR / 'cold.csv',  # 数据集路径
        'labelling_criteria': 'stereotype',       # 正类标签
        'dataset_name': 'COLD',                   # 数据集名称
        'sample_size': 1000000,                   # 采样大小
        'batch_size': 32,                         # 批次大小
        'epochs': 6,                              # 训练轮数
        'learning_rate': 2e-5,                    # 学习率
        'seed': 42,                               # 随机种子
    }
    
    # 选择要使用的中文模型（按参数量从小到大排序）
    # 顺序: albert(10M) -> rbt6(59M) -> macbert(102M) -> bert(102M)
    SELECTED_MODELS = OrderedDict([
    ('albert_chinese', CHINESE_MODELS['albert']),
    ('rbt6', CHINESE_MODELS['rbt6']),              # 替代 distilbert
    ('macbert', CHINESE_MODELS['macbert']),
    ('bert_chinese', CHINESE_MODELS['bert']),
    ])
    
    print("="*70)
    print("Chinese BERT Models Fine-Tuning for Stereotype Detection")
    print("Based on HEARTS Framework (NeurIPS 2024)")
    print("="*70)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print(f"\nSelected models (按参数量从小到大排序):")
    print(f"  {'Model Key':<20} {'HuggingFace Path':<45} {'参数量':<12} {'对应原论文'}")
    print(f"  {'-'*20} {'-'*45} {'-'*12} {'-'*15}")
    print(f"  {'albert_chinese':<20} {CHINESE_MODELS['albert']:<45} {'~10M':<12} ALBERT-V2")
    print(f"  {'macbert':<20} {CHINESE_MODELS['macbert']:<45} {'~102M':<12} 额外对比")
    print(f"  {'bert_chinese':<20} {CHINESE_MODELS['bert']:<45} {'~102M':<12} BERT")
    
    # 检测设备
    if torch.cuda.is_available():
        print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n⚠️  No GPU detected, using CPU (training will be slower)")
    
    # 加载数据
    train_data, test_data = data_loader(
        csv_file_path=CONFIG['csv_file_path'],
        labelling_criteria=CONFIG['labelling_criteria'],
        dataset_name=CONFIG['dataset_name'],
        sample_size=CONFIG['sample_size'],
        num_examples=5
    )
    
    # 存储所有实验结果
    all_results = []
    
    # 对每个模型运行实验
    for model_key, model_path in SELECTED_MODELS.items():
        try:
            result = run_experiment(
                model_key=model_key,
                model_path=model_path,
                train_data=train_data,
                test_data=test_data,
                batch_size=CONFIG['batch_size'],
                epochs=CONFIG['epochs'],
                learning_rate=CONFIG['learning_rate'],
                seed=CONFIG['seed']
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error running experiment for {model_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总结果
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    summary_data = []
    for result in all_results:
        report = result['report']
        if 'macro avg' in report.index:
            macro_f1 = report.loc['macro avg', 'f1-score']
            macro_precision = report.loc['macro avg', 'precision']
            macro_recall = report.loc['macro avg', 'recall']
        else:
            macro_f1 = macro_precision = macro_recall = None
            
        summary_data.append({
            'Model': result['model_key'],
            'Model Path': result['model_path'],
            'Macro Precision': macro_precision,
            'Macro Recall': macro_recall,
            'Macro F1': macro_f1,
            'Training Emissions (kg CO2)': result['training_emissions'],
            'Eval Emissions (kg CO2)': result['eval_emissions'],
            'Total Emissions (kg CO2)': result['total_emissions']
        })
        
        print(f"\n{result['model_key']}:")
        print(f"  Model Path: {result['model_path']}")
        print(f"  Output Dir: {result['model_output_dir']}")
        if macro_f1:
            print(f"  Macro Precision: {macro_precision:.4f}")
            print(f"  Macro Recall: {macro_recall:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Training Emissions: {result['training_emissions']:.6f} kg CO2")
        print(f"  Eval Emissions: {result['eval_emissions']:.6f} kg CO2")
        print(f"  Total Emissions: {result['total_emissions']:.6f} kg CO2")
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('experiment_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n📊 Summary saved to: experiment_summary.csv")
    
    # 打印对比表格
    print("\n" + "="*90)
    print("MODEL COMPARISON TABLE (对标原论文 Table 1)")
    print("="*90)
    print(f"\n{'Model':<18} {'Precision':<12} {'Recall':<12} {'Macro F1':<12} {'Emissions (kg CO2)':<20}")
    print(f"{'-'*18} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
    for data in summary_data:
        p = f"{data['Macro Precision']:.4f}" if data['Macro Precision'] else "N/A"
        r = f"{data['Macro Recall']:.4f}" if data['Macro Recall'] else "N/A"
        f1 = f"{data['Macro F1']:.4f}" if data['Macro F1'] else "N/A"
        emissions = f"{data['Total Emissions (kg CO2)']:.6f}"
        print(f"{data['Model']:<18} {p:<12} {r:<12} {f1:<12} {emissions:<20}")
    
    print("\n" + "="*70)
    print("✅ All experiments completed!")
    print("="*70)
    
    # 打印改进说明
    print("\n📝 相比原论文代码的改进:")
    print("  1. 评估阶段也追踪碳排放（原代码未追踪）")
    print("  2. 评估阶段优先使用GPU（原代码强制用CPU）")
    print("  3. 记录完整的碳排放数据（训练+评估）")