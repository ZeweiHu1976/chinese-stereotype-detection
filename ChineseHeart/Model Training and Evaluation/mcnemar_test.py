# McNemar's Test for Comparing ALBERT vs RBT6
# 统计显著性检验：比较两个模型的预测差异
# 基于 HEARTS 框架复现

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.stats import chi2
import torch
from tqdm import tqdm

# ==================== 环境配置 ====================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 基础目录配置 - 请根据实际情况修改
BASE_DIR = Path.cwd()

# 模型配置
MODELS_TO_COMPARE = {
    'albert_chinese': {
        'model_dir': 'model_output_albert_chinese/albert_chinese_cold',  # 训练后保存的模型路径
        'original_path': 'uer/albert-base-chinese-cluecorpussmall'
    },
    'rbt6': {
        'model_dir': 'model_output_rbt6/rbt6_cold',  # 训练后保存的模型路径
        'original_path': 'hfl/rbt6'
    }
}


def load_test_data(csv_file_path, labelling_criteria='stereotype', test_size=0.2, random_state=42):
    """
    加载测试数据（使用与训练相同的随机种子确保一致性）
    """
    print(f"Loading data from: {csv_file_path}")
    
    # 读取数据
    combined_data = pd.read_csv(csv_file_path, usecols=['text', 'label', 'group'])
    print(f"Total data size: {len(combined_data)}")
    
    # 标签二值化
    label2id = {label: (1 if label == labelling_criteria else 0) for label in combined_data['label'].unique()}
    combined_data['label'] = combined_data['label'].map(label2id)
    
    # 划分训练集和测试集（使用相同的random_state）
    train_data, test_data = train_test_split(
        combined_data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=combined_data['label']
    )
    
    print(f"Test data size: {len(test_data)}")
    print(f"Test label distribution:\n{test_data['label'].value_counts()}")
    
    return test_data.reset_index(drop=True)


def get_predictions(model_dir, test_texts, batch_size=32):
    """
    使用保存的模型对测试集进行预测
    
    Returns:
        predictions: 逐样本预测结果 (numpy array)
    """
    print(f"\nLoading model from: {model_dir}")
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    all_predictions = []
    
    # 分批预测
    num_batches = (len(test_texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Predicting"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_texts))
            batch_texts = test_texts[start_idx:end_idx].tolist()
            
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
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
    
    return np.array(all_predictions)


def mcnemar_test(y_true, pred_a, pred_b, model_a_name='Model A', model_b_name='Model B'):
    """
    执行McNemar's Test
    
    比较两个分类器的预测差异是否具有统计显著性
    
    Args:
        y_true: 真实标签
        pred_a: 模型A的预测
        pred_b: 模型B的预测
        model_a_name: 模型A名称
        model_b_name: 模型B名称
    
    Returns:
        dict: 包含检验结果的字典
    """
    # 计算正确/错误
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    
    # 构建2x2列联表
    # b: A正确, B错误
    # c: A错误, B正确
    b = np.sum(correct_a & ~correct_b)  # A对B错
    c = np.sum(~correct_a & correct_b)  # A错B对
    
    # 同时对或同时错的数量（用于完整报告）
    a = np.sum(correct_a & correct_b)   # 都对
    d = np.sum(~correct_a & ~correct_b) # 都错
    
    print("\n" + "="*60)
    print("McNemar's Test Results")
    print("="*60)
    
    print(f"\n2x2 Contingency Table:")
    print(f"{'':20} {model_b_name + ' Correct':>20} {model_b_name + ' Wrong':>20}")
    print(f"{model_a_name + ' Correct':20} {a:>20} {b:>20}")
    print(f"{model_a_name + ' Wrong':20} {c:>20} {d:>20}")
    
    print(f"\nKey values for McNemar's test:")
    print(f"  b ({model_a_name} correct, {model_b_name} wrong): {b}")
    print(f"  c ({model_a_name} wrong, {model_b_name} correct): {c}")
    
    # McNemar's test with continuity correction
    if b + c == 0:
        print("\n⚠️  No discordant pairs found. Models have identical predictions on misclassified samples.")
        return {
            'b': b, 'c': c,
            'chi2': None,
            'p_value': None,
            'significant': False,
            'message': 'No discordant pairs'
        }
    
    # 标准McNemar检验（带连续性校正）
    chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    # 不带连续性校正的版本（用于对比）
    chi2_no_correction = ((b - c) ** 2) / (b + c)
    p_value_no_correction = 1 - chi2.cdf(chi2_no_correction, df=1)
    
    print(f"\nMcNemar's Chi-squared (with continuity correction): {chi2_stat:.4f}")
    print(f"McNemar's Chi-squared (without correction): {chi2_no_correction:.4f}")
    print(f"p-value (with correction): {p_value:.6f}")
    print(f"p-value (without correction): {p_value_no_correction:.6f}")
    
    # 判断显著性
    alpha_levels = [0.05, 0.01, 0.001]
    print(f"\nSignificance levels:")
    for alpha in alpha_levels:
        significant = p_value < alpha
        symbol = "✓" if significant else "✗"
        print(f"  α = {alpha}: {symbol} {'Significant' if significant else 'Not significant'}")
    
    # 效应方向
    if b > c:
        better_model = model_b_name
        worse_model = model_a_name
    elif c > b:
        better_model = model_a_name
        worse_model = model_b_name
    else:
        better_model = "Neither (equal performance)"
        worse_model = None
    
    print(f"\nEffect direction:")
    print(f"  {better_model} performs better")
    print(f"  Advantage: {abs(b-c)} more correct predictions on discordant samples")
    
    # 计算准确率差异
    acc_a = np.mean(correct_a)
    acc_b = np.mean(correct_b)
    print(f"\nAccuracy comparison:")
    print(f"  {model_a_name}: {acc_a:.4f} ({np.sum(correct_a)}/{len(y_true)})")
    print(f"  {model_b_name}: {acc_b:.4f} ({np.sum(correct_b)}/{len(y_true)})")
    print(f"  Difference: {abs(acc_a - acc_b):.4f}")
    
    return {
        'a': a, 'b': b, 'c': c, 'd': d,
        'chi2': chi2_stat,
        'chi2_no_correction': chi2_no_correction,
        'p_value': p_value,
        'p_value_no_correction': p_value_no_correction,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
        'significant_0001': p_value < 0.001,
        'better_model': better_model,
        'accuracy_a': acc_a,
        'accuracy_b': acc_b,
        'acc_difference': abs(acc_a - acc_b)
    }


def save_predictions(test_data, pred_albert, pred_rbt6, output_path='predictions_comparison.csv'):
    """
    保存逐样本预测结果用于进一步分析
    """
    results_df = test_data.copy()
    results_df['pred_albert'] = pred_albert
    results_df['pred_rbt6'] = pred_rbt6
    results_df['albert_correct'] = (pred_albert == results_df['label']).astype(int)
    results_df['rbt6_correct'] = (pred_rbt6 == results_df['label']).astype(int)
    results_df['both_correct'] = ((results_df['albert_correct'] == 1) & (results_df['rbt6_correct'] == 1)).astype(int)
    results_df['both_wrong'] = ((results_df['albert_correct'] == 0) & (results_df['rbt6_correct'] == 0)).astype(int)
    results_df['albert_only_correct'] = ((results_df['albert_correct'] == 1) & (results_df['rbt6_correct'] == 0)).astype(int)
    results_df['rbt6_only_correct'] = ((results_df['albert_correct'] == 0) & (results_df['rbt6_correct'] == 1)).astype(int)
    
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nPredictions saved to: {output_path}")
    
    return results_df


def analyze_disagreements(results_df, num_examples=10):
    """
    分析两个模型预测不一致的样本
    """
    print("\n" + "="*60)
    print("Disagreement Analysis")
    print("="*60)
    
    # ALBERT对RBT6错的样本
    albert_only = results_df[results_df['albert_only_correct'] == 1]
    print(f"\n📌 Samples where ALBERT is correct but RBT6 is wrong ({len(albert_only)} samples):")
    if len(albert_only) > 0:
        for idx, row in albert_only.head(num_examples).iterrows():
            print(f"  Text: {row['text'][:80]}...")
            print(f"  True: {row['label']}, ALBERT: {row['pred_albert']}, RBT6: {row['pred_rbt6']}")
            print()
    
    # RBT6对ALBERT错的样本
    rbt6_only = results_df[results_df['rbt6_only_correct'] == 1]
    print(f"\n📌 Samples where RBT6 is correct but ALBERT is wrong ({len(rbt6_only)} samples):")
    if len(rbt6_only) > 0:
        for idx, row in rbt6_only.head(num_examples).iterrows():
            print(f"  Text: {row['text'][:80]}...")
            print(f"  True: {row['label']}, ALBERT: {row['pred_albert']}, RBT6: {row['pred_rbt6']}")
            print()
    
    # 按group分析
    print("\n📊 Disagreement by Group:")
    group_stats = results_df.groupby('group').agg({
        'albert_correct': 'mean',
        'rbt6_correct': 'mean',
        'albert_only_correct': 'sum',
        'rbt6_only_correct': 'sum'
    }).round(4)
    group_stats.columns = ['ALBERT Acc', 'RBT6 Acc', 'ALBERT Only Correct', 'RBT6 Only Correct']
    print(group_stats)
    
    return albert_only, rbt6_only


# ==================== 主程序 ====================
if __name__ == "__main__":
    
    # 配置
    CONFIG = {
        'csv_file_path': BASE_DIR / 'cold.csv',  # 数据集路径，请根据实际修改
        'labelling_criteria': 'stereotype',
        'test_size': 0.2,
        'random_state': 42,
        'batch_size': 32
    }
    
    print("="*70)
    print("McNemar's Test: ALBERT vs RBT6")
    print("Statistical Significance Testing for Model Comparison")
    print("="*70)
    
    # 检测设备
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    # 1. 加载测试数据
    print("\n[Step 1/4] Loading test data...")
    test_data = load_test_data(
        csv_file_path=CONFIG['csv_file_path'],
        labelling_criteria=CONFIG['labelling_criteria'],
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    y_true = test_data['label'].values
    test_texts = test_data['text']
    
    # 2. 获取ALBERT预测
    print("\n[Step 2/4] Getting ALBERT predictions...")
    pred_albert = get_predictions(
        model_dir=MODELS_TO_COMPARE['albert_chinese']['model_dir'],
        test_texts=test_texts,
        batch_size=CONFIG['batch_size']
    )
    
    # 3. 获取RBT6预测
    print("\n[Step 3/4] Getting RBT6 predictions...")
    pred_rbt6 = get_predictions(
        model_dir=MODELS_TO_COMPARE['rbt6']['model_dir'],
        test_texts=test_texts,
        batch_size=CONFIG['batch_size']
    )
    
    # 4. 执行McNemar's Test
    print("\n[Step 4/4] Running McNemar's Test...")
    results = mcnemar_test(
        y_true=y_true,
        pred_a=pred_albert,
        pred_b=pred_rbt6,
        model_a_name='ALBERT',
        model_b_name='RBT6'
    )
    
    # 5. 保存预测结果
    results_df = save_predictions(test_data, pred_albert, pred_rbt6)
    
    # 6. 分析预测不一致的样本
    analyze_disagreements(results_df, num_examples=5)
    
    # 7. 打印最终结论
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if results['p_value'] is not None:
        if results['significant_005']:
            print(f"\n✅ The difference between ALBERT and RBT6 is STATISTICALLY SIGNIFICANT")
            print(f"   (p = {results['p_value']:.6f} < 0.05)")
            print(f"\n   {results['better_model']} significantly outperforms the other model.")
        else:
            print(f"\n❌ The difference between ALBERT and RBT6 is NOT statistically significant")
            print(f"   (p = {results['p_value']:.6f} >= 0.05)")
            print(f"\n   The observed performance difference may be due to chance.")
    
    print(f"\n📊 Summary Statistics:")
    print(f"   ALBERT Accuracy: {results['accuracy_a']:.4f}")
    print(f"   RBT6 Accuracy: {results['accuracy_b']:.4f}")
    print(f"   Accuracy Difference: {results['acc_difference']:.4f}")
    print(f"   Chi-squared: {results['chi2']:.4f}")
    print(f"   p-value: {results['p_value']:.6f}")
    
    # 保存统计结果
    stats_df = pd.DataFrame([{
        'Metric': 'McNemar Chi-squared',
        'Value': results['chi2']
    }, {
        'Metric': 'p-value',
        'Value': results['p_value']
    }, {
        'Metric': 'Significant at α=0.05',
        'Value': results['significant_005']
    }, {
        'Metric': 'Significant at α=0.01',
        'Value': results['significant_001']
    }, {
        'Metric': 'ALBERT Accuracy',
        'Value': results['accuracy_a']
    }, {
        'Metric': 'RBT6 Accuracy',
        'Value': results['accuracy_b']
    }, {
        'Metric': 'Better Model',
        'Value': results['better_model']
    }])
    
    stats_df.to_csv('mcnemar_test_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n📁 Results saved to: mcnemar_test_results.csv")
    
    print("\n" + "="*70)
    print("✅ McNemar's Test completed!")
    print("="*70)
