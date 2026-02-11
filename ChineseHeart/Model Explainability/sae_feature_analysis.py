# SHAP (KernelExplainer) and LIME Analysis - HEARTS Replication
# 使用 KernelExplainer 避免 numba 兼容性问题
# 适配 Python 3.12 + 最新版 shap

import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_text import LimeTextExplainer
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
FILE_PATH = 'fullresult.csv'
MODEL_PATH = 'model_output_rbt6/rbt6_cold'
OUTPUT_DIR = 'explainability_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 中文分词器 ====================
def chinese_tokenizer(text):
    """基于正则的中文分词器"""
    pattern = r'[\u4e00-\u9fa5]+|[a-zA-Z]+|[0-9]+'
    tokens = re.findall(pattern, text)
    return tokens


# ==================== 模型封装 ====================
class ChineseStereotypeClassifier:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict_proba(self, texts):
        """预测概率 - 返回 numpy array [n_samples, 2]"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_probs = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                all_probs.append(probs)
        
        return np.array(all_probs, dtype=np.float64)


# ==================== 数据采样 ====================
def sample_observations(file_path, k=50, seed=42):
    print(f"Loading: {file_path}")
    data = pd.read_csv(file_path)
    print(f"Total: {len(data)}")
    
    sampled = pd.DataFrame()
    for group in data['group'].unique():
        g = data[data['group'] == group]
        correct = g[g['correct'] == 1].sample(n=min(k, len(g[g['correct']==1])), random_state=seed)
        wrong = g[g['correct'] == 0].sample(n=min(k, len(g[g['correct']==0])), random_state=seed)
        sampled = pd.concat([sampled, correct, wrong])
    
    sampled = sampled.reset_index(drop=True)
    print(f"Sampled: {len(sampled)} (correct={len(sampled[sampled['correct']==1])}, wrong={len(sampled[sampled['correct']==0])})")
    return sampled


# ==================== SHAP 分析 (Partition Explainer) ====================
def shap_analysis_partition(sampled_data, classifier):
    """
    使用 PartitionExplainer - 更稳定的文本解释方法
    """
    print("\n" + "="*60)
    print("SHAP Analysis (PartitionExplainer)")
    print("="*60)
    
    # 使用 Partition explainer，对文本更稳定
    masker = shap.maskers.Text(tokenizer=r'[\u4e00-\u9fa5]+|[a-zA-Z]+|[0-9]+')
    
    def predict_fn(texts):
        """确保返回正确格式的numpy数组"""
        probs = classifier.predict_proba(texts)
        return probs.astype(np.float64)
    
    # 使用 PartitionExplainer
    explainer = shap.PartitionExplainer(predict_fn, masker)
    
    results = []
    total = len(sampled_data)
    
    for idx, row in tqdm(sampled_data.iterrows(), total=total, desc="SHAP"):
        text = row['text']
        
        try:
            shap_values = explainer([text])
            
            # 获取 stereotype 类的 SHAP 值
            values = shap_values.values[0]  # [n_tokens, n_classes]
            if len(values.shape) > 1:
                values = values[:, 1]  # stereotype class
            
            tokens = shap_values.data[0]
            
            for token, value in zip(tokens, values):
                token = str(token).strip()
                if token:
                    results.append({
                        'sentence_id': idx,
                        'token': token,
                        'value_shap': float(value),
                        'sentence': text,
                        'group': row['group'],
                        'predicted_label': row['predicted_label'],
                        'actual_label': row['label'],
                        'correct': row['correct']
                    })
        except Exception as e:
            print(f"Row {idx} error: {type(e).__name__}")
            continue
    
    print(f"SHAP done: {len(results)} tokens")
    return pd.DataFrame(results)


# ==================== 备选: 手动 Shapley 采样 ====================
def shap_analysis_manual(sampled_data, classifier, n_samples=50):
    """
    手动实现简化版 Shapley 采样
    不依赖 numba，虽然慢但稳定
    """
    print("\n" + "="*60)
    print("SHAP Analysis (Manual Sampling)")
    print("="*60)
    
    results = []
    total = len(sampled_data)
    
    for idx, row in tqdm(sampled_data.iterrows(), total=total, desc="SHAP (Manual)"):
        text = row['text']
        tokens = chinese_tokenizer(text)
        
        if len(tokens) == 0:
            continue
        
        try:
            # 基准预测
            base_prob = classifier.predict_proba([text])[0, 1]  # stereotype prob
            
            # 对每个 token 计算边际贡献
            token_importance = []
            
            for i, token in enumerate(tokens):
                # 移除该 token
                tokens_without = tokens[:i] + tokens[i+1:]
                text_without = ''.join(tokens_without)
                
                if text_without.strip():
                    prob_without = classifier.predict_proba([text_without])[0, 1]
                else:
                    prob_without = 0.5  # 空文本的基准
                
                # 边际贡献 = 有该token的预测 - 没有该token的预测
                importance = base_prob - prob_without
                token_importance.append(importance)
            
            # 保存结果
            for token, value in zip(tokens, token_importance):
                results.append({
                    'sentence_id': idx,
                    'token': token,
                    'value_shap': float(value),
                    'sentence': text,
                    'group': row['group'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['label'],
                    'correct': row['correct']
                })
                
        except Exception as e:
            print(f"Row {idx} error: {e}")
            continue
    
    print(f"SHAP done: {len(results)} tokens")
    return pd.DataFrame(results)


# ==================== LIME 分析 ====================
def lime_analysis(sampled_data, classifier):
    print("\n" + "="*60)
    print("LIME Analysis")
    print("="*60)
    
    explainer = LimeTextExplainer(
        class_names=['non-stereotype', 'stereotype'],
        split_expression=chinese_tokenizer
    )
    
    results = []
    total = len(sampled_data)
    
    for idx, row in tqdm(sampled_data.iterrows(), total=total, desc="LIME"):
        text = row['text']
        tokens = chinese_tokenizer(text)
        
        if len(tokens) == 0:
            continue
        
        try:
            exp = explainer.explain_instance(
                text,
                classifier.predict_proba,
                num_features=len(tokens),
                num_samples=100
            )
            
            explanation = dict(exp.as_list(label=1))
            
            for token in tokens:
                results.append({
                    'sentence_id': idx,
                    'token': token,
                    'value_lime': float(explanation.get(token, 0.0)),
                    'sentence': text,
                    'group': row['group'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['label'],
                    'correct': row['correct']
                })
                
        except Exception as e:
            print(f"Row {idx} error: {e}")
            continue
    
    print(f"LIME done: {len(results)} tokens")
    return pd.DataFrame(results)


# ==================== 相似度计算 ====================
def compute_similarity(shap_df, lime_df, output_dir):
    print("\n" + "="*60)
    print("Computing Similarity Metrics")
    print("="*60)
    
    if len(shap_df) == 0 or len(lime_df) == 0:
        print("Error: Empty results!")
        return pd.DataFrame(), pd.DataFrame()
    
    sentence_results = []
    
    for sid in shap_df['sentence_id'].unique():
        s_shap = shap_df[shap_df['sentence_id'] == sid]
        s_lime = lime_df[lime_df['sentence_id'] == sid]
        
        if len(s_lime) == 0:
            continue
        
        merged = pd.merge(
            s_shap[['token', 'value_shap']],
            s_lime[['token', 'value_lime']],
            on='token', how='inner'
        )
        
        if len(merged) < 2:
            continue
        
        v_shap = merged['value_shap'].values
        v_lime = merged['value_lime'].values
        
        # Cosine similarity
        cos_sim = np.dot(v_shap, v_lime) / (np.linalg.norm(v_shap) * np.linalg.norm(v_lime) + 1e-8)
        
        # Pearson correlation
        try:
            pear_r, _ = pearsonr(v_shap, v_lime)
        except:
            pear_r = np.nan
        
        # JS divergence
        def to_prob(v):
            v = v - v.min() + 1e-8
            return v / v.sum()
        js_div = jensenshannon(to_prob(v_shap), to_prob(v_lime))
        
        meta = s_shap.iloc[0]
        sentence_results.append({
            'sentence_id': sid,
            'sentence': meta['sentence'],
            'group': meta['group'],
            'predicted_label': meta['predicted_label'],
            'actual_label': meta['actual_label'],
            'correct': meta['correct'],
            'n_tokens': len(merged),
            'cosine_similarity': cos_sim,
            'pearson_correlation': pear_r,
            'js_divergence': js_div
        })
    
    sent_df = pd.DataFrame(sentence_results)
    
    if len(sent_df) > 0:
        sent_df.to_csv(f'{output_dir}/sentence_similarity.csv', index=False, encoding='utf-8-sig')
        
        print(f"\nResults (n={len(sent_df)}):")
        print(f"  Cosine: {sent_df['cosine_similarity'].mean():.4f} ± {sent_df['cosine_similarity'].std():.4f}")
        print(f"  Pearson: {sent_df['pearson_correlation'].mean():.4f} ± {sent_df['pearson_correlation'].std():.4f}")
        print(f"  JS Div: {sent_df['js_divergence'].mean():.4f} ± {sent_df['js_divergence'].std():.4f}")
        
        # By correctness
        for c in [1, 0]:
            sub = sent_df[sent_df['correct'] == c]
            if len(sub) > 0:
                label = "Correct" if c == 1 else "Wrong"
                print(f"\n  {label} (n={len(sub)}):")
                print(f"    Cosine: {sub['cosine_similarity'].mean():.4f}")
                print(f"    Pearson: {sub['pearson_correlation'].mean():.4f}")
                print(f"    JS Div: {sub['js_divergence'].mean():.4f}")
    
    # Token stats
    merged_all = pd.merge(
        shap_df, lime_df[['sentence_id', 'token', 'value_lime']],
        on=['sentence_id', 'token'], how='inner'
    )
    
    if len(merged_all) > 0:
        merged_all.to_csv(f'{output_dir}/merged_shap_lime.csv', index=False, encoding='utf-8-sig')
        
        token_stats = merged_all.groupby('token').agg({
            'value_shap': ['mean', 'count'],
            'value_lime': 'mean'
        }).reset_index()
        token_stats.columns = ['token', 'shap_mean', 'count', 'lime_mean']
        token_stats = token_stats.sort_values('shap_mean', ascending=False)
        token_stats.to_csv(f'{output_dir}/token_stats.csv', index=False, encoding='utf-8-sig')
        
        print(f"\nTop stereotype tokens:")
        for _, r in token_stats.head(15).iterrows():
            print(f"  {r['token']}: SHAP={r['shap_mean']:.4f}, LIME={r['lime_mean']:.4f}")
    else:
        token_stats = pd.DataFrame()
    
    return sent_df, token_stats


# ==================== HEARTS Table 2 ====================
def generate_table2(shap_df, lime_df, output_dir):
    print("\n" + "="*60)
    print("Generating HEARTS Table 2")
    print("="*60)
    
    if len(shap_df) == 0 or len(lime_df) == 0:
        return pd.DataFrame()
    
    rows = []
    for sid in list(shap_df['sentence_id'].unique())[:25]:
        s_shap = shap_df[shap_df['sentence_id'] == sid]
        s_lime = lime_df[lime_df['sentence_id'] == sid]
        
        merged = pd.merge(
            s_shap[['token', 'value_shap']],
            s_lime[['token', 'value_lime']],
            on='token', how='inner'
        )
        
        if len(merged) < 2:
            continue
        
        v_shap, v_lime = merged['value_shap'].values, merged['value_lime'].values
        
        # Top tokens
        top = merged.nlargest(5, 'value_shap')
        rankings = ', '.join([f'"{r["token"]}": {r["value_shap"]:.3f}' for _, r in top.iterrows()])
        
        meta = s_shap.iloc[0]
        cos = np.dot(v_shap, v_lime) / (np.linalg.norm(v_shap) * np.linalg.norm(v_lime) + 1e-8)
        
        try:
            pear, _ = pearsonr(v_shap, v_lime)
        except:
            pear = np.nan
        
        rows.append({
            'Text': meta['sentence'][:40] + '...' if len(meta['sentence']) > 40 else meta['sentence'],
            'Pred': int(meta['predicted_label']),
            'True': int(meta['actual_label']),
            'Token Rankings': rankings,
            'Cos': round(cos, 3),
            'Pear': round(pear, 3) if not np.isnan(pear) else 'N/A'
        })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_csv(f'{output_dir}/hearts_table2.csv', index=False, encoding='utf-8-sig')
        print(df.head(10).to_string())
    return df


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("="*70)
    print("SHAP & LIME - HEARTS Replication")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载模型
    print("\n[1/5] Loading model...")
    classifier = ChineseStereotypeClassifier(MODEL_PATH)
    
    # 采样
    print("\n[2/5] Sampling...")
    sampled = sample_observations(FILE_PATH, k=50, seed=42)
    sampled.to_csv(f'{OUTPUT_DIR}/sampled_data.csv', index=False, encoding='utf-8-sig')
    
    # SHAP - 使用手动方法避免 numba 问题
    print("\n[3/5] SHAP...")
    try:
        # 先尝试 PartitionExplainer
        shap_results = shap_analysis_partition(sampled, classifier)
    except Exception as e:
        print(f"PartitionExplainer failed: {e}")
        print("Falling back to manual method...")
        shap_results = shap_analysis_manual(sampled, classifier)
    
    if len(shap_results) > 0:
        shap_results.to_csv(f'{OUTPUT_DIR}/shap_results.csv', index=False, encoding='utf-8-sig')
    
    # LIME
    print("\n[4/5] LIME...")
    lime_results = lime_analysis(sampled, classifier)
    if len(lime_results) > 0:
        lime_results.to_csv(f'{OUTPUT_DIR}/lime_results.csv', index=False, encoding='utf-8-sig')
    
    # 相似度
    print("\n[5/5] Similarity...")
    sent_sim, token_stats = compute_similarity(shap_results, lime_results, OUTPUT_DIR)
    
    # Table 2
    generate_table2(shap_results, lime_results, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✅ DONE")
    print(f"Output: {OUTPUT_DIR}/")
    print("="*70)