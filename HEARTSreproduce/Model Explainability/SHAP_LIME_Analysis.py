from transformers import pipeline
import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_text import LimeTextExplainer
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import os
import warnings

# 忽略部分计算警告
warnings.filterwarnings("ignore")

# ================= 配置路径 =================
input_file = 'full_results_albertv2.csv'
# 确保使用绝对路径
model_path = "/content/ChineseHeart/Model Training and Evaluation/model_output_albertv2/mgsd_trained"

# ================= 1. 数据采样函数 =================
def sample_observations(file_path, k, seed):
    print(f"正在从 {file_path} 采样数据...")
    data = pd.read_csv(file_path)
    
    combinations = data.groupby(['dataset_name', 'categorisation'])
    sampled_data = pd.DataFrame(columns=data.columns)
    
    for name, group in combinations:
        same_label = group[group['predicted_label'] == group['actual_label']]
        diff_label = group[group['predicted_label'] != group['actual_label']]
        
        if len(same_label) >= k:
            same_sample = same_label.sample(n=k, random_state=seed)
        else:
            same_sample = same_label
        
        if len(diff_label) >= k:
            diff_sample = diff_label.sample(n=k, random_state=seed)
        else:
            diff_sample = diff_label
        
        sampled_data = pd.concat([sampled_data, same_sample, diff_sample], axis=0)
    
    sampled_data.reset_index(drop=True, inplace=True)
    print(f"采样完成，共 {len(sampled_data)} 条数据。")
    return sampled_data

# ================= 2. SHAP 分析函数 =================
def shap_analysis(sampled_data, model_path):
    print("正在运行 SHAP 分析...")
    # 修复: 使用 top_k=None 
    pipe = pipeline("text-classification", model=model_path, tokenizer=model_path, top_k=None, device=0) 
    masker = shap.maskers.Text(tokenizer=r'\b\w+\b') 
    explainer = shap.Explainer(pipe, masker)

    results = []
    class_names = ['LABEL_0', 'LABEL_1']
    
    # 批量处理或逐行处理
    for index, row in sampled_data.iterrows():
        text_input = [row['text']]
        try:
            shap_values = explainer(text_input)
            
            label_index = class_names.index("LABEL_1")  
            specific_shap_values = shap_values[:, :, label_index].values
            
            tokens = re.findall(r'\w+', row['text'])
            # 防止 token 长度不匹配的保护
            min_len = min(len(tokens), len(specific_shap_values[0]))
            
            for i in range(min_len):
                results.append({
                    'sentence_id': index, 
                    'token': tokens[i], 
                    'value_shap': specific_shap_values[0][i],
                    'sentence': row['text'],
                    'dataset': row['dataset_name'],
                    'categorisation': row['categorisation'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['actual_label']
                })
        except Exception as e:
            print(f"SHAP Error at index {index}: {e}")
                
    return pd.DataFrame(results)

# ================= 3. LIME 分析函数 =================
def custom_tokenizer(text):
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens

def lime_analysis(sampled_data, model_path):
    print("正在运行 LIME 分析...")
    # 修复: Pipeline 初始化加入 top_k=None
    pipe = pipeline("text-classification", model=model_path, tokenizer=model_path, top_k=None, device=0)
    
    def predict_proba(texts):
        # 修复: 重写概率提取逻辑，确保返回 (N, 2) 的形状且顺序正确
        preds = pipe(texts) 
        formatted_probs = []
        for p in preds:
            # 按标签名称排序，确保 LABEL_0 在前，LABEL_1 在后
            sorted_p = sorted(p, key=lambda x: x['label'])
            scores = [x['score'] for x in sorted_p]
            formatted_probs.append(scores)
            
        return np.array(formatted_probs)    
    
    explainer = LimeTextExplainer(class_names=['LABEL_0', 'LABEL_1'], split_expression=lambda x: custom_tokenizer(x))  
    
    results = []
    
    for index, row in sampled_data.iterrows():
        text_input = row['text']
        tokens = custom_tokenizer(text_input)
        try:
            # num_features 设置为 token 长度，确保覆盖全句
            exp = explainer.explain_instance(text_input, predict_proba, num_features=len(tokens), num_samples=100)
            
            # 获取 label 1 的解释
            explanation_list = exp.as_list(label=1)
            token_value_dict = {token: value for token, value in explanation_list}

            for token in tokens:
                value = token_value_dict.get(token, 0)  
                results.append({
                    'sentence_id': index, 
                    'token': token, 
                    'value_lime': value,
                    'sentence': text_input,
                    'dataset': row['dataset_name'],
                    'categorisation': row['categorisation'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['actual_label']
                })
        except Exception as e:
            print(f"LIME Error explaining sentence {index}: {e}")

    return pd.DataFrame(results)

# ================= 4. 健壮的相似度计算函数 =================
def to_probability_distribution(values):
    values = np.array(values)
    min_val = np.min(values)
    if min_val < 0:
        values += abs(min_val)
    total = np.sum(values)
    if total > 0:
        values /= total
    else:
        # 如果总和为0，返回均匀分布防止报错
        values = np.ones_like(values) / len(values)
    return values

def compute_pearson_correlation(vector1, vector2):
    # 修复: 长度检查和零方差检查
    if len(vector1) < 2:
        return np.nan
    if np.std(vector1) == 0 or np.std(vector2) == 0:
        return np.nan
    try:
        correlation, _ = pearsonr(vector1, vector2)
        return correlation
    except:
        return np.nan

def compute_cosine_similarity(vector1, vector2):
    try:
        return cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    except:
        return np.nan

def compute_js_divergence(vector1, vector2):
    try:
        prob1 = to_probability_distribution(vector1.copy())
        prob2 = to_probability_distribution(vector2.copy())
        return jensenshannon(prob1, prob2)
    except:
        return np.nan

# ================= 主执行流程 =================

if __name__ == "__main__":
    # 1. 采样
    sampled_data = sample_observations(input_file, k=37, seed=42)
    sampled_data.to_csv('sampled_data.csv', index=False)

    # 2. 运行模型解释
    shap_results = shap_analysis(sampled_data, model_path)
    shap_results.to_csv('shap_results.csv', index=False)
    
    lime_results = lime_analysis(sampled_data, model_path)
    lime_results.to_csv('lime_results.csv', index=False)

    print("模型解释完成，开始计算相似度指标...")

    # 读取刚刚生成的数据（确保格式一致）
    shap_df = pd.read_csv('shap_results.csv')
    lime_df = pd.read_csv('lime_results.csv')

    # 3. 计算 Token 层面相似度
    print("计算 Token Level Similarity...")
    token_shap = shap_df.groupby('token')['value_shap'].apply(list).reset_index()
    token_lime = lime_df.groupby('token')['value_lime'].apply(list).reset_index()
    token_merged = pd.merge(token_shap, token_lime, on='token', how='inner')

    token_merged['cosine_similarity'] = token_merged.apply(lambda row: compute_cosine_similarity(np.array(row['value_shap']), np.array(row['value_lime'])), axis=1)
    token_merged['pearson_correlation'] = token_merged.apply(lambda row: compute_pearson_correlation(np.array(row['value_shap']), np.array(row['value_lime'])), axis=1)
    token_merged['js_divergence'] = token_merged.apply(lambda row: compute_js_divergence(np.array(row['value_shap']), np.array(row['value_lime'])), axis=1)
    
    token_merged.to_csv('token_level_similarity.csv', index=False)

    # 4. 计算 Sentence 层面相似度
    print("计算 Sentence Level Similarity...")
    # 寻找公共列用于合并
    cols_to_exclude = ['value_shap', 'value_lime', 'Unnamed: 0']
    common_columns = [col for col in shap_df.columns if col not in cols_to_exclude]
    common_columns = list(set(common_columns).intersection(lime_df.columns))

    merged_df = pd.merge(shap_df, lime_df, on=common_columns, suffixes=('_shap', '_lime'))
    
    # 按 sentence_id 聚合 values 为 list
    grouped = merged_df.groupby('sentence_id').agg({
        'value_shap': list,
        'value_lime': list
    }).reset_index()

    grouped['cosine_similarity'] = grouped.apply(lambda row: compute_cosine_similarity(np.array(row['value_shap']), np.array(row['value_lime'])), axis=1)
    grouped['pearson_correlation'] = grouped.apply(lambda row: compute_pearson_correlation(np.array(row['value_shap']), np.array(row['value_lime'])), axis=1)
    grouped['js_divergence'] = grouped.apply(lambda row: compute_js_divergence(np.array(row['value_shap']), np.array(row['value_lime'])), axis=1)

    grouped.to_csv('sentence_level_similarity_results.csv', index=False)
    
    print("✅ 所有步骤完成！结果文件已保存：")
    print("1. sampled_data.csv")
    print("2. shap_results.csv")
    print("3. lime_results.csv")
    print("4. token_level_similarity.csv")
    print("5. sentence_level_similarity_results.csv")
