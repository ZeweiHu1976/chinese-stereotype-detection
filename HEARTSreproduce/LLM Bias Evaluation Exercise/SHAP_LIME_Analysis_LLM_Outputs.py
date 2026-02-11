import pandas as pd
import numpy as np
import torch
import shap
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import os

# --- 修改点 1：设置本地模型路径 ---
model_path = "/content/ChineseHeart/Model Training and Evaluation/model_output_albertv2/mgsd_trained"
file_path = 'full_results_llm_exercise.csv'

# 检查路径是否存在
if not os.path.exists(model_path):
    print(f"警告：找不到模型路径 {model_path}，请检查路径是否正确。")

# 检测 GPU
device = 0 if torch.cuda.is_available() else -1
print(f"当前使用设备: {'GPU' if device == 0 else 'CPU'}")

sampled_data = pd.read_csv(file_path)

# --- 修改点 2：在函数中传入 device 并确保 pipeline 加载本地模型 ---

def shap_analysis(sampled_data, model_path, device):
    # 加载本地模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # return_all_scores 在新版本 transformers 中建议改为 top_k=None
    pipe = pipeline("text-classification", model=model_path, tokenizer=tokenizer, device=device, top_k=None)
    
    masker = shap.maskers.Text(tokenizer=r'\b\w+\b')  
    explainer = shap.Explainer(pipe, masker)

    results = []
    # 这里的 class_names 应根据你模型的具体标签映射确定，通常是 'LABEL_0', 'LABEL_1'
    class_names = ['LABEL_0', 'LABEL_1']
    
    for index, row in sampled_data.iterrows():
        text_input = [row['text']]
        shap_values = explainer(text_input)
        
        print(f"SHAP 处理中 -> Group: {row['group']} - Model: {row['model']}")
        
        # 假设我们关注偏见标签（通常是 LABEL_1）
        label_index = 1 
        specific_shap_values = shap_values[:, :, label_index].values
        
        tokens = re.findall(r'\w+', row['text'])
        for token, value in zip(tokens, specific_shap_values[0]):
            results.append({
                'sentence_id': index,
                'token': token,
                'value': value,
                'sentence': row['text'],
                'group': row['group'],
                'predicted_label': row['predicted_label'],
                'model': row['model']
            })
                
    return pd.DataFrame(results)

def custom_tokenizer(text):
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens

def lime_analysis(sampled_data, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline("text-classification", model=model_path, tokenizer=tokenizer, device=device, top_k=None)
    
    def predict_proba(texts):
        preds = pipe(texts)
        # 格式化 pipeline 输出为 numpy 数组以供 LIME 使用
        probabilities = []
        for p in preds:
            # 确保得分按 label 排序排列 [score_0, score_1]
            sorted_p = sorted(p, key=lambda x: x['label'])
            probabilities.append([x['score'] for x in sorted_p])
        return np.array(probabilities)
    
    explainer = LimeTextExplainer(class_names=['LABEL_0', 'LABEL_1'], split_expression=custom_tokenizer)  
    
    results = []
    
    for index, row in sampled_data.iterrows():
        text_input = row['text']
        tokens = custom_tokenizer(text_input)
        # LIME 在解释时会进行多次采样，num_samples=100 是为了速度，准确性要求高可设为 500
        exp = explainer.explain_instance(text_input, predict_proba, num_features=len(tokens), num_samples=100)
        
        print(f"LIME 处理中 -> Group: {row['group']} - Model: {row['model']}")

        explanation_list = exp.as_list(label=1)
        token_value_dict = {token: value for token, value in explanation_list}

        for token in tokens:
            value = token_value_dict.get(token, 0)  
            results.append({
                'sentence_id': index,
                'token': token,
                'value': value,
                'sentence': text_input,
                'group': row['group'],
                'predicted_label': row['predicted_label'],
                'model': row['model']
            })

    return pd.DataFrame(results)

# --- 执行分析 ---

print("开始 SHAP 分析...")
shap_results = shap_analysis(sampled_data, model_path, device)

print("开始 LIME 分析...")
lime_results = lime_analysis(sampled_data, model_path, device)

# 保存中间结果
lime_results.to_csv('lime_results.csv', index=False)
shap_results.to_csv('shap_results.csv', index=False)

# --- 相似度计算部分 (保持原样，但增加了一些鲁棒性处理) ---

merged_df = pd.merge(shap_results, lime_results, on=[col for col in shap_results.columns if col != 'value'], suffixes=('_shap', '_lime'))

grouped = merged_df.groupby('sentence_id').agg({
    'value_shap': list,
    'value_lime': list
})

def compute_metrics(row):
    v_shap = np.array(row['value_shap'])
    v_lime = np.array(row['value_lime'])
    
    # Cosine
    cos = cosine_similarity(v_shap.reshape(1, -1), v_lime.reshape(1, -1))[0][0]
    
    # Pearson
    if len(v_shap) > 1 and np.std(v_shap) > 0 and np.std(v_lime) > 0:
        corr, _ = pearsonr(v_shap, v_lime)
    else:
        corr = 0
        
    # JS Divergence
    def to_prob(v):
        v = v - np.min(v) + 1e-9 # 归一化到正数
        return v / np.sum(v)
    
    js = jensenshannon(to_prob(v_shap), to_prob(v_lime))
    
    return pd.Series([cos, corr, js], index=['cosine_similarity', 'pearson_correlation', 'js_divergence'])

print("计算解释一致性指标...")
metrics = grouped.apply(compute_metrics, axis=1)

# 将结果映射回原 dataframe
merged_df = merged_df.join(metrics, on='sentence_id')

merged_df.to_csv('similarity_results.csv', index=False)
print("任务完成！结果已保存至 similarity_results.csv")
