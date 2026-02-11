"""
Model Explainability Analysis for Chinese Regional Stereotype Detection
Adapted for RBT6 model with SHAP and LIME analysis

Version 3: Fixed RBT6 model embedding layer access
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_text import LimeTextExplainer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import os
import jieba
import warnings
warnings.filterwarnings('ignore')

# ===================== Configuration =====================
FILE_PATH = '/content/ChineseHeart/Model Explainability/fullresult.csv'
MODEL_PATH = '/content/ChineseHeart/Model Training and Evaluation/model_output_rbt6/rbt6_cold'
OUTPUT_DIR = 'explainability_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== Chinese Tokenizer =====================
def chinese_tokenizer(text):
    tokens = list(jieba.cut(text))
    tokens = [token.strip() for token in tokens if token.strip()]
    return tokens


def chinese_tokenizer_for_lime(text):
    return chinese_tokenizer(text)


# ===================== Sample Observations =====================
def sample_observations(file_path, k, seed):
    data = pd.read_csv(file_path)
    combinations = data.groupby(['group', 'original_label'])
    sampled_list = []
    
    for name, group_df in combinations:
        same_label = group_df[group_df['correct'] == 1]
        diff_label = group_df[group_df['correct'] == 0]
        
        if len(same_label) >= k:
            same_sample = same_label.sample(n=k, random_state=seed)
        else:
            same_sample = same_label
        
        if len(diff_label) >= k:
            diff_sample = diff_label.sample(n=k, random_state=seed)
        else:
            diff_sample = diff_label
        
        sampled_list.append(same_sample)
        sampled_list.append(diff_sample)
    
    sampled_data = pd.concat(sampled_list, axis=0, ignore_index=True)
    print(f"Sampled {len(sampled_data)} observations")
    print(sampled_data.head())
    return sampled_data


# ===================== Get Embedding Layer for Different Model Architectures =====================
def get_embedding_layer(model):
    """
    Get the word embedding layer from various model architectures.
    Supports BERT, RoBERTa, ALBERT, and other transformer models.
    """
    # Try different common paths for embedding layers
    possible_paths = [
        # BERT-based models (including RBT6, Chinese-BERT, etc.)
        lambda m: m.bert.embeddings.word_embeddings,
        # RoBERTa-based models
        lambda m: m.roberta.embeddings.word_embeddings,
        # ALBERT-based models
        lambda m: m.albert.embeddings.word_embeddings,
        # XLNet
        lambda m: m.transformer.word_embedding,
        # DistilBERT
        lambda m: m.distilbert.embeddings.word_embeddings,
        # ELECTRA
        lambda m: m.electra.embeddings.word_embeddings,
        # Generic get_input_embeddings (fallback)
        lambda m: m.get_input_embeddings(),
    ]
    
    for get_emb in possible_paths:
        try:
            emb = get_emb(model)
            if emb is not None:
                print(f"Found embedding layer: {type(emb)}")
                return emb
        except (AttributeError, TypeError):
            continue
    
    raise ValueError("Could not find embedding layer in the model")


# ===================== SHAP Analysis using Gradient-based Attribution =====================
def shap_analysis_gradient(sampled_data, model_path):
    """
    Compute gradient-based attribution scores (Gradient x Input).
    Compatible with RBT6 and other Chinese BERT models.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get embedding layer
    try:
        embedding_layer = get_embedding_layer(model)
    except ValueError as e:
        print(f"Error getting embedding layer: {e}")
        print("Falling back to attention-based analysis...")
        return shap_analysis_attention(sampled_data, model_path)
    
    results = []
    
    for index, row in sampled_data.iterrows():
        text = row['text']
        
        try:
            # Tokenize
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # Get tokens for output
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Get embeddings and enable gradients
            embeddings = embedding_layer(input_ids)
            embeddings = embeddings.clone().detach().requires_grad_(True)
            
            # Forward pass with embeddings
            # We need to modify the forward pass to use our embeddings
            model.zero_grad()
            
            # For BERT-based models, we need to pass through the encoder
            if hasattr(model, 'bert'):
                # Get the rest of BERT embeddings (position, token_type)
                position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
                token_type_ids = torch.zeros_like(input_ids)
                
                # Get position and token type embeddings
                position_embeddings = model.bert.embeddings.position_embeddings(position_ids)
                token_type_embeddings = model.bert.embeddings.token_type_embeddings(token_type_ids)
                
                # Combine embeddings
                combined_embeddings = embeddings + position_embeddings + token_type_embeddings
                combined_embeddings = model.bert.embeddings.LayerNorm(combined_embeddings)
                combined_embeddings = model.bert.embeddings.dropout(combined_embeddings)
                
                # Create a new tensor that requires grad
                combined_embeddings = combined_embeddings.clone().detach().requires_grad_(True)
                
                # Pass through encoder
                extended_attention_mask = model.bert.get_extended_attention_mask(
                    attention_mask, input_ids.shape, device
                )
                encoder_outputs = model.bert.encoder(
                    combined_embeddings,
                    attention_mask=extended_attention_mask,
                )
                sequence_output = encoder_outputs[0]
                pooled_output = model.bert.pooler(sequence_output) if model.bert.pooler else sequence_output[:, 0]
                
                # Classifier
                pooled_output = model.dropout(pooled_output)
                logits = model.classifier(pooled_output)
                
            else:
                # Fallback: try direct forward with inputs_embeds
                outputs = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
            
            # Get probability for stereotype class
            probs = torch.softmax(logits, dim=-1)
            stereotype_prob = probs[0, 1]
            
            # Backward pass
            stereotype_prob.backward()
            
            # Get gradients from combined_embeddings
            if 'combined_embeddings' in dir() and combined_embeddings.grad is not None:
                gradients = combined_embeddings.grad[0]
                input_embeds = combined_embeddings[0].detach()
            else:
                # This shouldn't happen, but handle it
                gradients = torch.zeros_like(embeddings[0])
                input_embeds = embeddings[0].detach()
            
            # Compute attribution scores (gradient * input, then sum over embedding dim)
            attributions = (gradients * input_embeds).sum(dim=-1)
            attributions = attributions.cpu().numpy()
            
            print(f"[SHAP-Grad] Index: {index} | Group: {row['group']} | "
                  f"Original Label: {row['original_label']} | "
                  f"Predicted: {row['predicted_label']} | Actual: {row['label']}")
            
            for token, value in zip(tokens, attributions):
                if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                    continue
                clean_token = token.replace('##', '')
                if clean_token.strip():
                    results.append({
                        'sentence_id': index,
                        'token': clean_token,
                        'value_shap': float(value),
                        'sentence': text,
                        'group': row['group'],
                        'original_label': row['original_label'],
                        'predicted_label': row['predicted_label'],
                        'actual_label': row['label']
                    })
                    
        except Exception as e:
            print(f"[SHAP-Grad] Error processing index {index}: {e}")
            # Fallback: use jieba tokenization with zero values
            tokens = chinese_tokenizer(text)
            for token in tokens:
                results.append({
                    'sentence_id': index,
                    'token': token,
                    'value_shap': 0.0,
                    'sentence': text,
                    'group': row['group'],
                    'original_label': row['original_label'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['label']
                })
    
    return pd.DataFrame(results)


# ===================== Alternative: Attention-based Attribution =====================
def shap_analysis_attention(sampled_data, model_path):
    """
    Use attention weights as a proxy for token importance.
    This is a simpler but often effective approach.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        output_attentions=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = []
    
    for index, row in sampled_data.iterrows():
        text = row['text']
        
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                attentions = outputs.attentions  # Tuple of attention tensors
            
            # Get tokens
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Average attention across all layers and heads
            # attentions: tuple of (batch, num_heads, seq_len, seq_len)
            attention_weights = torch.stack(attentions).mean(dim=(0, 1, 2))  # Average to get (seq_len,)
            attention_weights = attention_weights[0].cpu().numpy()  # First sequence
            
            print(f"[SHAP-Attn] Index: {index} | Group: {row['group']} | "
                  f"Original Label: {row['original_label']} | "
                  f"Predicted: {row['predicted_label']} | Actual: {row['label']}")
            
            for token, value in zip(tokens, attention_weights):
                if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                    continue
                clean_token = token.replace('##', '')
                if clean_token.strip():
                    results.append({
                        'sentence_id': index,
                        'token': clean_token,
                        'value_shap': float(value),
                        'sentence': text,
                        'group': row['group'],
                        'original_label': row['original_label'],
                        'predicted_label': row['predicted_label'],
                        'actual_label': row['label']
                    })
                    
        except Exception as e:
            print(f"[SHAP-Attn] Error processing index {index}: {e}")
            tokens = chinese_tokenizer(text)
            for token in tokens:
                results.append({
                    'sentence_id': index,
                    'token': token,
                    'value_shap': 0.0,
                    'sentence': text,
                    'group': row['group'],
                    'original_label': row['original_label'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['label']
                })
    
    return pd.DataFrame(results)


# ===================== Simple Occlusion-based Attribution (Most Reliable) =====================
def shap_analysis_occlusion(sampled_data, model_path):
    """
    Compute token importance by measuring prediction change when each token is removed.
    Most reliable method that works with any model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    def get_prob(text):
        """Get stereotype probability for a text."""
        if not text or not text.strip():
            return 0.5
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        return probs[0, 1].item()  # Stereotype probability
    
    results = []
    
    for index, row in sampled_data.iterrows():
        text = row['text']
        tokens = chinese_tokenizer(text)
        
        if not tokens:
            continue
        
        try:
            # Get baseline probability
            baseline_prob = get_prob(text)
            
            attributions = []
            for i, token in enumerate(tokens):
                # Create text with token removed
                tokens_without = tokens[:i] + tokens[i+1:]
                text_without = ''.join(tokens_without)
                
                # Get probability without this token
                prob_without = get_prob(text_without)
                
                # Attribution = change in probability when token is removed
                # Positive means token contributes to stereotype prediction
                attribution = baseline_prob - prob_without
                attributions.append(attribution)
            
            print(f"[SHAP-Occ] Index: {index} | Group: {row['group']} | "
                  f"Original Label: {row['original_label']} | "
                  f"Predicted: {row['predicted_label']} | Actual: {row['label']}")
            
            for token, value in zip(tokens, attributions):
                results.append({
                    'sentence_id': index,
                    'token': token,
                    'value_shap': float(value),
                    'sentence': text,
                    'group': row['group'],
                    'original_label': row['original_label'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['label']
                })
                
        except Exception as e:
            print(f"[SHAP-Occ] Error processing index {index}: {e}")
            for token in tokens:
                results.append({
                    'sentence_id': index,
                    'token': token,
                    'value_shap': 0.0,
                    'sentence': text,
                    'group': row['group'],
                    'original_label': row['original_label'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['label']
                })
    
    return pd.DataFrame(results)


# ===================== LIME Analysis =====================
def lime_analysis(sampled_data, model_path):
    """
    Compute LIME values for each token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    def predict_proba(texts):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(t) if t and str(t).strip() else "空" for t in texts]
        
        with torch.no_grad():
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probs
    
    explainer = LimeTextExplainer(
        class_names=['non-stereotype', 'stereotype'],
        split_expression=chinese_tokenizer_for_lime,
        bow=True,
        random_state=42
    )
    
    results = []
    
    for index, row in sampled_data.iterrows():
        text = row['text']
        tokens = chinese_tokenizer(text)
        
        if not tokens:
            continue
        
        try:
            exp = explainer.explain_instance(
                text,
                predict_proba,
                num_features=min(len(tokens), 20),
                num_samples=500
            )
            
            print(f"[LIME] Index: {index} | Group: {row['group']} | "
                  f"Original Label: {row['original_label']} | "
                  f"Predicted: {row['predicted_label']} | Actual: {row['label']}")
            
            explanation_list = exp.as_list(label=1)
            token_value_dict = {token: value for token, value in explanation_list}
            
            for token in tokens:
                value = token_value_dict.get(token, 0.0)
                results.append({
                    'sentence_id': index,
                    'token': token,
                    'value_lime': float(value),
                    'sentence': text,
                    'group': row['group'],
                    'original_label': row['original_label'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['label']
                })
                
        except Exception as e:
            print(f"[LIME] Error processing index {index}: {e}")
            for token in tokens:
                results.append({
                    'sentence_id': index,
                    'token': token,
                    'value_lime': 0.0,
                    'sentence': text,
                    'group': row['group'],
                    'original_label': row['original_label'],
                    'predicted_label': row['predicted_label'],
                    'actual_label': row['label']
                })

    return pd.DataFrame(results)


# ===================== Similarity Metrics =====================
def compute_cosine_similarity(vector1, vector2):
    v1 = np.array(vector1).reshape(1, -1)
    v2 = np.array(vector2).reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]


def compute_pearson_correlation(vector1, vector2):
    if len(vector1) < 2 or len(vector2) < 2:
        return np.nan
    # Check if vectors have variance
    if np.std(vector1) == 0 or np.std(vector2) == 0:
        return np.nan
    try:
        correlation, _ = pearsonr(vector1, vector2)
        return correlation
    except:
        return np.nan


def to_probability_distribution(values):
    values = np.array(values, dtype=np.float64)
    min_val = np.min(values)
    if min_val < 0:
        values = values + abs(min_val)
    values = values + 1e-10
    total = np.sum(values)
    if total > 0:
        values = values / total
    return values


def compute_js_divergence(vector1, vector2):
    prob1 = to_probability_distribution(np.array(vector1))
    prob2 = to_probability_distribution(np.array(vector2))
    return jensenshannon(prob1, prob2)


# ===================== Similarity Analysis =====================
def compute_token_level_similarity(shap_df, lime_df, output_dir):
    if shap_df.empty or lime_df.empty:
        print("Warning: Empty SHAP or LIME results")
        return pd.DataFrame()
    
    token_shap = shap_df.groupby('token')['value_shap'].apply(list).reset_index()
    token_lime = lime_df.groupby('token')['value_lime'].apply(list).reset_index()
    token_merged = pd.merge(token_shap, token_lime, on='token', how='inner')
    
    results = []
    for _, row in token_merged.iterrows():
        shap_vals = np.array(row['value_shap'])
        lime_vals = np.array(row['value_lime'])
        
        min_len = min(len(shap_vals), len(lime_vals))
        if min_len < 2:
            continue
            
        shap_vals = shap_vals[:min_len]
        lime_vals = lime_vals[:min_len]
        
        results.append({
            'token': row['token'],
            'num_occurrences': min_len,
            'cosine_similarity': compute_cosine_similarity(shap_vals, lime_vals),
            'pearson_correlation': compute_pearson_correlation(shap_vals, lime_vals),
            'js_divergence': compute_js_divergence(shap_vals, lime_vals)
        })
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        output_path = os.path.join(output_dir, 'token_level_similarity.csv')
        result_df.to_csv(output_path, index=False)
        print(f"Token-level similarity saved to {output_path}")
    return result_df


def compute_sentence_level_similarity(shap_df, lime_df, output_dir):
    if shap_df.empty or lime_df.empty:
        print("Warning: Empty SHAP or LIME results")
        return pd.DataFrame()
    
    merge_cols = ['sentence_id', 'token', 'sentence', 'group', 'original_label', 
                  'predicted_label', 'actual_label']
    
    merged_df = pd.merge(
        shap_df[merge_cols + ['value_shap']], 
        lime_df[merge_cols + ['value_lime']], 
        on=merge_cols, 
        how='inner'
    )
    
    if merged_df.empty:
        print("Warning: No matching records")
        return pd.DataFrame()
    
    results = []
    for sentence_id in merged_df['sentence_id'].unique():
        sentence_data = merged_df[merged_df['sentence_id'] == sentence_id]
        
        shap_vals = sentence_data['value_shap'].values
        lime_vals = sentence_data['value_lime'].values
        
        if len(shap_vals) < 2:
            continue
        
        first_row = sentence_data.iloc[0]
        
        results.append({
            'sentence_id': sentence_id,
            'sentence': first_row['sentence'],
            'group': first_row['group'],
            'original_label': first_row['original_label'],
            'predicted_label': first_row['predicted_label'],
            'actual_label': first_row['actual_label'],
            'num_tokens': len(shap_vals),
            'cosine_similarity': compute_cosine_similarity(shap_vals, lime_vals),
            'pearson_correlation': compute_pearson_correlation(shap_vals, lime_vals),
            'js_divergence': compute_js_divergence(shap_vals, lime_vals)
        })
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        output_path = os.path.join(output_dir, 'sentence_level_similarity.csv')
        result_df.to_csv(output_path, index=False)
        print(f"Sentence-level similarity saved to {output_path}")
    return result_df


def compute_group_level_statistics(sentence_similarity_df, output_dir):
    if sentence_similarity_df.empty:
        return pd.DataFrame()
    
    grouped = sentence_similarity_df.groupby(['group', 'original_label']).agg({
        'cosine_similarity': ['mean', 'std', 'min', 'max'],
        'pearson_correlation': ['mean', 'std', 'min', 'max'],
        'js_divergence': ['mean', 'std', 'min', 'max'],
        'sentence_id': 'count'
    }).reset_index()
    
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped = grouped.rename(columns={'sentence_id_count': 'num_sentences'})
    
    output_path = os.path.join(output_dir, 'group_level_statistics.csv')
    grouped.to_csv(output_path, index=False)
    print(f"Group-level statistics saved to {output_path}")
    return grouped


def compute_prediction_correctness_analysis(sentence_similarity_df, output_dir):
    if sentence_similarity_df.empty:
        return pd.DataFrame()
    
    df = sentence_similarity_df.copy()
    df['correct'] = (df['predicted_label'] == df['actual_label']).astype(int)
    
    grouped = df.groupby('correct').agg({
        'cosine_similarity': ['mean', 'std'],
        'pearson_correlation': ['mean', 'std'],
        'js_divergence': ['mean', 'std'],
        'sentence_id': 'count'
    }).reset_index()
    
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped = grouped.rename(columns={'sentence_id_count': 'num_sentences'})
    
    output_path = os.path.join(output_dir, 'correctness_analysis.csv')
    grouped.to_csv(output_path, index=False)
    print(f"Correctness analysis saved to {output_path}")
    return grouped


# ===================== Main Execution =====================
def main():
    print("=" * 60)
    print("Model Explainability Analysis for Chinese Stereotype Detection")
    print("=" * 60)
    
    # Step 1: Sample observations
    print("\n[Step 1] Sampling observations...")
    sampled_data = sample_observations(FILE_PATH, k=37, seed=42)
    sampled_path = os.path.join(OUTPUT_DIR, 'sampled_data.csv')
    sampled_data.to_csv(sampled_path, index=False)
    print(f"Sampled data saved to {sampled_path}")
    
    # Step 2: SHAP analysis using Occlusion (most reliable for any model)
    print("\n[Step 2] Running SHAP analysis (occlusion-based)...")
    shap_results = shap_analysis_occlusion(sampled_data, MODEL_PATH)
    shap_path = os.path.join(OUTPUT_DIR, 'shap_results.csv')
    shap_results.to_csv(shap_path, index=False)
    print(f"SHAP results saved to {shap_path}")
    print(f"SHAP results shape: {shap_results.shape}")
    
    # Check if SHAP produced valid results
    non_zero_shap = (shap_results['value_shap'] != 0).sum()
    print(f"Non-zero SHAP values: {non_zero_shap}/{len(shap_results)}")
    
    # Step 3: LIME analysis
    print("\n[Step 3] Running LIME analysis...")
    lime_results = lime_analysis(sampled_data, MODEL_PATH)
    lime_path = os.path.join(OUTPUT_DIR, 'lime_results.csv')
    lime_results.to_csv(lime_path, index=False)
    print(f"LIME results saved to {lime_path}")
    print(f"LIME results shape: {lime_results.shape}")
    
    non_zero_lime = (lime_results['value_lime'] != 0).sum()
    print(f"Non-zero LIME values: {non_zero_lime}/{len(lime_results)}")
    
    # Step 4: Compute similarity metrics
    print("\n[Step 4] Computing similarity metrics...")
    
    if shap_results.empty or lime_results.empty:
        print("Error: SHAP or LIME results are empty.")
        return
    
    token_similarity = compute_token_level_similarity(shap_results, lime_results, OUTPUT_DIR)
    sentence_similarity = compute_sentence_level_similarity(shap_results, lime_results, OUTPUT_DIR)
    
    if not sentence_similarity.empty:
        group_stats = compute_group_level_statistics(sentence_similarity, OUTPUT_DIR)
        correctness_analysis = compute_prediction_correctness_analysis(sentence_similarity, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    
    if not sentence_similarity.empty:
        print("\n[Summary Statistics]")
        print(f"Total sentences analyzed: {len(sentence_similarity)}")
        print(f"Average cosine similarity: {sentence_similarity['cosine_similarity'].mean():.4f}")
        
        valid_pearson = sentence_similarity['pearson_correlation'].dropna()
        if len(valid_pearson) > 0:
            print(f"Average Pearson correlation: {valid_pearson.mean():.4f}")
        else:
            print("Pearson correlation: N/A (insufficient variance)")
            
        print(f"Average JS divergence: {sentence_similarity['js_divergence'].mean():.4f}")


if __name__ == "__main__":
    main()