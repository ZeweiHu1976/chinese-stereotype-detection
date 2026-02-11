# SAE Training with TopK Sparsity
# 使用 TopK 强制稀疏的 SAE 训练
# 解决 L1 惩罚无效的问题

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm
import os
import json


# ==================== 配置 ====================
class SAEConfig:
    # 模型路径
    model_dir = "model_output_albert_chinese/albert_chinese_cold"
    data_path = "cold.csv"
    layer_idx = 8
    
    # SAE 参数
    hidden_dim = 768
    expansion_factor = 8
    
    @property
    def sae_dim(self):
        return self.hidden_dim * self.expansion_factor
    
    # ⭐ TopK 稀疏参数
    # K 值决定每个样本最多激活多少个特征
    # 推荐: sae_dim 的 1-5%
    k = 64  # 每个样本只激活 64 个特征 (64/6144 ≈ 1%)
    
    # 训练参数
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 10
    
    # 数据参数
    max_length = 256
    num_samples = None
    output_dir = "sae_output_topk"
    
    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"


# ==================== TopK SAE 模型 ====================
class TopKSparseAutoencoder(nn.Module):
    """
    TopK 稀疏自编码器
    强制每个样本只有 K 个特征激活
    """
    
    def __init__(self, input_dim, sae_dim, k):
        super().__init__()
        self.input_dim = input_dim
        self.sae_dim = sae_dim
        self.k = k
        
        # 编码器
        self.encoder = nn.Linear(input_dim, sae_dim)
        
        # 解码器（权重可以与编码器绑定或独立）
        self.decoder = nn.Linear(sae_dim, input_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # 使用较小的初始化，有助于稀疏性
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x):
        """
        编码 + TopK 稀疏化
        """
        # 线性变换
        pre_act = self.encoder(x)  # [batch, sae_dim]
        
        # ReLU 激活
        pre_act = F.relu(pre_act)
        
        # TopK 稀疏化：只保留最大的 K 个值
        topk_values, topk_indices = torch.topk(pre_act, self.k, dim=-1)
        
        # 创建稀疏激活
        sparse_act = torch.zeros_like(pre_act)
        sparse_act.scatter_(-1, topk_indices, topk_values)
        
        return sparse_act
    
    def decode(self, h):
        return self.decoder(h)
    
    def forward(self, x):
        h = self.encode(x)
        recon = self.decode(h)
        return recon, h


# ==================== 激活提取 ====================
def extract_activations(model, tokenizer, texts, layer_idx, config):
    """从模型提取激活"""
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), config.batch_size), desc="提取激活"):
            batch = texts[i:i+config.batch_size]
            
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=config.max_length, return_tensors="pt"
            ).to(config.device)
            
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1][:, 0, :].cpu()
            all_activations.append(hidden)
    
    return torch.cat(all_activations, dim=0)


# ==================== 训练函数 ====================
def train_topk_sae(activations, config):
    """训练 TopK SAE"""
    
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 创建 TopK SAE
    sae = TopKSparseAutoencoder(
        config.hidden_dim, 
        config.sae_dim, 
        config.k
    ).to(config.device)
    
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    
    history = {'loss': [], 'l0': []}
    
    print(f"\n{'='*60}")
    print(f"🚀 Training TopK SAE")
    print(f"   Device: {config.device}")
    print(f"   Dimensions: {config.hidden_dim} -> {config.sae_dim}")
    print(f"   K (max active features): {config.k}")
    print(f"   Target sparsity: {config.k}/{config.sae_dim} = {100*config.k/config.sae_dim:.1f}%")
    print(f"{'='*60}\n")
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        total_l0 = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False)
        for (batch,) in pbar:
            batch = batch.to(config.device)
            
            recon, h = sae(batch)
            
            # 只用重建损失（TopK 已经保证稀疏性）
            loss = F.mse_loss(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_l0 += (h > 0).float().sum(-1).mean().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        n = len(dataloader)
        avg_loss = total_loss / n
        avg_l0 = total_l0 / n
        
        history['loss'].append(avg_loss)
        history['l0'].append(avg_l0)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}: "
              f"Loss={avg_loss:.6f}, L0={avg_l0:.1f}/{config.sae_dim} "
              f"({100*avg_l0/config.sae_dim:.1f}%)")
    
    return sae, history


# ==================== 特征分析 ====================
def analyze_features(sae, activations, texts, labels, config, top_k=10):
    """分析学到的特征"""
    
    sae.eval()
    with torch.no_grad():
        h = sae.encode(activations.to(config.device)).cpu()
    
    labels_t = torch.tensor(labels)
    
    # 计算特征在正负样本上的激活差异
    pos_mean = h[labels_t == 1].mean(0)
    neg_mean = h[labels_t == 0].mean(0)
    diff = pos_mean - neg_mean
    
    # 最相关特征
    top_pos = torch.topk(diff, top_k).indices.tolist()
    top_neg = torch.topk(-diff, top_k).indices.tolist()
    
    print("\n" + "="*60)
    print("📊 与【刻板印象】最相关的特征 (正向)")
    print("="*60)
    
    for idx in top_pos[:5]:
        feat_act = h[:, idx]
        top_samples = torch.topk(feat_act, 5).indices.tolist()
        print(f"\n🔸 Feature #{idx} (diff: {diff[idx]:.4f})")
        for i, s_idx in enumerate(top_samples[:3]):
            print(f"   [{feat_act[s_idx]:.3f}] {texts[s_idx][:50]}...")
    
    print("\n" + "="*60)
    print("📊 与【非刻板印象】最相关的特征 (负向)")
    print("="*60)
    
    for idx in top_neg[:5]:
        feat_act = h[:, idx]
        top_samples = torch.topk(feat_act, 5).indices.tolist()
        print(f"\n🔹 Feature #{idx} (diff: {diff[idx]:.4f})")
        for i, s_idx in enumerate(top_samples[:3]):
            print(f"   [{feat_act[s_idx]:.3f}] {texts[s_idx][:50]}...")
    
    return {'pos_features': top_pos, 'neg_features': top_neg, 'diff_scores': diff}


# ==================== 主函数 ====================
def main():
    config = SAEConfig()
    
    print("="*70)
    print("🧠 TopK SAE Training for Chinese ALBERT")
    print("="*70)
    
    # 设备信息
    print(f"\n📱 Device: {config.device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载模型
    print(f"\n📦 Loading model: {config.model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = model.to(config.device)
    print("   ✅ Model loaded")
    
    # 加载数据
    print(f"\n📂 Loading data: {config.data_path}")
    df = pd.read_csv(config.data_path, usecols=['text', 'label'])
    label_map = {'stereotype': 1, 'non-stereotype': 0, 'neutral': 0, 'unrelated': 0}
    df['label_binary'] = df['label'].map(lambda x: label_map.get(x, 0))
    
    if config.num_samples:
        df = df.sample(n=config.num_samples, random_state=42)
    
    texts = df['text'].tolist()
    labels = df['label_binary'].tolist()
    print(f"   ✅ Loaded {len(texts)} samples")
    print(f"   Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # 提取激活
    print(f"\n🔍 Extracting layer {config.layer_idx} activations...")
    activations = extract_activations(model, tokenizer, texts, config.layer_idx, config)
    print(f"   ✅ Shape: {activations.shape}")
    
    # 训练 TopK SAE
    print(f"\n🏋️ Training TopK SAE (K={config.k})...")
    sae, history = train_topk_sae(activations, config)
    
    # 保存
    os.makedirs(config.output_dir, exist_ok=True)
    torch.save(sae.state_dict(), f"{config.output_dir}/sae_topk_weights.pt")
    
    save_config = {
        'hidden_dim': config.hidden_dim,
        'sae_dim': config.sae_dim,
        'k': config.k,
        'layer_idx': config.layer_idx,
    }
    with open(f"{config.output_dir}/config.json", 'w') as f:
        json.dump(save_config, f, indent=2)
    
    with open(f"{config.output_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n💾 Saved to: {config.output_dir}/")
    
    # 分析特征
    print(f"\n🔬 Analyzing features...")
    analysis = analyze_features(sae, activations, texts, labels, config)
    
    # 保存分析
    analysis_save = {
        'pos_features': analysis['pos_features'],
        'neg_features': analysis['neg_features'],
    }
    with open(f"{config.output_dir}/feature_analysis.json", 'w') as f:
        json.dump(analysis_save, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TopK SAE Training Complete!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   - Final Loss: {history['loss'][-1]:.6f}")
    print(f"   - L0: {history['l0'][-1]:.1f}/{config.sae_dim} ({100*history['l0'][-1]/config.sae_dim:.1f}%)")
    print(f"   - K (target): {config.k} ({100*config.k/config.sae_dim:.1f}%)")


if __name__ == "__main__":
    main()
