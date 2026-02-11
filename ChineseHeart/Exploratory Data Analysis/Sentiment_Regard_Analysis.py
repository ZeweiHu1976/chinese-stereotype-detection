import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 1. 优化后的加载函数
def load_data(filepath):
    # 指定编码以防中文乱码，常用 'utf-8' 或 'utf-8-sig'
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f"成功加载数据集，包含 {df.shape[0]} 条观测，列名: {df.columns.tolist()}")
    # 清理：确保 text 列没有空值
    df['text'] = df['text'].astype(str).fillna("")
    return df

# 2. 更加通用的分类器基类（增加显卡支持）
class BaseClassifier:
    def __init__(self, model_name):
        # 自动检测是否存在 GPU
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            truncation=True, 
            device=self.device
        )

    def compute_batch(self, texts, batch_size=32):
        # 直接传入整个列表进行批量预测
        results = self.pipe(texts, batch_size=batch_size)
        return [res['label'] for res in results]

# 3. 执行分析的主函数
def analyse_sentiment_and_regard(df, text_col):
    print("正在初始化模型...")
    sentiment_clf = BaseClassifier("cardiffnlp/twitter-roberta-base-sentiment-latest")
    regard_clf = BaseClassifier("sasha/regardv3")

    texts = df[text_col].tolist()

    print(f"正在分析情感 (Sentiment)...")
    df['Sentiment'] = sentiment_clf.compute_batch(texts)

    print(f"正在分析关注度 (Regard)...")
    df['Regard'] = regard_clf.compute_batch(texts)

    return df

# --- 执行流程 ---
if __name__ == "__main__":
    # 加载 cold.csv
    # 确保文件路径正确
    df_mgsd = load_data("cold.csv")
    
    # 进行分析
    # 这里的 text_col 对应你文件里的 'text' 列
    results = analyse_sentiment_and_regard(df_mgsd, text_col='text')
    
    # 保存结果
    output_path = "cold_analysed_results.csv"
    results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"分析完成！结果已保存至: {output_path}")

