import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch # 引入 torch 用于检测 GPU

def predict_and_save(input_csv, model, tokenizer, device):
    try:
        data = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_csv}，跳过。")
        return

    # 检查是否存在 'text' 列
    if 'text' not in data.columns:
        print(f"错误：文件 {input_csv} 中缺少 'text' 列，跳过。")
        return
    
    texts = data['text'].tolist()
    
    # 初始化 pipeline，传入 device 参数以利用 GPU (device=0) 或 CPU (device=-1)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    
    # 批量推理
    results = classifier(texts)
    
    data['prediction'] = [result['label'] for result in results]
    
    # 尝试只保存需要的列，如果 group 列不存在则不保存它
    cols_to_save = ['text', 'prediction']
    if 'group' in data.columns:
        cols_to_save.insert(1, 'group')
        
    output_data = data[cols_to_save]  
    
    output_file = input_csv.replace('.csv', '_predictions.csv')
    output_data.to_csv(output_file, index=False)
    print(f"预测结果已保存至 {output_file}")

def main():
    files_to_process = [
        'Claude-2 Outputs.csv', 'Claude-3.5-Sonnet Outputs.csv', 'Claude-3-Sonnet Outputs.csv', 
        'Gemini-1.0-Pro Outputs.csv', 'Gemini-1.5-Pro Outputs.csv', 'GPT-3.5-Turbo Outputs.csv',
        'GPT-4o Outputs.csv', 'GPT-4-Turbo Outputs.csv', 'Llama-3-70B-T Outputs.csv', 
        'Llama-3.1-405B-T Outputs.csv', 'Mistral Large 2 Outputs.csv', 'Mistral Medium Outputs.csv'
    ]
    
    # --- 修改点：指定本地模型路径 ---
    model_path = "/content/ChineseHeart/Model Training and Evaluation/model_output_albertv2/mgsd_trained"
    
    print(f"正在加载本地模型：{model_path} ...")
    
    try:
        # 从本地路径加载模型和分词器
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError as e:
        print(f"加载模型失败，请检查路径是否正确：{e}")
        return

    # 检测是否有 GPU 可用 (CUDA)，如果有则 device=0，否则 device=-1 (CPU)
    device = 0 if torch.cuda.is_available() else -1
    print(f"推理运行设备: {'GPU' if device == 0 else 'CPU'}")
    
    for filename in files_to_process:
        print(f"正在处理 {filename}...")
        predict_and_save(filename, model, tokenizer, device)

if __name__ == "__main__":
    main()
