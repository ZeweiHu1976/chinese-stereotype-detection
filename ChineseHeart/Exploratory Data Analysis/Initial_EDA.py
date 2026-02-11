import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with {df.shape[0]} observations.")
    return df

# Target variable distribution (label)
def prepare_target_variable_distribution(df, target_col):
    return df[target_col].value_counts()

# Group distribution
def prepare_group_distribution(df, group_col):
    return df[group_col].value_counts()

# Text length analysis
def prepare_text_length_analysis(df, text_col):
    df[text_col] = df[text_col].fillna('').astype(str)
    df['text_length'] = df[text_col].apply(len)
    length_counts = df['text_length'].value_counts().rename('count_of_texts')
    df = df.join(length_counts, on='text_length')
    return df[['text_length', 'count_of_texts']]
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Word cloud
def create_word_cloud(df, text_col, output_filename):
    text = " ".join(df[text_col].astype(str))
    words = " ".join(jieba.cut(text))

    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        max_words=200,
        max_font_size=250,
        font_path="/content/NotoSansCJK-Regular.ttc"  # ✅ 100% 存在
    ).generate(words)

    plt.figure(figsize=(16, 8), dpi=600)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_filename, dpi=600)
    plt.close()

# ===== Execute EDA for cold.csv =====
df_cold = load_data("cold.csv")

target_dist = prepare_target_variable_distribution(df_cold, target_col='label')
group_dist = prepare_group_distribution(df_cold, group_col='group')
text_length = prepare_text_length_analysis(df_cold, text_col='text')

create_word_cloud(df_cold, text_col='text', output_filename='cold_wordcloud.png')

# Save results
target_dist.to_csv('cold_label_distribution.csv')
group_dist.to_csv('cold_group_distribution.csv')
text_length.to_csv('cold_text_length_analysis.csv')
