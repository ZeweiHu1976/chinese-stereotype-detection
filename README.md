# ChineseHEARTS: Detecting Regional Stereotypes in Chinese Texts

Adapting the HEARTS framework for Chinese regional stereotype detection.

**Author:** Zewei Hu  
**Email:** ucab351@ucl.ac.uk  
**Course:** COMP0189 AI for Sustainable Development, UCL

## Overview

This project adapts the [HEARTS framework](https://arxiv.org/abs/2409.11579) to detect regional stereotypes in Chinese text. Unlike general cultural stereotypes, regional bias in China is rooted in the **Hukou (Household Registration) system**, which creates structural divides between urban and rural populations.

## Original Paper

**HEARTS: A Holistic Framework for Explainable, Sustainable and Robust Text Stereotype Detection**

Key contributions:
1. Introduced EMGSD, a multi-grain stereotype dataset
2. Developed a fine-tuned ALBERT-V2 classifier achieving over 80% accuracy with low carbon footprint
3. Implemented token-level explainability using SHAP and LIME
4. Applied the framework to audit LLMs for bias

## Dataset: CRSD-7k

| Item | Description |
|------|-------------|
| **Size** | 7,296 samples |
| **Source** | Refined from [COLD dataset](https://github.com/thu-coai/COLDataset) |
| **Format** | Stereotype–Neutral–Unrelated triplets |
| **Process** | AI-assisted extraction + human verification |

## Model Performance

### Reproduction Results

| Model | Macro F1 | Note |
|-------|----------|------|
| Original ALBERT (paper) | 81.5% | Baseline |
| Reproduced ALBERT | 78.12% | Within ±5% |

### Chinese Model Comparison

| Model | Macro F1 | Carbon Emissions |
|-------|----------|------------------|
| **RBT6** | **93.54%** | **22% lower** |
| Chinese ALBERT | 91.70% | Baseline |

- Statistical significance: McNemar's χ² = 8.02, p < 0.01
- RBT6 is a distilled Chinese BERT model (6 layers)

## Explainability Analysis

We compared SHAP and LIME explanations to verify model reliability:

### Token Level
| Metric | Value |
|--------|-------|
| JS Divergence (median) | ≈ 0 |
| Cosine Similarity (median) | 0.98 |

### Sentence Level
| Metric | Mean | Median |
|--------|------|--------|
| Cosine Similarity | 0.66 | 0.72 |
| Pearson Correlation | 0.64 | 0.71 |
|  |  | - |

The high consistency between SHAP and LIME indicates strong model reliability.

## SAE Analysis

We used Sparse Autoencoders (SAE) to understand how stereotypes are encoded in the model at a semantic level, revealing connections between stereotype features and specific demographic groups.

## SDG Alignment

- **SDG 10 (Reduced Inequalities):** Detecting regional stereotypes helps identify discrimination against people from specific geographic areas
- **SDG 16 (Peace, Justice & Strong Institutions):** Contributes to reducing online hate speech and promoting inclusive digital communities

## Limitations

- Limited dataset size; unable to conduct province-level ablation studies
- Single annotator may introduce subjective bias
- Model trained on short sentences, limiting performance on longer texts
- Class imbalance: predominantly negative stereotypes

## Future Work

- Explore SAE to identify and deactivate bias-encoding neurons for model debiasing
- Expand dataset with multi-annotator verification
- Test on real-world applications (e.g., recruitment systems)

## Installation

```bash
# Clone the repository
git clone https://github.com/ZeweiHu1976/chinese-stereotype-detection.git
cd chinese-stereotype-detection

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9
- Transformers >= 4.0
- SHAP
- LIME
- CodeCarbon (for carbon tracking)

## Acknowledgements

- Original HEARTS paper authors
- COLD dataset creators
- UCL COMP0189 teaching team

## License

MIT License
