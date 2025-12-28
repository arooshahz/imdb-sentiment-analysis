# 🎬 IMDB Sentiment Analysis using Transformers

## Project Overview
This project presents an end-to-end Natural Language Processing pipeline for sentiment analysis on the IMDB movie reviews dataset. The goal is to classify reviews as positive or negative using both classical machine learning models and Transformer-based architectures.

---

## Dataset
- **IMDB Movie Reviews Dataset**
- 50,000 labeled reviews
- Binary sentiment labels (positive / negative)
- Balanced class distribution

---

## Exploratory Data Analysis (EDA)
- Analysis of review length distribution
- Class balance verification
- Word frequency analysis by sentiment
- Identification of noisy patterns and outliers

---

## Models
### Baseline Models
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM

### Transformer Model
- Fine-tuned `bert-base-uncased`
- Max sequence length: 256
- Trained using HuggingFace Trainer API

---

## Results

| Model | Accuracy | F1-score |
|------|----------|----------|
| Logistic Regression | 0.89300 | 0.894789 |
| Linear SVM | 0.88580 | 0.886639 |
| BERT (Fine-tuned) | **0.92384** | **0.924773** |

---

## Error Analysis
- Analysis of misclassified samples
- Impact of long reviews and truncation
- Challenges with negation and mixed sentiment
- Confusion matrix visualization

---

## Tech Stack
- Python
- HuggingFace Transformers & Datasets
- PyTorch
- Scikit-learn
- Matplotlib 


