# Sentiment-Explorer-Persian-Comment-Classification

This project involves the design and implementation of a sentiment classification pipeline for Persian-language product reviews. It combines data preprocessing, word embedding using Word2Vec, and classification using Logistic Regression.

---

## 📁 Dataset

Two files were provided:

- `train.csv`: Contains Persian reviews with corresponding sentiment labels (`recommendation_status`)
- `test.csv`: Contains new reviews for which predictions must be made

Both files are stored in the `data/` directory.

---

## ⚙️ Project Pipeline

The overall workflow includes the following stages:

### 1. Data Import
The dataset files were read into the programming environment using `pandas`:

```python
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
