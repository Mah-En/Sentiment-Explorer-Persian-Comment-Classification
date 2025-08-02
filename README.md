# 📊 Sentiment Classification on Persian Text Data

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
````

### 2. Label Encoding

Since machine learning models require numerical inputs, the `recommendation_status` column was mapped to binary values:

```python
label_map = {"recommended": 1, "not_recommended": 0}
train_data["recommendation_status"] = train_data["recommendation_status"].map(label_map)
```

---

## 🧼 Text Preprocessing

A preprocessing function `preprocess_text()` was implemented to normalize and clean Persian text. Steps include:

* Normalization (e.g., handling Persian characters)
* Tokenization
* Removing punctuation
* Removing Persian and Latin digits
* Stopword removal
* Stemming
* Whitespace cleanup

Example:

```python
example = "من متولد سال ۱۳۷۷ هستم"
print(preprocess_text(example))  # Output: ['متولد', 'سال', 'هس']
```

Each review was processed and stored in a new column:

```python
train_data["preprocess"] = train_data["comment"].apply(preprocess_text)
```

---

## 🔎 Word Embedding with Word2Vec

A `Word2Vec` model was trained on the preprocessed tokens:

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences=train_data["preprocess"], vector_size=100, window=5, min_count=1)
```

### Sentence Embedding

Each sentence was represented by the average of its word vectors using the `sentence_vector()` function:

```python
sentence_vectors = train_data["preprocess"].apply(sentence_vector)
X = np.array(sentence_vectors.to_list())
y = train_data["recommendation_status"].values
```

---

## 🧠 Model Training

The dataset was split into training and validation sets (80/20) using `train_test_split`:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

A logistic regression model was trained:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---

## 📈 Evaluation

The model was evaluated using accuracy score:

```python
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)  # Should be > 50%
```

---

## 🔍 Prediction Function

A utility function `predict_recommendation()` was created to handle new inputs:

```python
comment = "این محصول فوق‌العاده بود و واقعاً از خریدم راضی‌ام!"
result = predict_recommendation(comment)
print(result)  # Output: recommended / not_recommended / no_idea
```

Additional examples:

```python
print(predict_recommendation("کیفیت این محصول بسیار پایین بود و اصلاً راضی نیستم."))
print(predict_recommendation("محصول معمولی بود، نه خوب و نه بد."))
```

---

## 🗃️ Final Submission

All predictions on the test set were stored in a new `DataFrame` named `submission`:

```python
submission = pd.DataFrame({"class": test_predictions})
submission.to_csv("result.csv", index=False)
```

The result file was then zipped for submission:

```bash
zip result.zip result.csv
```

Expected structure:

| class            |
| ---------------- |
| not\_recommended |
| not\_recommended |
| recommended      |
| ...              |

---

## ✅ Conclusion

This project demonstrated the complete pipeline of sentiment analysis on Persian text, including:

* Clean preprocessing for right-to-left text
* Use of Word2Vec for custom embeddings
* Logistic regression for baseline classification
* Evaluation and prediction for real-world deployment

**Future directions:**

* Integrate transformer-based models like BERT (ParsBERT)
* Use TF-IDF or FastText for enhanced embeddings
* Improve neutral class prediction with confidence thresholds

---

## 🧑‍💻 Author

Developed and documented as part of a natural language processing project on Persian-language sentiment classification.

```

---

Let me know if you'd like this in a downloadable `README.md` file or a rendered preview.
```
