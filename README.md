# Persian Sentiment Classification: Sentiment Explorer

A comprehensive project aimed at building a sentiment classification pipeline for Persian-language product reviews using modern natural language processing techniques and traditional machine learning algorithms.

---

## Overview

This project was designed to process and classify Persian comments based on sentiment orientation (recommendation vs. non-recommendation). The task involved several key stages:

- Preprocessing Persian text data
- Training a Word2Vec model to create word embeddings
- Constructing sentence embeddings by averaging word vectors
- Using a Logistic Regression classifier to predict sentiment
- Generating structured outputs for evaluation

This report documents the design rationale, development process, and insights gathered throughout the implementation.

---

## Dataset Description

Two datasets were provided:

- `train.csv`: Contains Persian comments with a column called `recommendation_status` indicating the sentiment label.
- `test.csv`: Contains new comments with no labels; the goal is to predict their sentiment class.

The datasets had already undergone initial cleaning. There were no missing values or noisy entries requiring removal. However, numerical encoding and detailed analysis of the text structure were necessary.

---

## Data Preprocessing

Preprocessing Persian-language text presents unique challenges:

### Key Steps Implemented:

1. **Text Normalization**: Converted different forms of Persian letters to a standard format (e.g., Arabic "ي" to Persian "ی").
2. **Tokenization**: Split sentences into words.
3. **Digit Removal**: Removed both Persian (e.g., ۱۲۳) and Latin (123) digits.
4. **Punctuation Removal**: Eliminated symbols such as `!`, `؟`, `،`, etc.
5. **Stopword Removal**: Removed frequently occurring, semantically light words (e.g., "که", "از", "برای").
6. **Stemming**: Reduced words to their root forms using Persian language rules.
7. **Whitespace Cleaning**: Removed excessive spaces and line breaks.

All steps were encapsulated in a single preprocessing function that could be applied to any Persian sentence.

---

## Word Embedding with Word2Vec

To represent words numerically, a custom **Word2Vec** model was trained on the preprocessed comments. Word2Vec enabled the project to capture semantic relationships between words by mapping them into a continuous vector space.

Each sentence was converted into a fixed-size vector by averaging its word vectors — a technique known as **sentence embedding by mean pooling**.

---

## Sentiment Classification Model

### Model Used:
- **Logistic Regression** — a simple yet effective linear classifier for binary classification tasks.

### Data Split:
- 80% of the training set was used to train the model.
- 20% was used as a validation set to assess model performance.

### Evaluation Metric:
- **Accuracy** — the percentage of correct predictions on the validation set.
- The model achieved accuracy well above the minimum acceptable threshold of **50%**.

---

## Prediction Function

A general-purpose function was developed to classify new comments using the trained pipeline. It:

1. Preprocesses the comment text.
2. Converts it to a sentence vector.
3. Feeds it to the trained classifier.
4. Returns a label:  
   - `recommended`  
   - `not_recommended`  
   - `no_idea` (fallback for ambiguous input)

---

## Test Set Inference & Submission

Predictions were made on the `test.csv` dataset. Each entry was classified and stored in a new DataFrame with the following structure:

| class           |
|-----------------|
| not_recommended |
| recommended     |
| ...             |

This DataFrame was saved as a CSV and archived as `result.zip` for final evaluation and submission.

---

## Summary of Achievements

- ✅ Successfully cleaned and normalized Persian-language comments
- ✅ Trained Word2Vec model to capture semantic similarity
- ✅ Created an end-to-end sentiment classification pipeline
- ✅ Reached high classification accuracy on validation data
- ✅ Generated standardized output for evaluation

---

## Future Work

While the current pipeline performs reliably, the following enhancements could yield stronger results:

- 📌 **Switch to transformer-based models** such as ParsBERT or multilingual BERT for contextual embeddings
- 📌 **Use of FastText** for better handling of out-of-vocabulary words in Persian
- 📌 **Add explainability layer** for model predictions (e.g., LIME, SHAP)
- 📌 **Hyperparameter tuning** using cross-validation or grid search
- 📌 **Multi-class sentiment support** (positive, neutral, negative) for finer-grained analysis

---

## Author

This project was developed as part of a structured machine learning assignment focused on natural language processing with a concentration in Persian text mining.

---

## References

- Gensim Word2Vec Documentation  
- scikit-learn API Reference  
- Hazm (Python toolkit for Persian NLP)  
- fastText by Facebook AI  
- ParsBERT: A Transformer-based Model for Persian Language Understanding

