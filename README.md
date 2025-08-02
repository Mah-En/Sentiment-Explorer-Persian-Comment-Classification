# Sentiment-Explorer-Persian-Comment-Classification


This report details the "Sentiment Explorer" project, which focused on the preprocessing of Persian text and the development of a sentiment classification model.

---

#### **1. Project Overview**

[cite_start]The primary objective of this project was to prepare a Persian text dataset for sentiment classification and to train a model capable of categorizing comments as `recommended`, `not_recommended`, or `no_idea`[cite: 1, 2]. [cite_start]This project served as a practical application of data analysis, text preprocessing, and machine learning skills[cite: 2].

---

#### **2. Project Phases**

[cite_start]The project was executed in several key stages[cite: 2]:

* **Analysis of the Training Dataset:**
    * [cite_start]Data Cleaning: The datasets were already clean, with no redundant or irrelevant entries[cite: 2].
    * [cite_start]Dataset Preprocessing: Categorical sentiment labels were converted into numerical values to be used by machine learning models[cite: 2]. [cite_start]The labels `not_recommended`, `recommended`, and `no_idea` were mapped to 0, 1, and 2, respectively[cite: 2]. [cite_start]The `train.csv` file contained 149,400 entries, with 49,800 entries for each category[cite: 2]. [cite_start]The `test.csv` file contained 600 entries[cite: 2].

* **Text Preprocessing:**
    * [cite_start]A function named `preprocess_text` was created to perform several text preprocessing steps for the Persian language[cite: 2].
    * [cite_start]This function included normalization, tokenization, number removal (both Persian and Latin), punctuation removal, stemming, and stopword removal[cite: 2].
    * [cite_start]An example of the function's output shows that an input like "من متولد سال ۱۳۷۷ هستم" would be processed into `['متولد', 'سال', 'هس']`[cite: 2].

* **Word Embedding:**
    * [cite_start]The `Word2Vec` model from the `gensim` library was used to convert preprocessed words into numerical vector representations[cite: 2].
    * [cite_start]The model was configured with a vector size of 100, a window of 5, and a minimum word count of 1[cite: 2].
    * [cite_start]A `sentence_vector` function was designed to compute the average of word vectors for each review, creating a single vector representation for the entire sentence[cite: 2].

* **Sentiment Classification Model Training:**
    * [cite_start]The dataset was split into training and validation sets, with 80% of the data used for training and 20% for evaluation[cite: 2].
    * [cite_start]The `Logistic Regression` algorithm was chosen for training the sentiment classifier[cite: 2].

* **Model Evaluation:**
    * [cite_start]The trained model's performance was evaluated using the `accuracy_score` metric on the validation dataset[cite: 2].
    * [cite_start]The model achieved an accuracy of approximately 67.13\%[cite: 2].

* **Final Predictions and Submission:**
    * [cite_start]A `predict_recommendation` function was created to predict the sentiment of new comments[cite: 2].
    * [cite_start]This function was used to predict sentiment labels for the `test.csv` dataset[cite: 2].
    * [cite_start]The final predictions were stored in a `submission.csv` file with a single column, `class`, containing the predicted sentiment labels (`recommended`, `not_recommended`, or `no_idea`)[cite: 2]. [cite_start]The final project files were packaged into a `result.zip` for submission[cite: 2].
