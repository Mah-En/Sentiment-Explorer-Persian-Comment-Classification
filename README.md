### **Project Report: Sentiment Explorer - A Deep Dive into Persian Comment Classification**

This report provides a comprehensive overview of the "Sentiment Explorer" project, which focused on the end-to-end process of preprocessing Persian text data and building a sentiment classification model. The project demonstrates a robust pipeline for handling unstructured text data, from raw input to final predictions.

---

### **1. Project Overview and Objectives**

The primary objective of the "Sentiment Explorer" project was to preprocess a Persian text dataset and use it to train a sentiment classification model. [cite_start]The model's task was to categorize customer comments into one of three classes: `recommended`, `not_recommended`, or `no_idea`[cite: 1]. [cite_start]This project was an excellent opportunity to apply and refine skills in data analysis, preprocessing, and machine learning, with a strong focus on ensuring the quality and robustness of each step[cite: 1].

The project's success was fundamentally tied to the quality of the data preparation and the efficacy of the chosen machine learning model. [cite_start]The main phases of this project were designed to create a solid foundation for the classification task, starting from the raw data and culminating in a final set of predictions[cite: 1].

---

### **2. Project Phases: A Detailed Breakdown**

The project was structured into several distinct, yet interconnected, phases:

#### **A. Analysis and Preprocessing of the Training Dataset**

* [cite_start]**Initial Data Loading:** The project began by loading the `train.csv` and `test.csv` files into a programming environment[cite: 1]. [cite_start]The datasets were located in a `data` directory[cite: 1].
* [cite_start]**Data Integrity:** The provided datasets had already undergone an initial cleaning process, ensuring there were no redundant or irrelevant entries[cite: 1].
* [cite_start]**Data Structure:** Initial inspection revealed that the `train.csv` file contained 149,400 entries across two columns: `body` (containing the comment text) and `recommendation_status` (the sentiment label)[cite: 1]. [cite_start]The `test.csv` file contained 600 entries with only the `body` column[cite: 1].
* [cite_start]**Target Label Distribution:** An analysis of the `recommendation_status` column in the training data showed a perfectly balanced distribution, with 49,800 entries for each of the three classes: `not_recommended`, `recommended`, and `no_idea`[cite: 1].
* [cite_start]**Categorical to Numerical Conversion:** To prepare the data for machine learning algorithms, which require numerical input, the categorical sentiment labels were mapped to integer values[cite: 1]. [cite_start]Specifically, `not_recommended` was mapped to `0`, `recommended` to `1`, and `no_idea` to `2`[cite: 1].

#### **B. Comprehensive Text Preprocessing**

This was a critical phase for ensuring the quality of the data fed into the model. [cite_start]A custom function, `preprocess_text`, was implemented to handle the specific linguistic challenges of the Persian language[cite: 1]. This function included a series of systematic steps:

* [cite_start]**Normalization:** The text was normalized to a standard format[cite: 1].
* [cite_start]**Punctuation and Number Removal:** All punctuation marks (e.g., `?`, `!`) and both Persian and Latin numerals were removed from the text[cite: 1].
* [cite_start]**Tokenization:** The text was split into individual words or tokens[cite: 1].
* [cite_start]**Stemming:** Words were reduced to their root forms to group similar words together and reduce the vocabulary size[cite: 1].
* **Stopword Removal:** While this step was mentioned in the project outline, the provided code snippet for `preprocess_text` did not explicitly show stopword removal, indicating it may have been an intended but un-implemented feature.
* [cite_start]**Extra Space Removal:** The pipeline was designed to ensure consistent spacing between words and sentences[cite: 1].
* [cite_start]**Application to Dataset:** The `preprocess_text` function was applied to the entire `body` column of the `train_data` DataFrame, and the results were stored in a new column named `preprocess`[cite: 1].

#### **C. Word Embedding with Word2Vec**

After preprocessing, the text data needed to be converted into a numerical format. [cite_start]This was achieved using the `Word2Vec` algorithm[cite: 1].

* [cite_start]**Model Training:** The `Word2Vec` model from the `gensim` library was trained on the preprocessed text data (`train_data['preprocess']`)[cite: 1]. [cite_start]The model was configured with `vector_size=100`, `window=5`, `min_count=1`, and `workers=4`[cite: 1].
* [cite_start]**Model Validation:** The trained model was tested by finding the words most similar to the word "دوست" ("friend")[cite: 1]. [cite_start]The results showed that the model was able to identify semantically similar words[cite: 1].
* [cite_start]**Sentence Vectorization:** A `sentence_vector` function was developed to create a single vector for each review[cite: 1]. [cite_start]This function computed the average of the `Word2Vec` vectors for all the words in a given sentence, providing a unified numerical representation for the entire comment[cite: 1].

#### **D. Model Training and Evaluation**

With the data fully prepared, the next phase was to train and evaluate a machine learning model.

* [cite_start]**Data Splitting:** The processed data was split into training and evaluation sets[cite: 1]. [cite_start]`80%` of the data was allocated for training and `20%` for evaluation[cite: 1]. [cite_start]`X` represented the sentence vectors and `y` contained the numerical sentiment labels[cite: 1].
* [cite_start]**Model Selection:** `Logistic Regression` was selected as the classification algorithm for this project[cite: 1].
* [cite_start]**Training:** The `LogisticRegression` model was trained using the `X_train` and `y_train` subsets[cite: 1].
* [cite_start]**Evaluation:** The model's performance was evaluated using the `accuracy_score` metric on the `X_test` and `y_test` validation sets[cite: 1]. [cite_start]The model achieved an accuracy of approximately **67.13%**[cite: 1].

#### **E. Final Predictions and Submission**

The final phase involved applying the trained model to the test data and preparing the submission files.

* [cite_start]**Prediction Function:** A `predict_recommendation` function was created[cite: 1]. [cite_start]This function takes a raw comment as input, preprocesses it, converts it into a sentence vector, and then uses the trained `Logistic Regression` model to predict the sentiment class[cite: 1]. [cite_start]The function returns one of the three sentiment strings: `recommended`, `not_recommended`, or `no_idea`[cite: 1].
* [cite_start]**Test Data Prediction:** The `predict_recommendation` function was applied to all comments in the `test.csv` dataset[cite: 1].
* [cite_start]**Submission File Generation:** The final predictions were compiled into a `submission.csv` file, which contained a single column named `class`[cite: 1]. [cite_start]This file, along with the project notebook, was packaged into a `result.zip` file for submission[cite: 1].

---

### **3. Conclusion**

The "Sentiment Explorer" project successfully demonstrated a complete workflow for Persian sentiment analysis. By meticulously implementing each phase—from text preprocessing and word embedding to model training and evaluation—the project achieved its core objectives. The `Logistic Regression` model's accuracy of `67.13%` validates the effectiveness of the prepared data and the overall pipeline. This project provides a solid foundation for more complex natural language processing tasks in the Persian language, showcasing a robust approach to turning unstructured text data into actionable insights.
