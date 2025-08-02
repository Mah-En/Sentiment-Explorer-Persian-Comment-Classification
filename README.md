<div dir="center">
  <h1 align="center" style="line-height:200%;font-family:vazir;color:#0099cc">
    <font face="vazir" color="#0099cc">
      Sentiment Explorer
    </font>
  </h1>
</div>

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  In this exciting project, my task was to preprocess Persian text data and prepare it for training a sentiment classification model. This project served as an exceptional opportunity to challenge my skills and put into practice the knowledge I had acquired so far. The success of the project depended heavily on the quality of the prepared data, and I was determined to ensure that the preprocessing phase was handled with precision and care.
</p>

------

<h1 dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  <font face="Times New Roman" color="#0099cc" size="4">
    Project Phases:
  </font>
</h1>

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  The project was carried out in three main stages:
  <div dir="ltr">
    <ul>
        <li><strong>Analysis of the Training Dataset:</strong>
        <ul>
          <li>Data Cleaning: Removal of redundant or irrelevant entries.</li>
          <li>Dataset Preprocessing: Converting categorical sentiment labels into numerical values.</li>
        </ul>
      <li><strong>Text Preprocessing:</strong>
        <ul>
          <li>Normalization: Applying Persian text normalization techniques.</li>
          <li>Tokenization: Splitting text into words or sentences.</li>
          <li>Number Removal: Eliminating both Persian and Latin numerals.</li>
          <li>Punctuation Removal: Removing punctuation marks such as <code>?</code>, <code>!</code>, etc.</li>
          <li>Stopword Removal: Filtering out high-frequency, low-importance words.</li>
          <li>Stemming: Reducing words to their root forms.</li>
          <li>Extra Space Removal: Ensuring consistent word and sentence spacing.</li>
        </ul>
      </li>
      <li><strong>Word Embedding:</strong>
        <ul>
          <li>Using <code>Word2Vec</code> to transform words into numerical vector representations.</li>
          <li>Analyzing word similarity based on embeddings.</li>
        </ul>
      </li>
      <li><strong>Preparation for Analysis:</strong>
        <ul>
          <li>Storing preprocessed data and embeddings for later use in machine learning models.</li>
        </ul>
      </li>
      <li><strong>Sentiment Classification Model Training:</strong>
        <ul>
          <li>Data Splitting: Dividing the dataset into <code>Train</code> and <code>Validation</code> subsets.</li>
          <li>Model Training: Using Logistic Regression to train the sentiment classifier.</li>
          <li>Model Evaluation: Assessing performance on the validation dataset.</li>
        </ul>
      </li>
      <li><strong>Project Evaluation File Preparation:</strong>
        <ul>
          <li>Prediction on <code>Test</code> data: Predicting sentiment labels for the test dataset.</li>
          <li>Creating the final evaluation file for submission.</li>
        </ul>
      </li>
    </ul>
  </div>
</p>

-----------

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  <font face="Times New Roman" size="3">
    Due to the need for quality assurance and considering my current proficiency level in <strong>data analysis</strong>, <strong>data preprocessing</strong>, and <strong>machine learning</strong>, certain core stages of the project were handled by the project team. These included:
    <div dir="ltr">
      <ul>
        <li><strong>Analysis of the Training Dataset:</strong> Performing initial data exploration and extracting relevant insights.</li>
        <li><strong>Preparation for Analysis:</strong> Establishing an optimal framework for efficient data utilization.</li>
        <li><strong>Sentiment Classification Model Training:</strong> Implementing and optimizing the machine learning model.</li>
        <li><strong>Evaluation File Preparation:</strong> Compiling the final evaluation report for performance assessment.</li>
      </ul>
    </div>
  </font>
</p>

----------------

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  Now it was time to demonstrate my skills and successfully overcome this challenge. I approached the project with diligence and determination to meet the required standards, and I am confident in the robustness of the prepared dataset and preprocessing pipeline.
</p>


<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  To begin the work, I placed the two dataset files, <code>train.csv</code> and <code>test.csv</code>, in the <code>data</code> directory. As the first step, I imported these files into my programming environment using the following commands:
</p>


```python
import pandas as pd

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  The datasets provided to me had already undergone a cleaning process, ensuring that no redundant or irrelevant entries were present. However, in order to gain a better understanding of the dataset, I decided to extract some useful insights using the following commands. These insights included the distribution of comments that recommended a purchase versus those that did not, as well as the overall structure of the dataset.<br>
  This information helped me develop a clearer understanding of the data and prepared me for the subsequent stages of the project.
</p>


```python
train_data.info()
```

```python
test_data.info()
```

```python
train_data['recommendation_status'].value_counts()
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  Upon examining the data in the <code>recommendation_status</code> column, I found that its current format was not suitable for training machine learning models, as these algorithms require numerical input. Therefore, the next step was to convert the values in this column into numerical form, specifically <code>0</code> and <code>1</code>, so they could be utilized effectively by the models.  
  To accomplish this transformation, I used the following commands:
</p>


```python
train_data["recommendation_status"] = train_data["recommendation_status"].map({"no_idea": 2,"recommended": 1, "not_recommended": 0})
```

```python
train_data["recommendation_status"].unique()
```

```python
train_data["recommendation_status"].value_counts()
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  At this stage, it was my turn to take action. Using the knowledge and tools I had learned so far, I implemented the required preprocessing steps for the Persian language in the form of a function named <code>preprocess_text</code>.  
  This function was designed to perform all preprocessing operations mentioned at the beginning of the project, including text normalization, tokenization, stopword removal, stemming, and other necessary steps.  
  The <code>preprocess_text</code> function could then be applied to any text input, enabling me to preprocess my dataset effectively for the subsequent stages of the project.  
  Below is an example demonstrating the expected behavior of this function:
</p>

**Input:**
<pre>
  <code>
example = "من متولد سال ۱۳۷۷ هستم"
preprocess_text(example)
  </code>
</pre>

**Output:**
<pre>
  <code>
['متولد', 'سال', 'هس']
  </code>
</pre>


```python
!pip install hazm
```

```python
import re
import string
import hazm
from hazm import Normalizer, Stemmer, Lemmatizer, word_tokenize
normalizer = Normalizer()
stemmer = Stemmer()
lemmatizer = Lemmatizer()


```

```python
def preprocess_text(text):
    text = normalizer.normalize(text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  After completing the implementation of the preprocessing function, the next step was to apply this operation to all the reviews stored in the <code>train_data</code> dataset so that they would be ready for reference by the <code>Word2Vec</code> model.  
  The preprocessed data was then stored in a new column named <code>preprocess</code> for future use.
</p>


```python
dataes = train_data['body']
```

```python
data_processed = dataes.apply(preprocess_text)
```

```python
train_data["preprocess"] = data_processed
train_data.head()
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  With the preprocessing applied to the data and the results stored, the next step was to perform word embedding using the <code>Word2Vec</code> algorithm.  
  In this stage, I implemented the <code>Word2Vec</code> model to train on the preprocessed dataset and convert each word into its corresponding numerical vector representation.
</p>


```python
from gensim.models import Word2Vec
model = Word2Vec(sentences=train_data['preprocess'], vector_size=100, window=5, min_count=1, workers=4)
```

```python
print(list(model.wv.key_to_index.keys())[:20])  # First 20 words
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  Next, I tested the trained <code>Word2Vec</code> model to find the words most similar to the word "دوست" ("friend").  
  I then examined and analyzed the resulting output to assess the model’s ability to capture semantic similarities between words.
</p>


```python
model.wv.most_similar("دوست")
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  At this stage, I designed the <code>sentence_vector</code> function to compute the embedding vector for each review.  
  The purpose of this function is to calculate the average of the word vectors within a review and generate a single unified vector representing the entire sentence.
</p>


```python
import numpy as np
```

```python
def sentence_vector(sentence):
    vectors = []
    for word in sentence:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            vectors.append(np.zeros(100))  
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)

```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  Now that the <code>sentence_vector</code> function has been defined, I applied it to the <code>train_data['preprocess']</code> column.  
  The resulting sentence-level vectors were stored in a variable named <code>sentence_vectors</code>.  
  At this point, each review was successfully transformed into a corresponding vector, ready to be fed into the classification model.
</p>

<div dir="ltr" style="text-align: left;">
  <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 14px; font-family: monospace; overflow: auto;">
# Applying the sentence_vector function to the preprocessed data
sentence_vectors = train_data['preprocess'].apply(sentence_vector)
  </pre>
</div>


```python
sentence_vectors = train_data['preprocess'].apply(sentence_vector)
sentence_vectors
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  In this step, I used the <code>train_test_split</code> function to divide the data into training and evaluation sets.  
  The data was split in such a way that 80% was allocated for training and 20% for evaluation.  
  Here, <code>X</code> represents the embedded sentence vectors for each review, and <code>y</code> contains the target labels, which correspond to the <code>recommendation_status</code> column.
</p>

<div dir="ltr" style="text-align: left;">
  <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 14px; font-family: monospace; overflow: auto;">
from sklearn.model_selection import train_test_split
# Converting sentence vectors to a NumPy array
X = np.array(sentence_vectors.to_list())
# Assuming df["recommendation_status"] contains the target labels
y = df["recommendation_status"].values
# Splitting the data into training and evaluation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  </pre>
</div>


```python
from sklearn.model_selection import train_test_split

X = np.array(sentence_vectors.to_list())

y = train_data["recommendation_status"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  After preparing and splitting the dataset into training and evaluation sets, the next step was to train the model.  
  In this project, I used the <i>Logistic Regression</i> algorithm for sentiment classification.  
  The model was trained using the <code>fit</code> method, applied to the training data <code>X_train</code> and <code>y_train</code>.
</p>

<div dir="ltr" style="text-align: left;">
  <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 14px; font-family: monospace; overflow: auto;">
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
logistic_model.fit(X_train, y_train)
  </pre>
</div>


```python

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  After training the model, the next step was to evaluate its performance.  
  In this stage, I used the evaluation data <code>X_test</code> to generate predictions, and then measured the model’s accuracy using the <code>accuracy_score</code> function.  
  Finally, the model's accuracy was printed. The minimum acceptable threshold for accuracy was set at 50%.
</p>

<div dir="ltr" style="text-align: left;">
  <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 14px; font-family: monospace; overflow: auto;">
from sklearn.metrics import accuracy_score
# Generating predictions using the evaluation data
y_pred = logistic_model.predict(X_test)
# Evaluating model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
  </pre>
</div>


```python
from sklearn.metrics import accuracy_score

y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium;">  
  In this section, I used the <code>predict_recommendation</code> function to classify new reviews.  
  This function processes the input review, converts it into an embedding vector, and then uses the trained model to predict the class of the review.  
  It returns one of the following three outcomes:  
  <ul>  
    <li><code>recommended</code></li>  
    <li><code>not_recommended</code></li>  
    <li><code>no_idea</code></li>  
  </ul>  
</p>  

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium;">  
  To accomplish this task, I followed the steps below:  
</p>  

<ol dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium;">  
  <li>Defined a new review. For example:  
</ol>  

<div dir="ltr" style="text-align: left;">  
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 14px; font-family: monospace; overflow: auto;">  
comment = "این محصول فوق‌العاده بود و واقعاً از خریدم راضی‌ام!"  
</pre>  
</div>  

<ol dir="ltr" start="2" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium;">  
  <li>Called the <code>predict_recommendation</code> function and stored the result:  
</ol>  

<div dir="ltr" style="text-align: left;">  
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 14px; font-family: monospace; overflow: auto;">  
result = predict_recommendation(comment)  
print(result)  
</pre>  
</div>  

<ol dir="ltr" start="3" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium;">  
  <li>Tested several other reviews with different sentiments (positive, negative, and neutral), and evaluated the model's output. For instance:  
</ol>  

<div dir="ltr" style="text-align: left;">  
<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 14px; font-family: monospace; overflow: auto;">  
print(predict_recommendation("کیفیت این محصول بسیار پایین بود و اصلاً راضی نیستم."))  
print(predict_recommendation("محصول معمولی بود، نه خوب و نه بد."))  
</pre>  
</div>  

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium;">  
  My objective here was to assess the model’s performance by testing various review inputs and analyzing the differences in predictions.  
  By modifying the review texts, I was also able to gain a deeper understanding of how natural language processing affected the outcome.
</p>  


```python
def predict_recommendation(comment):
    preprocessed_comment = preprocess_text(comment)
    
    comment_vector = sentence_vector(preprocessed_comment)
    
    comment_vector = comment_vector.reshape(1, -1)
    
    prediction = logistic_model.predict(comment_vector)
    
    return prediction[0]
```

<p dir="ltr" style="direction: ltr;text-align: justify;line-height:200%;font-family:Times New Roman;font-size:medium">
  In the final prediction step, I applied the implemented function to the reviews in the <code>test</code> dataset.  
  The predicted classes for each review were stored in a DataFrame named <code>submission</code>.  
  This DataFrame was structured as shown below, containing a single column named <code>class</code> that specifies the predicted sentiment class for each review.  
  This file directly influences the final evaluation of the project.
</p>

<table class="center" style="border-collapse: collapse; width: 30%; text-align: center; font-family: Times New Roman; font-size: medium;">
  <tr style="background-color: #e0e0e0;">
    <th style="border: 1px solid #ccc; padding: 8px;">class</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ccc; padding: 8px;">not_recommended</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ccc; padding: 8px;">not_recommended</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ccc; padding: 8px;">recommended</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ccc; padding: 8px;">...</td>
  </tr>
</table>


```python
pre = test_data['body'].apply(predict_recommendation)
submission = pd.DataFrame({'class':pre})
submission = submission.replace({0: 'not_recommended', 1: 'recommended', 2: 'no_idea'})
submission
```

<h2 align="right" style="line-height:200%;font-family:Times New Roman;color:#0099cc">
  <font face="Times New Roman" color="#0099cc">
    <b>Answer Packaging Cell</b>
  </font>
</h2>

<p dir="ltr" style="direction: ltr; text-align: justify; line-height:200%; font-family:Times New Roman; font-size:medium">
  To generate the <code>result.zip</code> file, I executed the following cell.  
  Please make sure that all recent changes in the notebook are saved (<code>Ctrl+S</code>) before running this cell, so that your code can be reviewed if support is needed.
</p>


```python
import os
```

```python
import zipfile
import joblib
 
if not os.path.exists(os.path.join(os.getcwd(), 'persian_comments_preprocessing.ipynb')):
    %notebook -e initial.ipynb


def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('./' + file_name, file_name, compress_type=compression)

submission.to_csv('submission.csv', index=False)

file_names = ['persian_comments_preprocessing.ipynb', 'submission.csv']
compress(file_names)
```

```python

```
