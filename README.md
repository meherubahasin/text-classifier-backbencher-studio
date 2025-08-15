# Sentiment Classification
---

**Project Description**
This project implements a **Convolutional Neural Network (CNN)** for binary sentiment classification on text data of the [IMDB Movie Review Dataset from Kaggle](https://www.kaggle.com/datasets/mantri7/imdb-movie-reviews-dataset/). 
## Approaches

1. **Logistic Regression (LR)**  
   - Baseline ML model using TF-IDF features.
2. **Naive Bayes (NB)**  
   - Probabilistic text classifier.
3. **Custom CNN**  
   - Deep learning model with embedding and multiple Conv1D layers for sequential feature extraction.
  
## Tools & Libraries
- Python 3.11+  
- [Pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [Scikit-learn](https://scikit-learn.org/)  
- [TensorFlow / Keras](https://www.tensorflow.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Joblib](https://joblib.readthedocs.io/)
- 
The pipeline includes:

1. **Data Preprocessing**
   * Naming the columns in the CSV file for distinction
   * Remove HTML tags using regex
   * Remove punctuation and numbers
   * Convert all text to lowercase
   * Remove common English stopwords using NLTK
   * Split the train dataset into 80% training and 20% test sets
   * Vectorization such as TF-IDF for ML models
   * Tokenization with Keras 'Tokenizer' (max vocab size: 30,000) for CNN model
   * Conversion of text to padded sequences (max length: 300)

2. **Model Architecture of CNN model**

   * **Embedding layer** for word vector representation
   * Multiple 'Conv1D' layers with varying kernel sizes for feature extraction
   * Global Max Pooling and Max Pooling layers for dimensionality reduction
   * Fully connected Dense layers with Dropout for regularization
   * Final Sigmoid output for binary classification

3. **Training & Saving**

   * Binary cross-entropy loss with Adam optimizer
   * Training for 5 epochs with validation split
   * Saves both the trained model ('.h5') and tokenizer ('.pkl')

4. **Evaluation & Visualization**

   * Accuracy score on test data
   * Classification report (precision, recall, F1-score)
   * Confusion matrix visualization
     
## Results

| Model                | Accuracy | Precision | Recall  | F1-Score |
|----------------------|----------|-----------|---------|----------|
| Logistic Regression  | 0.8830   | 0.8832    | 0.8828  | 0.8830   |
| Naive Bayes          | 0.8538   | 0.8609    | 0.8440  | 0.8524   |
| Custom CNN           | 0.8600   | 0.8600    | 0.8600  | 0.8600   |

**Summary:**  
- Logistic Regression achieved the highest accuracy among ML models.  
- Naive Bayes performs slightly lower but is still effective for baseline comparisons.  
- Custom CNN captures sequential text features, balancing precision and recall across classes.  

**Dependencies**:
'tensorflow', 'scikit-learn', 'matplotlib', 'joblib'


