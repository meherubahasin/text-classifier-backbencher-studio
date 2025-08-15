import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Download the model from text_classifier.ipynb or use the demo in a separate cell in the notebook
model = tf.keras.models.load_model("review_classifier_cnn.h5")
tokenizer = joblib.load("tokenizer.pkl")
max_seq_len = 300  

def preprocess_input(text):
    import re, string, nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    nltk.download('punkt')

    stop_words = set(stopwords.words('english'))
    text = re.sub(r'<.*?>', ' ', text) 
    text = ''.join(ch for ch in text if ch not in string.punctuation and not ch.isdigit())
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)


while True:
    user_input = input("Enter a review (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break

    clean_text = preprocess_input(user_input)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded_seq = pad_sequences(seq, maxlen=max_seq_len, padding='post')


    prob = model.predict(padded_seq)[0][0]
    sentiment = "Positive" if prob > 0.5 else "Negative"

    print(f"Predicted Sentiment: {sentiment} ({prob:.2f} confidence)")
