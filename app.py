import streamlit as st # Web app interface

#Tokenizer, stopwords list, lemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import pickle
import nltk
import sqlite3
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from datetime import datetime

# Preprocessing functions

def preprocess_text_spam(text):
    """Performs text preprocessing: lowercasing,
        punctuation stripping, tokenization"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = word_tokenize(text)
    return " ".join(tokens)


def preprocess_text_toxic(text):
    """Performs text preprocessing steps:
        punctuation stripping, stopword removal, lemmatization
    Args:
        text (str): The input text
        
    Returns:
        str: The preprocessed text"""
    # Lowercasing
    text = text.lower()

    # remove URLs, hashtags and usernames
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", '', text)

    # Handle punctuation
    ## Replace apostrophes and hyphen with whitespace
    text = re.sub(r"['\-]", " ", text)

    ## Replace non-alphanumeric symbols with whitespace
    text = re.sub(r"[^\w\s]", ' ', text)

    # Tokenization
    tokens = word_tokenize(text)

    #Stop Word Removal  
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in
        stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)
# -- Database integration

def init_db(db_path="predictions.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            input_text TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- Load Models and Vectorizers ---

try: 
    # --- Spam Detection ---
    spam_model_path = 'spam_classifier_model.pkl'
    spam_vectorizer_path = 'spam_data_vectorizer.pkl'
    with open(spam_model_path, 'rb') as f:
        spam_model = pickle.load(f)

    with open(spam_vectorizer_path, 'rb') as f:
        spam_vectorizer = pickle.load(f)

    # --- Toxic Text Detection ---
    toxicity_model_path = 'toxicity_classifier_model.pkl'
    toxicity_vectorizer_path = 'toxicity_data_vectorizer.pkl'
    with open(toxicity_model_path, 'rb') as f:
        toxic_model = pickle.load(f)

    with open(toxicity_vectorizer_path, 'rb') as f:
        toxic_vectorizer = pickle.load(f)


except FileNotFoundError:
    print("Model or vectorizer file(s) not found. Confirm",
        "presence in 'models' and 'vectorizer' directories")
    st.stop()
except Exception as e:
    st.error(f"An error occured: {e}")
    st.stop()

def log_prediction(task, input_text, prediction, probability, db_path="predictions.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Format timestamp
    cursor.execute("""
        INSERT INTO predictions (task, input_text, prediction, probability, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (task, input_text, prediction, probability, timestamp))
    conn.commit()
    conn.close()

def predict_spam(text, model, vectorizer, threshold=0.5):
    processed_text = preprocess_text_spam(text)
    vectorized_text = vectorizer.transform([processed_text])
    probability = model.predict_proba(vectorized_text)[0, 1]
    prediction = (probability > threshold).astype(int)
    return prediction, probability

def predict_offensive(text, model, vectorizer, threshold=0.5):
    processed_text = preprocess_text_toxic(text)
    vectorized_text = vectorizer.transform([processed_text])
    probability = model.predict_proba(vectorized_text)[0, 1]
    prediction = (probability > threshold).astype(int)
    return prediction, probability

# --- Streamlit App ---
st.title("Spam and Toxic Text Detection")
st.write("This app uses machine learning models to detect spam and offensive text.")

task_choice = st.selectbox("Choose a Task:", ["Spam Detection", "Toxic Text Detection"])
text_input = st.text_area("Enter text:", "Type your text here...")
threshold = st.slider("Classification Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)


if st.button("Detect"):
    if text_input:
        if task_choice == "Spam Detection":
            prediction, probability = predict_spam(text_input, spam_model, spam_vectorizer, threshold)
            log_prediction("Spam Detection", text_input, int(prediction), float(probability))
            if prediction == 1:
                st.error(f"This text is classified as SPAM (probability: {probability:.2f})")
            else:
                st.success(f"This text is classified as NOT SPAM (probability: {probability:.2f})")
        elif task_choice == "Toxic Text Detection":
            prediction, probability = predict_offensive(text_input, toxic_model, toxic_vectorizer, threshold)
            log_prediction("Toxic Text Detection", text_input, int(prediction), float(probability))

            if prediction == 1:
                st.error(f"This text is classified as TOXIC (probability: {probability:.2f})")
            else:
                st.success(f"This text is classified as NOT TOXIC (probability: {probability:.2f})")
    else:
        st.warning("Please enter some text.")

# Model Information
st.sidebar.header("About the Models")
# Add sidebar content
st.sidebar.header("About This App")

# App Description
st.sidebar.markdown("""
This app uses machine learning models to detect **spam** and **toxic text** in user inputs. It is designed to help identify unwanted or harmful content in real-time.

- **Spam Detection**: Classifies text as spam or not spam.
- **Toxic Text Detection**: Identifies offensive or toxic language in text.
""")

# Model Information
st.sidebar.subheader("Models Used")
st.sidebar.markdown("""
- **Spam Detection**: Trained on the [Tiago et. al SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) using a Multinomial Naive Bayes classifier.
- **Toxic Text Detection**: Trained on the [Davidson et. al Hate Speech and Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language) using a Logistic Regression model.
""")

# How It Works
st.sidebar.subheader("How It Works")
st.sidebar.markdown("""
1. Enter text in the input box.
2. Select the task: Spam Detection or Toxic Text Detection.
3. Click **Detect** to see the prediction and probability score.
""")

# Contact Information
st.sidebar.subheader("Contact")
st.sidebar.markdown("""
If you have any questions or feedback, feel free to reach out:
- Email: obinnaokpolu@gmail.com
- GitHub: [My GitHub Profile](https://github.com/obinna-okpolu)
""")

# Additional Notes
st.sidebar.subheader("Disclaimer")
st.sidebar.markdown("""
This app is for educational purposes only. The predictions are based on statistical models and may not always be 100% accurate.
""")
