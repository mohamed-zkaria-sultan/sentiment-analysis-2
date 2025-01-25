import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
import unicodedata
import html
import string

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load stop words
stop_words = stopwords.words('english')

# Load the Keras model
model = tf.keras.models.load_model('my_model.keras')

# Load the tokenizer
tok =  tf.keras.preprocessing.text.Tokenizer(
    num_words=None,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ',
    char_level=False,
    oov_token=None,
    analyzer=None
)
# Helper functions
def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))

def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def replace_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_whitespaces(text):
    return text.strip()

def remove_stopwords(words, stop_words):
    return [word for word in words if word not in stop_words]

def stem_words(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

def text2words(text):
    return word_tokenize(text)

def normalize_text(text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    words = remove_stopwords(words, stop_words)
    words = lemmatize_words(words)
    words = lemmatize_verbs(words)
    return ' '.join(words)

# Streamlit app
st.title("Sentiment Analysis App")
st.subheader("Enter a review to analyze its sentiment")

# Text input for the user to type a review
user_review = st.text_area("Write your review here:")

# Analyze button
if st.button("Analyze Sentiment"):
    if user_review:
        processed_text = normalize_text(user_review)
        # Preprocess using the tokenizer
        sequences = tok.texts_to_sequences([processed_text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
        
        # Predict sentiment
        result = model.predict(padded_sequences)
        sentiment_labels = ["NEGATIVE", "POSITIVE"]
        predicted_class = tf.argmax(result, axis=1).numpy()[0]
        confidence = tf.reduce_max(result, axis=1).numpy()[0]
        sentiment = sentiment_labels[predicted_class]
        
        # Display results
        if sentiment == "POSITIVE":
            st.success(f"The review is Positive 😊 (Confidence: {confidence:.2f})")
        elif sentiment == "NEGATIVE":
            st.error(f"The review is Negative 😞 (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter a review to analyze!")
