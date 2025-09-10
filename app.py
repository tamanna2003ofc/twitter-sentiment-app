import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# ------------------ Caching ------------------ #
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# ------------------ Prediction ------------------ #
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# ------------------ Card for Tweets ------------------ #
def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    emoji = "😊" if sentiment == "Positive" else "😞"
    card_html = f"""
    <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h5 style="color: white;">{emoji} {sentiment}</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# ------------------ Main App ------------------ #
def main():
    # ------------------ Sidebar ------------------ #
    st.sidebar.markdown(
        "<h2 style='color:#4CAF50;'>🐦 Twitter Sentiment Analyzer</h2>", 
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        """
        **Instructions:**
        - Choose 'Input text' to analyze your own tweet or text.
        - Choose 'Get tweets from user' to fetch latest tweets from a username.
        - Click the button to analyze sentiment.
        """
    )
    st.sidebar.markdown("---")

    # Optional: number of tweets to fetch
    num_tweets = st.sidebar.slider(
        "Number of tweets to fetch", 
        min_value=1, max_value=20, value=5
    )
    st.sidebar.markdown("---")

    # Model info / version
    st.sidebar.markdown(
        "**Model:** TF-IDF + Logistic Regression\n"
        "**Version:** 1.0"
    )
    st.sidebar.markdown("---")

    # Credits / links
    st.sidebar.markdown(
        "[GitHub Repository](https://github.com/your-repo-link)\n\n"
        "Developed by Your Name"
    )

    # ------------------ Header ------------------ #
    st.markdown(
        "<h1 style='color: #4CAF50; text-align:center;'>🐦 Twitter Sentiment Analyzer</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color: #555;'>Analyze your tweets instantly with AI-powered sentiment detection!</p>",
        unsafe_allow_html=True
    )

    #

