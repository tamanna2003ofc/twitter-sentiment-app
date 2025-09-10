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
    st.sidebar.markdown("<h2 style='color:#4CAF50;'>🐦 Twitter Sentiment Analyzer</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("""
        **Instructions:**
        - Choose 'Input text' to analyze your own tweet or text.
        - Choose 'Get tweets from user' to fetch latest tweets from a username.
        - Click the button to analyze sentiment.
    """)
    st.sidebar.markdown("---")

    # Number of tweets to fetch
    num_tweets = st.sidebar.slider("Number of tweets to fetch", 1, 20, 5)
    st.sidebar.markdown("---")

    # Model info / version
    st.sidebar.markdown("**Model:** TF-IDF + Logistic Regression\n**Version:** 1.0")
    st.sidebar.markdown("---")

    # Credits / links
    st.sidebar.markdown("[GitHub Repository](https://github.com/your-repo-link)\n\nDeveloped by Tamanna Thakur")

    # ------------------ Header ------------------ #
    st.markdown("<h1 style='color: #4CAF50; text-align:center;'>🐦 Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: #555;'>Analyze your tweets instantly with AI-powered sentiment detection!</p>", unsafe_allow_html=True)

    # Load stopwords, model, vectorizer, scraper
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    # ------------------ User Input Options ------------------ #
    st.markdown("<div style='background-color:#e0f7fa;padding:15px;border-radius:10px;'><h4>Choose how to analyze sentiment:</h4></div>", unsafe_allow_html=True)
    option = st.selectbox("", ["Input text", "Get tweets from user"])

    # ------------------ Layout: Columns ------------------ #
    col1, col2 = st.columns([1, 2])  # Input column (1/3) and output column (2/3)

    # ------------------ Option 1: Input Text ------------------ #
    if option == "Input text":
        with col1:
            text_input = st.text_area("Enter text to analyze sentiment", height=150)
            analyze_btn = st.button("Analyze")
        with col2:
            if analyze_btn:
                if text_input.strip() == "":
                    st.warning("Please enter some text to analyze.")
                else:
                    sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                    color = "green" if sentiment == "Positive" else "red"
                    emoji = "😊" if sentiment == "Positive" else "😞"
                    st.markdown(f"<h3 style='color:{color}'>{emoji} {sentiment}</h3>", unsafe_allow_html=True)

    # ------------------ Option 2: Get Tweets from User ------------------ #
    elif option == "Get tweets from user":
        with col1:
            username = st.text_input("Enter Twitter username (without @)")
            fetch_btn = st.button("Fetch Tweets")
        with col2:
            if fetch_btn:
                if username.strip() == "":
                    st.warning("Please enter a username.")
                else:
                    tweets_data = scraper.get_tweets(username, mode='user', number=num_tweets)
                    if 'tweets' in tweets_data:
                        st.markdown("<h4 style='color:#4CAF50'>Latest Tweets:</h4>", unsafe_allow_html=True)
                        for tweet in tweets_data['tweets']:
                            tweet_text = tweet['text']
                            sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                            card_html = create_card(tweet_text, sentiment)
                            st.markdown(card_html, unsafe_allow_html=True)
                    else:
                        st.error("No tweets found or an error occurred.")

# ------------------ Run App ------------------ #
if __name__ == "__main__":
    main()

