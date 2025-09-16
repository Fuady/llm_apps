import streamlit as st
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sidebar inputs
st.sidebar.header("Settings")
url = st.sidebar.text_input("Webpage to scrape", "https://uk.trustpilot.com/review/www.easyjet.com")
num_pages = st.sidebar.number_input("Number of pages to scrape", min_value=1, max_value=200, value=50)
model_choice = st.sidebar.selectbox("Choose Model", ["SVM", "Logistic Regression", "LSTM"])

st.title("EasyJet Sentiment Analysis Dashboard")

# Scraping reviews
data = []
for i in range(1, num_pages + 1):
    response = requests.get(f"{url}?page={i}")
    soup = BeautifulSoup(response.text, "html.parser")
    for e in soup.select('article'):
        data.append({
            'review_title': e.h2.text if e.h2 else None,
            'review_date_original': e.select_one('[data-service-review-date-of-experience-typography]').text.split(': ')[-1] if e.select_one('[data-service-review-date-of-experience-typography]') else None,
            'review_rating': e.select_one('[data-service-review-rating] img').get('alt') if e.select_one('[data-service-review-rating] img') else None,
            'review_text': e.select_one('[data-service-review-text-typography]').text if e.select_one('[data-service-review-text-typography]') else None,
            'page_number': i
        })

df = pd.DataFrame(data)

# Preprocessing
def preprocess_data(df):
    df['review_rating_num'] = df['review_rating'].str.extract('(\\d+)').astype(float)
    df['sentiment'] = df['review_rating_num'].apply(lambda x: 'Positive' if x > 3 else 'Neutral' if x == 3 else 'Negative')
    df['clean_text'] = df['review_text'].fillna("").astype(str).str.replace('[^a-zA-Z]', ' ', regex=True).str.lower()
    return df

df = preprocess_data(df)

# st.subheader("Scraping Overview")
# st.write(f"Total pages scraped: {num_pages}")
# st.dataframe(df.head(3))

st.success("Scraping done!")



# Sentiment count interactive bar chart
st.subheader("📊 Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
title="Count of Positive, Neutral, and Negative Feedback",
text='Count')
st.plotly_chart(fig_bar)


# Additional visualization: sentiment trend over pages
st.subheader("📈 Sentiment Trend Over Pages")
trend_data = df.groupby(['page_number', 'sentiment']).size().reset_index(name='count')
fig_trend = px.line(trend_data, x='page_number', y='count', color='sentiment',
title="Trend of Sentiment Across Scraped Pages")
st.plotly_chart(fig_trend)

# Train/test split
X = df['clean_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
X_train = X_train.fillna("")
X_test = X_test.fillna("")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model selection
if model_choice == "SVM":
    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)
    predictions = clf.predict(X_test_tfidf)

elif model_choice == "Logistic Regression":
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    predictions = clf.predict(X_test_tfidf)

elif model_choice == "LSTM":
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200)

    lstm_model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=200),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def encode_labels(y):
        return y.map({'Negative': 0, 'Neutral': 1, 'Positive': 2}).values

    encoded_y_train = encode_labels(y_train)
    encoded_y_test = encode_labels(y_test)

    lstm_model.fit(X_train_seq, encoded_y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=1)
    predictions = np.argmax(lstm_model.predict(X_test_seq), axis=1)
    predictions = pd.Series(predictions).map({0:"Negative",1:"Neutral",2:"Positive"})


# Wordclouds
st.subheader("WordClouds")
positive_text = " ".join(df[df['sentiment']=='Positive']['clean_text'])
negative_text = " ".join(df[df['sentiment']=='Negative']['clean_text'])

col1, col2 = st.columns(2)
with col1:
    st.write("Positive Sentiment")
    pos_wc = WordCloud(width=400, height=300).generate(positive_text)
    fig, ax = plt.subplots()
    ax.imshow(pos_wc)
    ax.axis("off")
    st.pyplot(fig)

with col2:
    st.write("Negative Sentiment")
    neg_wc = WordCloud(width=400, height=300).generate(negative_text)
    fig, ax = plt.subplots()
    ax.imshow(neg_wc)
    ax.axis("off")
    st.pyplot(fig)

# Recommendations
st.subheader("Recommendations")
recommendations = [
    "Improve flight punctuality to reduce negative feedback.",
    "Enhance customer service training.",
    "Offer clearer communication on delays and cancellations.",
    "Streamline baggage handling process.",
    "Provide compensation vouchers for inconveniences.",
    "Improve app and website booking user experience.",
    "Offer loyalty rewards for frequent flyers.",
    "Maintain affordable pricing while improving quality.",
    "Add more in-flight amenities.",
    "Highlight positive experiences in marketing campaigns."
]
for rec in recommendations:
    st.write(f"- {rec}")

# Download button
st.download_button("Download Data as CSV", df.to_csv(index=False).encode('utf-8'), "easyjet_reviews.csv", "text/csv")