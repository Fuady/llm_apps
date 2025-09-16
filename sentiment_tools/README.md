# EasyJet Sentiment Analysis Dashboard

This project is a **Streamlit web application** that scrapes customer reviews from **Trustpilot** and performs **sentiment analysis** using different machine learning models. It helps analyze feedback to improve products and services by visualizing sentiment insights and generating actionable recommendations.

---

## Features

- **Web Scraping**: Extracts reviews (title, date, rating, text, page number) from Trustpilot.
- **Sidebar Settings**:
  - Input the Trustpilot review URL.
  - Select the number of pages to scrape (1–200).
  - Choose the sentiment analysis model: `SVM`, `Logistic Regression`, or `LSTM`.
- **Overview**:
  - Displays total number of pages scraped.
  - Shows a preview of 3 rows of scraped reviews.
- **Sentiment Analysis**:
  - Preprocesses review text with cleaning, tokenization, and vectorization.
  - Classifies reviews into `Positive`, `Neutral`, or `Negative`.
- **Visualization**:
  - WordClouds for positive and negative feedback.
- **Recommendations**:
  - Provides 10 actionable suggestions to reduce negative feedback and strengthen positives.
- **Export**:
  - Download scraped reviews with sentiment labels as CSV.

---

## Installation

Clone the repository and install the required packages:

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

---

## Usage

Run the Streamlit app locally:

```bash
streamlit run streamlit_easyjet_sentiment.py
```

Then open the browser link (usually `http://localhost:8501`).

---

## Models

1. **SVM (Support Vector Machine)** – Default linear classifier, efficient for text classification.
2. **Logistic Regression** – Simple yet effective baseline model.
3. **LSTM (Long Short-Term Memory)** – Deep learning approach using embeddings and recurrent layers.

---

## Requirements

See [requirements.txt](./requirements.txt) for full list. Key libraries include:
- Streamlit
- Requests
- BeautifulSoup4
- Pandas
- NLTK
- Scikit-learn
- TensorFlow / Keras
- WordCloud
- Matplotlib

---

## Recommendations Generated

The app outputs 10 business recommendations, for example:
- Improve flight punctuality to reduce negative feedback.
- Enhance customer service training.
- Offer clearer communication on delays and cancellations.
- Provide compensation vouchers for inconveniences.

---

## Notes

- Trustpilot’s HTML structure may change, requiring scraper updates.
- Training the **LSTM** model may take longer compared to SVM or Logistic Regression.
- Data is not saved between runs (scraping occurs each time).

---

## License

This project is for educational and analytical purposes only. Use responsibly and respect Trustpilot’s [Terms of Service](https://legal.trustpilot.com/end-user-terms-and-conditions).

