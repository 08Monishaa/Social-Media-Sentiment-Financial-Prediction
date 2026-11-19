
# Social Media Sentiment and Financial Trend Prediction

## 1. Project Overview

This project analyzes social media sentiment and historical
stock market data to investigate whether public opinion is
associated with subsequent stock price movements.

It classifies social media posts into Positive, Negative,
and Neutral categories, identifies trending topics, and
predicts the next trading day's stock price direction.

## 2. Objectives

- Perform sentiment analysis on sample social media posts.
- Identify trending financial topics.
- Collect historical stock market information.
- Combine sentiment features with financial indicators.
- Predict upward or downward stock movements.
- Compare a baseline model with a sentiment-enhanced model.

## 3. Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- NLTK VADER
- Scikit-learn
- Random Forest Classifier
- Matplotlib and Seaborn
- Yahoo Finance (yfinance)

## 4. Dataset Description

### Social Media Dataset

File: social_media_posts.csv

Contains 100 synthetic sample posts about TCS, Infosys,
and Wipro, with dates and company names.

The posts are created for academic demonstration.
They are not actual social media posts.

### Stock Market Dataset

File: stock_data.csv

Historical TCS stock market data downloaded using
Yahoo Finance through the yfinance Python library.

Ticker: TCS.NS

Period requested: 2024-01-01 to 2026-01-01.

## 5. Data Processing

- Prepared and cleaned text data.
- Classified posts using VADER sentiment scores.
- Aggregated sentiment scores by date.
- Calculated daily stock returns.
- Calculated 5-day and 10-day moving averages.
- Calculated stock price volatility.
- Created next-day UP/DOWN prediction labels.
- Handled missing and infinite values.

## 6. Algorithms

### Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner)
is used to calculate sentiment polarity.

### Trending Topic Identification

CountVectorizer is used to identify frequently
occurring words and topics.

### Financial Prediction

Random Forest Classifier is used to predict the
next trading day's stock direction.

## 7. Model Comparison

Two models are evaluated:

1. Baseline: Historical stock price and technical features.
2. Enhanced: Historical features combined with sentiment.

A chronological 80:20 train-test split is used.

### Actual Model Results

| Metric | Baseline | Enhanced |
|---|---:|---:|
| Accuracy | 0.5000 | 0.5417 |

See model_comparison_results.csv for all evaluation metrics.

## 8. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

## 9. Visualizations

The visualizations folder contains:

1. Sentiment distribution.
2. Trending topics.
3. Historical stock prices.
4. Baseline vs enhanced model performance.

## 10. Installation

Install the dependencies:

pip install -r requirements.txt

## 11. Execution Instructions

1. Open Google Colab.
2. Upload Social_Media_Sentiment_Financial_Prediction.ipynb.
3. Run the notebook cells sequentially.
4. Connect to the internet to download stock data.
5. Review sentiment reports, graphs and predictions.

## 12. Limitations

- Social media posts are synthetic sample data.
- Sentiment results do not represent actual investor opinion.
- The sample dataset is limited in size and time coverage.
- Stock market movements are affected by many factors.
- Predictions are experimental and not investment advice.

## 13. Conclusion

The project demonstrates how Natural Language Processing
and Machine Learning can be combined to study the
relationship between public sentiment and financial trends.

It compares historical-price-only prediction against
sentiment-enhanced prediction using a time-based split.
