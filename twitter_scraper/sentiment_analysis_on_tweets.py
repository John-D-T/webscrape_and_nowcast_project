"""
pip install textblob
pip install vaderSentiment

3.7 VENV

https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27
"""
from textblob import TextBlob
import pandas as pd
import os
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string

def sentiment_analysis(df, column_name):
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df[column_name].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    return df

def plot_sentiment_analysis(df):
    # TODO - revisit this sentiment plot when I have 2021 dates
    df['date'] = pd.to_datetime(df['date'])
    df.plot(x='date', y='sentiment')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(os.getcwd(), 'compiled_tweets_odeon.csv'))
    column_name = 'tweet'
    df = sentiment_analysis(df, column_name)

    plot_sentiment_analysis(df)

