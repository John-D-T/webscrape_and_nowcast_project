"""
pip install textblob

3.7 VENV

"""
from textblob import TextBlob
import pandas as pd
import os
import matplotlib.pyplot as plt

def sentiment_analysis(df, column_name):
    df['polarity'] = df[column_name].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df[column_name].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')
    return df

def plot_sentiment_analysis(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby(['date', 'sentiment']).size().reset_index(name='counts')
    df_pivot = df.pivot(index='date', columns='sentiment', values='counts')
    df_pivot.plot(kind='line')
    plt.show()


if __name__ == '__main__':

    df = pd.read_csv(os.path.join(os.getcwd(), 'compiled_tweets_odeon.csv'))
    column_name = 'tweet'

    df = sentiment_analysis(df, column_name)

    plot_sentiment_analysis(df)

