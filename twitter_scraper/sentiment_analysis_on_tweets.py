"""
pip install textblob
pip install vaderSentiment
pip install wordcloud
pip install polars

3.7 VENV

https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import numpy as np
import polars as pl
from langdetect import detect
import re
from PIL import Image


def sentiment_analysis(df, column_name):
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df[column_name].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    return df

def plot_sentiment_analysis(df):
    # TODO - revisit this sentiment plot when I have 2021 dates
    df['date'] = pd.to_datetime(df['date'])

    # Set the cutoff date, based on when covid started in the UK
    cutoff_date = pd.to_datetime('2020-02-01')

    # Filter the DataFrame
    df = df[df['date'] < cutoff_date]

    grouped_df = df.groupby(pd.Grouper(key='date', freq='M')).agg({'sentiment': 'mean'}).reset_index()
    grouped_df['date_grouped'] = pd.to_datetime(grouped_df['date']).apply(
        lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
    grouped_df['date'] = grouped_df['date_grouped']
    grouped_df.plot(x='date', y='sentiment')
    plt.title('sentiment score for tweets containing "odeon"')
    plt.ylabel('vaderSentiment score')
    plt.show()

def generate_wordcloud(df):
    # https://towardsdatascience.com/text-analytics-101-word-cloud-and-sentiment-analysis-2c3ade81c7e8
    # https://towardsdatascience.com/how-to-make-word-clouds-in-python-that-dont-suck-86518cdcb61f

    def tweet_cleaner(x):
        tweet = re.sub("[@&][A-Za-z0-9_]+", "", x)  # Remove mentions
        tweet = re.sub(r"http\S+", "", tweet)  # Remove media links

        return pd.Series([tweet])

    df = df[['tweet']]
    df[['plain_text']] = df.tweet.apply(tweet_cleaner)
    # Convert all text to lowercase
    df.plain_text = df.plain_text.str.lower()
    # Remove newline character
    df.plain_text = df.plain_text.str.replace('\n', '')
    # Remove 'odeon' term
    df.plain_text = df.plain_text.str.replace('odeon', '')
    # Replacing any empty strings with null
    #df = df.replace(r'^\s*$', np.nan, regex=True)
    if df.isnull().sum().plain_text == 0:
        print('no empty strings')
    else:
        #df.dropna(inplace=True)
        pass

    # def detect_textlang(text):
    #     try:
    #         src_lang = detect(text)
    #         if src_lang == 'en':
    #             return 'en'
    #         else:
    #             # return "NA"
    #             return src_lang
    #     except:
    #         return "NA"
    #
    # df['text_lang'] = df.plain_text.apply(detect_textlang)
    #
    # # Group tweets by language and list the top 10
    # plt.figure(figsize=(4, 3))
    # df.groupby(df.text_lang).plain_text.count().sort_values(ascending=False).head(10).plot.bar()
    # plt.show()
    #
    # # Filter out non english tweets
    # df = df[(df.text_lang == "en")]

    # # Attempting to divide text into positive and negative lists
    # uncategorized_plain_text = ' '.join(df.plain_text)
    # uncategorized_plain_text_list = uncategorized_plain_text.split(' ')
    # df_words_pl = pl.DataFrame(uncategorized_plain_text_list, schema=['word'])
    # df_words = pd.DataFrame(uncategorized_plain_text_list, columns=['word'])
    #
    # # Chunk attempt
    # list_df = np.array_split(df_words, 100)
    # for subset_df in list_df:
    #     subset_df['sentiment'] = SentimentIntensityAnalyzer().polarity_scores(subset_df['word'])['compound'] > 0
    #
    # df_words['sentiment'] = SentimentIntensityAnalyzer().polarity_scores(df_words['word'])['compound'] > 0
    #
    # positive_text_list = [word for word in uncategorized_plain_text_list if
    #                       SentimentIntensityAnalyzer().polarity_scores(word)['compound'] > 0]
    # negative_text_list = [word for word in uncategorized_plain_text_list if
    #                       SentimentIntensityAnalyzer().polarity_scores(word)['compound'] < 0]
    #
    # positive_text = ' '.join(positive_text_list)
    # negative_text = ' '.join(negative_text_list)
    # change the value to black
    def black_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return ("hsl(0,100%, 1%)")

    # Plotting the image
    mask = np.array(Image.open(os.path.join(os.getcwd(), 'film_icon.PNG')))
    wordcloud = WordCloud(mask=mask, background_color="white", width=3000,
                          height=2000, max_words=500).generate(' '.join(df.plain_text))
    # set the word color to black
    wordcloud.recolor(color_func = black_color_func)
    plt.figure(figsize=[15, 10])
    # plot the wordcloud
    plt.imshow(wordcloud, interpolation="bilinear")
    # remove plot axes
    plt.axis("off")

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(os.getcwd(), 'compiled_tweets_odeon.csv'))
    column_name = 'tweet'
    df = sentiment_analysis(df, column_name)

    #plot_sentiment_analysis(df)

    generate_wordcloud(df)

