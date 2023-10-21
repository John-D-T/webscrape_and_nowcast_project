import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

"""
3.7 VENV
pip install scipy matplotlib seaborn pandas
"""

def get_average_imdb_ratings_all_movies():
    """
    Function to plot the average imdb ratings for movies, in order to see distributions over time. We filter on movies
    which have more that 5000 user reviews. This is to avoid skew with lots of low grossing/low popularity movies.
    :return:
    """
    # 368,000 movies pre data cleaning
    pd.set_option('display.max_rows', 500)

    imdb_df = pd.read_csv(os.path.join(os.getcwd(), 'genre', 'all_movies.csv'))
    imdb_df = imdb_df[imdb_df['votes'].notna()]
    imdb_df['votes'] = imdb_df['votes'].replace({'K': '000', 'M': '000000'}, regex=True).map(pd.eval).astype(int)
    imdb_df['user_rating_adjusted'] = pd.to_numeric(imdb_df['votes'], errors='coerce').fillna(0)
    imdb_df = imdb_df[imdb_df['user_rating_adjusted'] > 5000]

    imdb_df['rating_adjusted'] = pd.to_numeric(imdb_df['rating'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['rating_adjusted'])

    imdb_df['year_adjusted'] = pd.to_numeric(imdb_df['year'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['year_adjusted'])

    p = sns.regplot(x="rating", y="gross(in $)", data=imdb_df, line_kws={"color": "blue"})

    slope, intercept, r, p, sterr = stats.linregress(x=p.get_lines()[0].get_xdata(), y=p.get_lines()[0].get_ydata())

    plt.text(2, 600000000, 'y = ' + str(round(intercept, 3)) + ' + ' + str(round(slope, 3)) + 'x', bbox = dict(facecolor = 'blue', alpha = 0.5))
    plt.xlabel("IMDB rating")
    plt.ylabel("Total Gross (in billions of USD)")
    plt.title("IMDB Rating vs Total Gross (in billions of USD)")
    plt.ylim(0, 1000000000)
    plt.clf()


if __name__ == '__main__':
    get_average_imdb_ratings_all_movies()