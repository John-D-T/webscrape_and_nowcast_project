import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

"""
3.7 VENV
"""

def get_average_imdb_ratings_all_movies():
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

    #imdb_df = imdb_df[imdb_df['gross(in $)'] > 50000000]
    p = sns.regplot(x="rating", y="gross(in $)", data=imdb_df, scatter_kws={"color": "black"}, line_kws={"color": "blue"}, ci=95)

    #.set(title='Imdb Rating vs Total Gross (in 10s of millions)')

    slope, intercept, r, p, sterr = stats.linregress(x=p.get_lines()[0].get_xdata(), y=p.get_lines()[0].get_ydata())

    plt.text(8, 95, 'y = ' + str(round(intercept, 3)) + ' + ' + str(round(slope, 3)) + 'x')

    plt.clf()
    sns.regplot(x="rating", y="gross(in $)", data=imdb_df, order=2)
    'gross(in $)'


if __name__ == '__main__':
    get_average_imdb_ratings_all_movies()