"""
3.11 (64 bit)

pip install scipy
pip install matplotlib
pip install seaborn
pip install pandas
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from common.latex_file_generator import save_df_as_image
from common.file_cleanup import latex_file_cleanup


def plot_imdb_rating_against_box_office(number_of_votes=5000):
    """
    Function to plot the average imdb ratings for movies, in order to see distributions over time. We filter on movies
    which have more than 5000 user reviews. This is to avoid skew with lots of low grossing/low popularity movies.

    Function to:

    1. Call load_imdb_dataframe_and_apply_filters() to load and filter on df:
    - Filter on where:
        'votes' column is not NULL
        Movies had more than 5000 votes

    2. Plot the filtered dataframe using seaborn and scipy

    BONUS - the function also:
        - aggregates the dataframe in order to get the year and average rating for that year
        - save this to a pdf

    :return:
    """
    imdb_df = load_imdb_dataframe_and_apply_filters(number_of_votes_adjusted_filter=number_of_votes)

    p = sns.regplot(x="rating", y="gross(in $)", data=imdb_df, line_kws={"color": "blue"})

    slope, intercept, r, p, sterr = stats.linregress(x=p.get_lines()[0].get_xdata(), y=p.get_lines()[0].get_ydata())

    plt.text(2, 600000000, 'y = ' + str(round(intercept, 3)) + ' + ' + str(round(slope, 3)) + 'x', bbox = dict(facecolor = 'blue', alpha = 0.5))
    plt.xlabel("IMDB rating")
    plt.ylabel("Total Gross (in billions of USD)")
    plt.title("IMDB Rating vs Total Gross (in billions of USD)")
    plt.ylim(0, 1000000000)
    plt.clf()


def plot_average_imdb_ratings_per_year(number_of_votes_adjusted_filter=None, save_image=True,
                                       gross_filter=None, year_adjusted_filter=None):
    """
    Function to:

    1. Call load_imdb_dataframe_and_apply_filters() to load and filter on df:
        - Filter on df where:
            'votes' column is not NULL
            the movies gross more than 1 million (1,000,000) USD
            the movies were after 1999 (to match scope of study)

    2. aggregates the dataframe in order to get the year and average rating for that year
    3. Save the aggregated dataframe to a pdf

    :param number_of_votes_adjusted_filter: number of votes we want to filter movies on (e.g. they have to have at least 1000 votes)
    :param save_image: Boolean to determine whether we want to save the table as a pdf (using latex)
    :param gross_filter: total gross (revenue) we want to filter movies on
    :param year_adjusted_filter: filtering on movies past a certain year
    :return: dataframe containing all information on movies in imdb. Columns include:
        - name of the movie
        - when it was released
        - rating (across all voters)
        - director
        - genre
        and more
    """
    imdb_df = load_imdb_dataframe_and_apply_filters(number_of_votes_adjusted_filter=number_of_votes_adjusted_filter,
                                                    gross_filter=gross_filter,
                                                    year_adjusted_filter=year_adjusted_filter)

    if save_image:
        # https://stackoverflow.com/questions/44522741/pandas-mean-typeerror-could-not-convert-to-numeric
        imdb_df_grouped = imdb_df.groupby('year-adjusted')['rating-adjusted'].mean().reset_index()

        save_df_as_image(df=imdb_df_grouped, file_name='imdb_rating_368k')

        latex_file_cleanup(file_name='imdb_rating_368k')

    return imdb_df


def load_imdb_dataframe_and_apply_filters(number_of_votes_adjusted_filter=None, gross_filter=None,
                                          year_adjusted_filter=None):
    """
    Function to:

    1. Load in csv containing all imdb ratings (thanks to concatenate_imdb_movies())
    2. Filter on df where:
            'votes' column is not NULL
            (Optional) filter on movies with more than a certain number of votes (e.g. 5000)
            (Optional) filter movies gross more than a certain amount (e.g. 1,000,000)
            (Optional) filter on movies were after a certain date (e.g. 1999)
    3. Return this dataframe

    :param number_of_votes_adjusted_filter:
    :param gross_filter:
    :param year_adjusted_filter:
    :return:
    """
    pd.set_option('display.max_rows', 500)

    # all_movies.csv contain 368,000 movies pre data cleaning
    imdb_df = pd.read_csv(os.path.join(os.getcwd(), 'genre', 'all_movies.csv'))
    imdb_df = imdb_df[imdb_df['votes'].notna()]
    imdb_df['votes'] = imdb_df['votes'].replace({'K': '000', 'M': '000000'}, regex=True).map(pd.eval).astype(int)
    imdb_df['user_rating_adjusted'] = pd.to_numeric(imdb_df['votes'], errors='coerce').fillna(0)

    if number_of_votes_adjusted_filter:
        imdb_df = imdb_df[imdb_df['user_rating_adjusted'] > number_of_votes_adjusted_filter]

    if gross_filter:
        imdb_df = imdb_df[imdb_df['gross(in $)'] > gross_filter]

    imdb_df['rating-adjusted'] = pd.to_numeric(imdb_df['rating'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['rating-adjusted'])

    imdb_df['year-adjusted'] = pd.to_numeric(imdb_df['year'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['year-adjusted'])

    if year_adjusted_filter:
        imdb_df['year-adjusted'] = imdb_df['year-adjusted'].abs()
        imdb_df = imdb_df[imdb_df['year-adjusted'] > year_adjusted_filter]

    return imdb_df


if __name__ == '__main__':
    plot_imdb_rating_against_box_office()