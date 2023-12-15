import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from common.latex_file_generator import save_df_as_image


def get_average_imdb_ratings_all_movies():
    """
    Function to:

    1. Load in csv containing all imdb ratings (thanks to concatenate_imdb_movies())
    2. Filter on where:
            'votes' column is not NULL
                (optional filter on movies with more than 5000 votes)
            the movies gross more than 1 million (1,000,000) USD
            the movies were after 1999 (to match scope of study)
    3. Return this dataframe

    BONUS - the function also:
        - aggregates the dataframe in order to get the year and average rating for that year
        - save this to a pdf

    :return:
    """
    # 368,000 movies pre data cleaning
    pd.set_option('display.max_rows', 500)

    imdb_df = pd.read_csv(os.path.join(os.getcwd(), 'genre', 'all_movies.csv'))
    imdb_df = imdb_df[imdb_df['votes'].notna()]
    imdb_df['votes'] = imdb_df['votes'].replace({'K': '000', 'M': '000000'}, regex=True).map(pd.eval).astype(int)
    imdb_df['user_rating_adjusted'] = pd.to_numeric(imdb_df['votes'], errors='coerce').fillna(0)
    # imdb_df = imdb_df[imdb_df['user_rating_adjusted'] > 5000]

    imdb_df = imdb_df[imdb_df['gross(in $)'] > 1000000]

    imdb_df['rating_adjusted'] = pd.to_numeric(imdb_df['rating'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['rating_adjusted'])

    imdb_df['year_adjusted'] = pd.to_numeric(imdb_df['year'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['year_adjusted'])
    imdb_df['year_adjusted'] = imdb_df['year_adjusted'].abs()
    imdb_df = imdb_df[imdb_df['year_adjusted'] > 1999]

    # https://stackoverflow.com/questions/44522741/pandas-mean-typeerror-could-not-convert-to-numeric
    imdb_df_grouped = imdb_df.groupby('year_adjusted')['rating_adjusted'].mean().reset_index()

    save_df_as_image(df=imdb_df_grouped, file_name='imdb_rating_368k')

    return imdb_df


def concatenate_imdb_movies():
    """
    One time function to combine all the individual imdb files into one.
    :return:
    """
    extension = 'csv'
    path = os.path.join(os.getcwd(), 'genre')
    os.chdir(path)
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv("all_movies.csv", index=False, encoding='utf-8-sig')


def plot_imdb_distributions(df):
    """
    Function to plot ratings for all imdb movies over the years

    :param df: dataframe containing imdb ratings for movies.
    :return:
    """

    # We split the ratings by year (to plot as seperate lines)
    def generate_imdb_dictionary(df):
        years = [year for year in range(2004, 2023, 1)]
        return {y: df[df['year_adjusted'] == y] for y in years}

    df_plot_dictionary = generate_imdb_dictionary(df)

    # we now plot these 20 individual lines onto the same graph
    for year in df_plot_dictionary:
        sns.distplot(df_plot_dictionary[year][['rating']], hist=False, rug=True, label=year)

    plt.legend()
    plt.xlabel("IMDb rating")
    plt.show()

    return df_plot_dictionary


def two_sample_kolmogorov_smirnov_test(df, year_one, year_two):
    """
    Function is to compare distribution of the two columns using two-sample Kolmogorov-Smirnov test, it is included in the scipy.stats:

    Here we look for whether the p values are less than 0.05 - if they are, it means they are statistically significant at 5%

    :param df: input imdb dataframe
    :param year_one: the first year of the two we want to compare
    :param year_two: the second year of the two we want to compare
    :return:
    """
    year_one_df = df[year_one]
    year_two_df = df[year_two]
    kolmogorov_smirnov_value = ks_2samp(year_one_df['rating'], year_two_df['rating'])
    return kolmogorov_smirnov_value


if __name__ == '__main__':
    df = get_average_imdb_ratings_all_movies()
    df = plot_imdb_distributions(df)

    ks_2samp_2015_2016 = two_sample_kolmogorov_smirnov_test(df, '2015', '2016')
    # ks_2samp_2015_2021 = two_sample_kolmogorov_smirnov_test(df, '2015', '2021')
    # ks_2samp_2012_2016 = two_sample_kolmogorov_smirnov_test(df, '2012', '2016')
    # ks_2samp_2015_2022 = two_sample_kolmogorov_smirnov_test(df, '2015', '2022')