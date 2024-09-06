"""
PYTHON 3.11 (64 BIT)

pip install seaborn
pip install scipy
pip install matplotlib
pip install texttable
pip install latextable
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from imdb_analysis.plots import plot_average_imdb_ratings_per_year


def concatenate_imdb_movies(extension='csv', file_name="all_movies.csv"):
    """
    One time function to combine all the individual imdb files into one.
    :return:
    """
    os.chdir(os.path.join(os.getcwd(), 'genre'))

    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv(file_name, index=False, encoding='utf-8-sig')


def plot_imdb_distributions(df, range_start=2004, range_end=2023, range_interval=1):
    """
    Function to plot ratings for all imdb movies over the years

def plot_imdb_distributions(df, range_start=2004, range_end=2023, range_interval=1):
    """
    Function to plot ratings for all imdb movies over the years
    :param df: dataframe containing imdb ratings for movies.
    :param range_start: start of year range (default is 2004)
    :param range_end: end of year range (default is 2023)
    :param range_interval: interval of year range (default is 1)
    :return:
    """

    # We split the ratings by year (to plot as seperate lines)
    df_plot_dictionary = generate_imdb_dictionary_split_by_year(df=df, range_start=range_start,
                                                                range_end=range_end,
                                                                range_interval=range_interval)

    # we now plot these 20 individual lines onto the same graph
    for year in df_plot_dictionary:
        sns.distplot(df_plot_dictionary[year][['rating']], hist=False, rug=True, label=year)

    plt.legend()
    plt.xlabel("IMDb rating")
    plt.show()

    return df_plot_dictionary


def generate_imdb_dictionary_split_by_year(df, range_start, range_end, range_interval):
    """
    Function which processes a dataframe containing imdb ratings, splits these up by year,
    then returns a dictionary with the key:value pairs being:
        key: year
        value: dataframe filtered on the 'year-adjusted' column equalling that year

    :param df: dataframe containing imdb ratings for movies.
    :param range_start: start of year range (default is 2004)
    :param range_end: end of year range (default is 2023)
    :param range_interval: interval of year range (default is 1)

    :return: A dictionary containing:
        key: year
        value: dataframe filtered on the 'year-adjusted' column equalling that year
    """
    years = [year for year in range(range_start, range_end, range_interval)]
    return {y: df[df['year-adjusted'] == y] for y in years}



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
    df = plot_average_imdb_ratings_per_year(gross_filter=1000000, year_adjusted_filter=1999)
    df = plot_imdb_distributions(df)

    ks_2samp_2015_2016 = two_sample_kolmogorov_smirnov_test(df=df, year_one='2015', year_two='2016')
    # ks_2samp_2015_2021 = two_sample_kolmogorov_smirnov_test(df, '2015', '2021')
    # ks_2samp_2012_2016 = two_sample_kolmogorov_smirnov_test(df, '2012', '2016')
    # ks_2samp_2015_2022 = two_sample_kolmogorov_smirnov_test(df, '2015', '2022')