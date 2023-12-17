import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from imdb_analysis.plots import plot_average_imdb_ratings_per_year


def concatenate_imdb_movies(extension='csv'):
    """
    One time function to combine all the individual imdb files into one.
    :return:
    """
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
    df = plot_average_imdb_ratings_per_year(gross_filter=1000000, year_adjusted_filter=1999)
    df = plot_imdb_distributions(df)

    ks_2samp_2015_2016 = two_sample_kolmogorov_smirnov_test(df, '2015', '2016')
    # ks_2samp_2015_2021 = two_sample_kolmogorov_smirnov_test(df, '2015', '2021')
    # ks_2samp_2012_2016 = two_sample_kolmogorov_smirnov_test(df, '2012', '2016')
    # ks_2samp_2015_2022 = two_sample_kolmogorov_smirnov_test(df, '2015', '2022')