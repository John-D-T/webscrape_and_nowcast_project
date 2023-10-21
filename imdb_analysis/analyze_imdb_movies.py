import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from common.latex_file_generator import save_df_as_image


def get_average_imdb_ratings_all_movies():
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
    extension = 'csv'
    path = os.path.join(os.getcwd(), 'genre')
    os.chdir(path)
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv("all_movies.csv", index=False, encoding='utf-8-sig')


def plot_imdb_distributions(df):

    # We split the ratings by year (to plot as seperate lines)
    imdb_2022 = df.loc[df['year_adjusted'] == 2022]
    imdb_2021 = df.loc[df['year_adjusted'] == 2021]
    imdb_2020 = df.loc[df['year_adjusted'] == 2020]
    imdb_2019 = df.loc[df['year_adjusted'] == 2019]
    imdb_2018 = df.loc[df['year_adjusted'] == 2018]
    imdb_2017 = df.loc[df['year_adjusted'] == 2017]
    imdb_2016 = df.loc[df['year_adjusted'] == 2016]
    imdb_2015 = df.loc[df['year_adjusted'] == 2015]
    imdb_2014 = df.loc[df['year_adjusted'] == 2014]
    imdb_2013 = df.loc[df['year_adjusted'] == 2013]
    imdb_2012 = df.loc[df['year_adjusted'] == 2012]
    imdb_2011 = df.loc[df['year_adjusted'] == 2011]
    imdb_2010 = df.loc[df['year_adjusted'] == 2010]
    imdb_2009 = df.loc[df['year_adjusted'] == 2009]
    imdb_2008 = df.loc[df['year_adjusted'] == 2008]
    imdb_2007 = df.loc[df['year_adjusted'] == 2007]
    imdb_2006 = df.loc[df['year_adjusted'] == 2006]
    imdb_2005 = df.loc[df['year_adjusted'] == 2005]
    imdb_2004 = df.loc[df['year_adjusted'] == 2004]

    sns.distplot(imdb_2004[['rating']], hist=False, rug=True, label='2004')
    sns.distplot(imdb_2005[['rating']], hist=False, rug=True, label='2005')
    sns.distplot(imdb_2006[['rating']], hist=False, rug=True, label='2006')
    sns.distplot(imdb_2007[['rating']], hist=False, rug=True, label='2007')
    sns.distplot(imdb_2008[['rating']], hist=False, rug=True, label='2008')
    sns.distplot(imdb_2009[['rating']], hist=False, rug=True, label='2009')
    sns.distplot(imdb_2010[['rating']], hist=False, rug=True, label='2010')
    sns.distplot(imdb_2011[['rating']], hist=False, rug=True, label='2011')
    sns.distplot(imdb_2012[['rating']], hist=False, rug=True, label='2012')
    sns.distplot(imdb_2013[['rating']], hist=False, rug=True, label='2013')
    sns.distplot(imdb_2014[['rating']], hist=False, rug=True, label='2014')
    sns.distplot(imdb_2015[['rating']], hist=False, rug=True, label='2015')
    sns.distplot(imdb_2016[['rating']], hist=False, rug=True, label='2016')
    sns.distplot(imdb_2017[['rating']], hist=False, rug=True, label='2017')
    sns.distplot(imdb_2018[['rating']], hist=False, rug=True, label='2018')
    sns.distplot(imdb_2019[['rating']], hist=False, rug=True, label='2019')
    sns.distplot(imdb_2020[['rating']], hist=False, rug=True, label='2020')
    sns.distplot(imdb_2021[['rating']], hist=False, rug=True, label='2021')
    sns.distplot(imdb_2022[['rating']], hist=False, rug=True, label='2022')
    plt.legend()
    plt.xlabel("IMDb rating")
    plt.show()

    # You can compare distribution of the two columns using two-sample Kolmogorov-Smirnov test, it is included in the scipy.stats:
    # Here we note that none of the p values are less than 0.05 - not statistically significant at 5%
    ks_2samp_2015_2016 = ks_2samp(imdb_2015['rating'], imdb_2016['rating'])
    ks_2samp_2015_2021 = ks_2samp(imdb_2015['rating'], imdb_2021['rating'])
    ks_2samp_2012_2016 = ks_2samp(imdb_2012['rating'], imdb_2016['rating'])
    ks_2samp_2015_2022 = ks_2samp(imdb_2015['rating'], imdb_2022['rating'])


if __name__ == '__main__':
    df = get_average_imdb_ratings_all_movies()
    plot_imdb_distributions(df)