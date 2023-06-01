"""
PYTHON 3.7 (64 BIT)

Plot all data sources - see if there is seasonality (regular fluctuations)
   Seasonally adjust if needed

interactive dummy?
Also can eyeball residuals, to see first if it's worth adding.

read: https://towardsdatascience.com/finding-seasonal-trends-in-time-series-data-with-python-ce10c37aa861
      https://machinelearningmastery.com/time-series-seasonality-with-python/
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot
import os
from regression_analysis.linear_regression_analysis import GeneratingDataSourceDataframes

from statsmodels.tsa.seasonal import seasonal_decompose

def checking_google_trends_for_seasonality():
    # TODO - add seasonality checks for all google trend keywords
    # clearing out existing graphs
    pyplot.clf()

    filepath = os.path.join(os.path.dirname(os.getcwd()), 'google_trends_scraper', 'academy_awards.csv')

    # index_col helps adjust x axis to fit months
    df = pd.read_csv(filepath, skiprows=[0,1], index_col="Month")

    ax = df.plot(title='')

    ax.set(xlabel='x axis', ylabel='y axis')
    pyplot.show()


def checking_bfi_box_office_for_seasonality():
    """
    Check for seasonality. We remove seasonality by adding the variable as a control.
    :return:
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    # clearing out existing graphs
    pyplot.clf()

    gen = GeneratingDataSourceDataframes()

    df = gen.create_box_office_df()

    # converting date to a yyyy_mm_dd_format in order to fit the 'MS' frequency for the modelling later
    df['date_yyyy_mm_dd'] = pd.to_datetime(['{}-01'.format(y_m) for y_m in df.date_grouped])

    df = df.sort_values(by='date_yyyy_mm_dd', ascending=False)
    df = df.set_index('date_yyyy_mm_dd')
    df_subset = df['monthly_gross']

    # TODO - figure what the title, xlabel, and ylabel should be
    ax = df_subset.plot(title='')

    ax.set(xlabel='x axis', ylabel='y axis')
    pyplot.show()

    # Fluctuations not as visually obvious, so add python checked for seasonality

    # Converting the NaN to zeroes (because box office was actually 0 during covid)
    # Might need to cut down the range of dates then (won't work with multiplicative model)
    df_subset = df_subset.asfreq('MS').fillna(0)

    decompose_result_mult = seasonal_decompose(df_subset, model="additive")

    observed = decompose_result_mult.observed
    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid

    decompose_result_mult.plot()

def checking_monthly_admissions_for_seasonality():
    # clearing out existing graphs
    pyplot.clf()

    series = pd.read_csv('.csv', header=0, index_col=0)
    series.plot()
    pyplot.show()


if __name__ == '__main__':
    #checking_google_trends_for_seasonality()
    checking_bfi_box_office_for_seasonality()
    # checking_monthly_admissions_for_seasonality()