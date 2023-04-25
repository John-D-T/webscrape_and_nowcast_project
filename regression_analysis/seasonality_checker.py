"""
PYTHON 3.7 (64 BIT)

Plot all data sources - see if there is seasonality (regular fluctuations)
   Seasonally adjust if needed

interactive dummy?
Also can eyeball residuals, to see first if it's worth adding.

read: https://towardsdatascience.com/finding-seasonal-trends-in-time-series-data-with-python-ce10c37aa861
      https://machinelearningmastery.com/time-series-seasonality-with-python/
"""

from pandas import read_csv
from matplotlib import pyplot
import os
from regression_analysis.linear_regression_analysis import create_box_office_df

from statsmodels.tsa.seasonal import seasonal_decompose

def checking_google_trends_for_seasonality():
    # TODO - add seasonality checks for all google trend keywords
    # clearing out existing graphs
    pyplot.clf()

    filepath = os.path.join(os.path.dirname(os.getcwd()), 'google_trends_scraper', 'academy_awards.csv')

    # index_col helps adjust x axis to fit months
    df = read_csv(filepath, skiprows=[0,1], index_col="Month")

    ax = df.plot(title='')

    ax.set(xlabel='x axis', ylabel='y axis')
    pyplot.show()


def checking_bfi_box_office_for_seasonality():
    # clearing out existing graphs
    pyplot.clf()

    df = create_box_office_df()

    df = df.set_index('date_grouped')
    df_subset = df['monthly_gross']

    # TODO - figure out how to plot label for y-axis
    ax = df_subset.plot(title='')

    ax.set(xlabel='x axis', ylabel='y axis')
    pyplot.show()

    # TODO - fluctuations not as visually obvious, so add python checked for seasonality

    analysis = df[['monthly_gross']].copy()

    # TODO - https://stackoverflow.com/questions/60017052/decompose-for-time-series-valueerror-you-must-specify-a-period-or-x-must-be
    decompose_result_mult = seasonal_decompose(analysis, model="multiplicative")

    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid

    decompose_result_mult.plot()

def checking_monthly_admissions_for_seasonality():
    # clearing out existing graphs
    pyplot.clf()

    series = read_csv('.csv', header=0, index_col=0)
    series.plot()
    pyplot.show()


if __name__ == '__main__':
    #checking_google_trends_for_seasonality()
    checking_bfi_box_office_for_seasonality()
    # checking_monthly_admissions_for_seasonality()