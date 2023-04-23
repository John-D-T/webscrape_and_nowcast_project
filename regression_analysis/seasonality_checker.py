"""
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


def checking_google_trends_for_seasonality():
    # TODO - add seasonality checks for all google trend keywords
    # clearing out existing graphs
    pyplot.clf()

    filepath = os.path.join(os.path.dirname(os.getcwd()), 'google_trends_scraper', 'academy_awards.csv')
    df = read_csv(filepath, skiprows=[0,1])

    # TODO - figure out how to adjust x axis to fit months
    df.plot()
    pyplot.show()


def checking_bfi_box_office_for_seasonality():
    # clearing out existing graphs
    pyplot.clf()

    series = read_csv('.csv', header=0, index_col=0)
    series.plot()
    pyplot.show()


def checking_monthly_admissions_for_seasonality():
    # clearing out existing graphs
    pyplot.clf()

    series = read_csv('.csv', header=0, index_col=0)
    series.plot()
    pyplot.show()


if __name__ == '__main__':
    checking_google_trends_for_seasonality()
    # checking_bfi_box_office_for_seasonality()
    # checking_monthly_admissions_for_seasonality()