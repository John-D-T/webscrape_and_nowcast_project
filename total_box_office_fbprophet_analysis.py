

import pandas as pd
import os
import fbprophet

'''
pip install fbprophet

read: https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/
'''

def analyze_bfi_net_box_office():
    gdp_file = 'monthly_gdp_uk.csv'
    box_office_file = 'compiled_top_15_box_office.csv'

    # creating our time series df
    # creating box office df
    # todo - look to switch to the 15 movie one? Can just run a group by sum to get the box office. can ALSO get number
    # todo - of cinemas, site average, and maybe more
    box_office_df = pd.read_csv(os.path.join(os.getcwd(), 'output', box_office_file))
    # todo - only take the weekly gross and date columns from box office df
    weekend_gross_column = ''
    box_office_date_column = ''
    box_office_refine_df = box_office_df[[weekend_gross_column, box_office_date_column]]

    # creating our gdp df

    gdp_df = pd.read_csv(os.path.join(os.getcwd(), 'input', gdp_file))

    # filter on gdp df for 1. Time period we want and 2. Columns we want
    # filter on Title, gross value
    gdp_date_column = 'Title'
    gross_value_column = 'Gross Value Added - Monthly (Index 1dp) :CVM SA'
    gdp_refine_df = gdp_df[[gdp_date_column, gross_value_column]]


if __name__ == '__main__':
    analyze_bfi_net_box_office()