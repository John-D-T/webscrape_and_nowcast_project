

import pandas as pd
import os
import fbprophet

'''
pip install fbprophet

read: https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/
'''

def analyze_bfi_net_box_office():
    gdp_file = 'quarterly_gdp_uk.csv'
    box_office_file = 'compiled_top_box_office.csv'

    # creating our time series df
    box_office_df = pd.read_csv(os.path.join(os.getcwd(), 'output', box_office_file))
    # todo - only take the weekly gross and date columns from box office df

    # creating our gdp df
    gdp_df = pd.read_csv(os.path.join(os.getcwd(), 'input', gdp_file))


if __name__ == '__main__':
    analyze_bfi_net_box_office()