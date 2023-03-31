

import pandas as pd
import numpy as np
import os
import sktime

'''
pip install cython

# then following these steps: https://www.sktime.net/en/latest/installation.html#development-versions
git clone https://github.com/sktime/sktime.git
cd sktime
git checkout main
git pull

pip install sktime --seems to work now

read: https://analyticsindiamag.com/sktime-library/
https://towardsdatascience.com/sktime-a-unified-python-library-for-time-series-machine-learning-3c103c139a55
'''

def analyze_bfi_net_box_office():
    gdp_file = 'monthly_gdp_uk_cleaned.csv'
    box_office_file = 'compiled_top_15_box_office.csv'

    # creating our time series df
    ### creating box office df
    # todo - look to switch to the 15 movie one? Can just run a group by sum to get the box office. can ALSO get number
    # todo - of cinemas, site average, and maybe more
    box_office_df = pd.read_csv(os.path.join(os.getcwd(), 'output', box_office_file))

    # todo - only take the weekly gross and date columns from box office df
    weekend_gross_column = 'Weekend Gross'
    box_office_date_column = 'date'
    no_of_cinemas = 'Number of cinemas'
    distributor = 'Distributor'
    box_office_refine_df = box_office_df[[weekend_gross_column, distributor, box_office_date_column, no_of_cinemas]]

    box_office_refine_df['total_weekend_gross'] = box_office_refine_df.groupby('date')['Weekend Gross'].transform(np.sum)



    ### creating our gdp df

    gdp_df = pd.read_csv(os.path.join(os.getcwd(), 'input', gdp_file))

    # filter on gdp df for 1. Time period we want and 2. Columns we want
    # filter on Title, gross value
    gdp_date_column = 'Title'
    gross_value_column = 'Gross Value Added - Monthly (Index 1dp) :CVM SA'
    gdp_refine_df = gdp_df[[gdp_date_column, gross_value_column]]


if __name__ == '__main__':
    # Setting up config to avoid truncation of columns or column names:
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    analyze_bfi_net_box_office()