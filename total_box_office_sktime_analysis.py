

import pandas as pd
import numpy as np
import os
import sktime
from sktime.datasets import load_airline
'''
Issues are to do with installing skikit-learn (1.3.0) and scipy (1.3.2)

import wheel.pep425tags as w
print(w.get_supported(archive_root=''))
cp38-none-any is compatible - find correct wheel (looks promising here - https://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn)

then install that wheel of scikit-learn and scipy (downloaded wheel beforehand)
then pip install sktime


ImportError: DLL load failed while importing _qhull: The specified module could not be found.
https://stackoverflow.com/questions/63613167/pycharm-error-dll-load-failed-while-importing-qhull-the-specified-module-could


Usage:
read: https://analyticsindiamag.com/sktime-library/
https://towardsdatascience.com/sktime-a-unified-python-library-for-time-series-machine-learning-3c103c139a55
'''

def analyze_bfi_net_box_office():
    gdp_file = 'monthly_gdp_uk_cleaned.csv'
    box_office_file = 'compiled_top_15_box_office.csv'

    # creating our time series df
    ### creating box office df
    box_office_df = pd.read_csv(os.path.join(os.getcwd(), 'output', box_office_file))

    # todo - only take the weekly gross and date columns from box office df
    weekend_gross_column = 'Weekend Gross'
    box_office_date_column = 'date'
    no_of_cinemas = 'Number of cinemas'
    distributor = 'Distributor'
    box_office_refine_df = box_office_df[[weekend_gross_column, distributor, box_office_date_column, no_of_cinemas]]

    box_office_refine_df['total_weekend_gross'] = box_office_refine_df.groupby('date')['Weekend Gross'].transform(np.sum)
    box_office_refine_df['date'] = pd.to_datetime(box_office_refine_df['date'], format='%d-%m-%Y')

    # TODO - aggregate data by month (to match frequency of GVA/GDP data

    ### creating our gdp df

    gdp_df = pd.read_csv(os.path.join(os.getcwd(), 'input', gdp_file))

    # filter on gdp df for 1. Time period we want and 2. Columns we want
    # filter on Title, gross value
    gdp_date_column = 'Title'
    gross_value_column = 'Gross Value Added - Monthly (Index 1dp) :CVM SA'
    gdp_refine_df = gdp_df[[gdp_date_column, gross_value_column]]

    # clean date column format to match DD-MM-YY format, similar to the box office df
    gdp_refine_df = gdp_refine_df.rename(columns={gdp_date_column: "date", gross_value_column: "GVA"})

    gdp_refine_df['date'] = pd.to_datetime(gdp_refine_df['date'], format='%Y %b')

    # merge the two df - to then pass into sktime
    merged_df = pd.concat(objs=[box_office_refine_df, gdp_refine_df], keys='date')

if __name__ == '__main__':
    # Setting up config to avoid truncation of columns or column names:
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    analyze_bfi_net_box_office()