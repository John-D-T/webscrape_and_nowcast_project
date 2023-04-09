

import pandas as pd
import numpy as np
import os
import sklearn
import scipy
import seaborn

'''
3.8 SHAKY, BUT WORKS ON 3.7 VENV - different versions = more compatible? No issues when installing scipy and scikit learn

Linear regression notes:
https://towardsdatascience.com/demystifying-ml-part1-basic-terminology-linear-regression-a89500a9e
'''

def create_time_series_df(gdp_df, box_office_df, monthly_admission_df):

    # creating our time series df to pass into multivariate regression
    merged_df = pd.merge(box_office_df, gdp_df, on=['date_grouped'])
    merged_df = pd.merge(merged_df, monthly_admission_df, on=['date_grouped'])




def create_gdp_df():
    gdp_file = 'monthly_gva_uk_cleaned.csv'
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
    gdp_refine_df['date_grouped'] = pd.to_datetime(gdp_refine_df['date']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

    return gdp_refine_df.drop(columns=['date'])

def create_box_office_df():
    box_office_file = 'compiled_top_15_box_office.csv'

    ### creating box office df
    box_office_df = pd.read_csv(os.path.join(os.getcwd(), 'output', box_office_file))

    weekend_gross_column = 'Weekend Gross'
    box_office_date_column = 'date'
    no_of_cinemas = 'Number of cinemas'
    distributor = 'Distributor'
    box_office_refine_df = box_office_df[[weekend_gross_column, distributor, box_office_date_column, no_of_cinemas]]

    box_office_refine_df = box_office_refine_df.rename(columns={'Weekend Gross': 'weekend_gross', 'Number of cinemas': 'number_of_cinemas', 'Distributor': 'distributor'})

    # box_office_refine_df['total_weekend_gross'] = box_office_refine_df.groupby('date')['Weekend Gross'].transform(np.sum)
    box_office_refine_df['date'] = pd.to_datetime(box_office_refine_df['date'], format='%d-%m-%Y')

    # aggregate data by month (to match frequency of GVA/GDP data)
    box_office_refine_df['date_grouped'] = pd.to_datetime(box_office_refine_df['date']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

    box_office_grouped_df = box_office_refine_df.groupby('date_grouped')['weekend_gross', 'number_of_cinemas'].sum()
    box_office_grouped_df = box_office_grouped_df.reset_index()

    '''
    to make sure that box_office_grouped_df aggregates as expected, we check 
    1. box_office_refine_df['date_grouped'] = box_office_refine_df['date_grouped'].astype("string")
       subset = box_office_refine_df.query('date_grouped.str.contains("2007-10")', engine='python')
       subset.sum()
    -- 35,475,564 & 15923
    vs 
    2. box_office_grouped_df[box_office_grouped_df_2.date_grouped == '2007-10']
    -- 35,475,564 & 15923
    '''
    return box_office_grouped_df

def create_monthly_admission_df():
    '''
    CSV taken from https://www.cinemauk.org.uk/the-industry/facts-and-figures/uk-cinema-admissions-and-box-office/monthly-admissions/
    :return:
    '''
    monthly_admissions_file = 'monthly_admissions_uk.csv'

    # creating monthly admissions df
    monthly_admission_df = pd.read_csv(os.path.join(os.getcwd(), 'input', monthly_admissions_file))

    return monthly_admission_df

def create_google_trends_df():
    return ''

if __name__ == '__main__':
    # Setting up config to avoid truncation of columns or column names:
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    google_trends_df = create_google_trends_df()

    gdp_df = create_gdp_df()

    box_office_df = create_box_office_df()

    monthly_admission_df = create_monthly_admission_df()

    create_time_series_df(gdp_df, box_office_df, monthly_admission_df)