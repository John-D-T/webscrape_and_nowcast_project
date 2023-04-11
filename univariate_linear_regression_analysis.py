

import pandas as pd
import numpy as np
import os
import sklearn
import scipy
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS

'''
3.8 SHAKY, BUT WORKS ON 3.7 VENV - different versions = more compatible? No issues when installing scipy and scikit learn

Linear regression notes:
https://towardsdatascience.com/demystifying-ml-part1-basic-terminology-linear-regression-a89500a9e
https://medium.com/analytics-vidhya/the-pitfalls-of-linear-regression-and-how-to-avoid-them-b93626e1a020
https://towardsdatascience.com/regression-plots-in-python-with-seaborn-118472b12e3d
https://codeburst.io/multiple-linear-regression-sklearn-and-statsmodels-798750747755


'''

def univariate_regression_box_office_gva(gdp_df, box_office_df):

    # clearing out existing graphs
    plt.clf()

    # creating our dataframe to pass into regression
    merged_df = pd.merge(box_office_df, gdp_df, on=['date_grouped'])

    sns.regplot(x="weekend_gross", y="GVA", data=merged_df).set(title='Univariate regression of Weekend Gross vs GVA')

    plt.clf()
    sns.regplot(x="weekend_gross", y="GVA", data=merged_df, order=2)

def univariate_regression_monthly_admission_gva(gdp_df, monthly_admission_df):

    # clearing out existing graphs
    plt.clf()

    # creating our time series df to pass into regression
    merged_df = pd.merge(monthly_admission_df, gdp_df, on=['date_grouped'])

    sns.regplot(x="monthly admissions", y="GVA", data=merged_df).set(title='Univariate regression of Monthly Admissions vs GVA')

    plt.clf()
    sns.regplot(x="monthly admissions", y="GVA", data=merged_df, order=2)

def multivariate_linear_regression(gdp_df, box_office_df, monthly_admissions_df):
    # clearing out existing graphs
    plt.clf()

    merged_df = pd.merge(box_office_df, gdp_df, on=['date_grouped'])
    merged_df = pd.merge(merged_df, monthly_admission_df, on=['date_grouped'])

    # convert date to numerical value
    import datetime as ddt
    merged_df['date_grouped'] = pd.to_datetime(merged_df['date_grouped'])
    merged_df['date_grouped'] = merged_df['date_grouped'].map(ddt.datetime.toordinal)

    X = merged_df[['date_grouped','weekend_gross','monthly admissions']]
    Y = merged_df['GVA']

    # initiating linear regression
    reg = LinearRegression()
    reg.fit(X, Y)

    Intercept = reg.intercept_
    Coefficients = reg.coef_

    print(Intercept)  # -2931.893
    print(Coefficients)  # [4.10063018e-03 1.34416632e-08 4.43920342e-07]

    # statsmodel functionality, with more detail:
    X = add_constant(X)  # to add constant value in the model
    model = OLS(Y, X).fit()  # fitting the model
    predictions = model.summary()

    # summary of the OLS regression
    predictions


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
    monthly_admission_df = pd.read_csv(os.path.join(os.getcwd(), 'input', monthly_admissions_file), thousands=',')

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

    univariate_regression_box_office_gva(gdp_df, box_office_df)

    univariate_regression_monthly_admission_gva(gdp_df, monthly_admission_df)

    multivariate_linear_regression(gdp_df, box_office_df, monthly_admission_df)