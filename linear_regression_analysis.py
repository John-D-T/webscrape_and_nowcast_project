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
csv_extension = '.csv'
academy_awards = 'academy_awards'
cinema_showings = 'cinema_showings'
cinemas_near_me = 'cinemas_near_me'
films = 'films'
films_near_me = 'films_near_me'
gdp_file = 'monthly_gdp_uk_v2_cleaned.csv'
box_office_file = 'compiled_top_15_box_office.csv'

def univariate_regression_box_office_gdp(gdp_df, box_office_df):

    # clearing out existing graphs
    plt.clf()

    # creating our dataframe to pass into regression
    merged_df = pd.merge(box_office_df, gdp_df, on=['date_grouped'])

    sns.regplot(x="monthly_gross", y="gdp", data=merged_df).set(title='Univariate regression of Weekend Gross vs GDP')

    plt.clf()
    sns.regplot(x="monthly_gross", y="gdp", data=merged_df, order=2)

def univariate_regression_monthly_admission_gdp(gdp_df, monthly_admission_df):

    # clearing out existing graphs
    plt.clf()

    # creating our time series df to pass into regression
    merged_df = pd.merge(monthly_admission_df, gdp_df, on=['date_grouped'])

    sns.regplot(x="monthly_admissions", y="gdp", data=merged_df).set(title='Univariate regression of Monthly Admissions vs GDP')

    plt.clf()
    sns.regplot(x="monthly_admissions", y="gdp", data=merged_df, order=2)

def multivariate_linear_regression(gdp_df, box_office_df, monthly_admissions_df, box_office_weightings_df, google_trends_df):
    # clearing out existing graphs
    plt.clf()

    merged_df = pd.merge(pd.merge(pd.merge(pd.merge(box_office_df, gdp_df, on=['date_grouped']), box_office_weightings_df, on=['date_grouped']), google_trends_df, on=['date_grouped']), monthly_admissions_df, on=['date_grouped'])

    # convert date to numerical value
    import datetime as ddt
    merged_df['date_grouped'] = pd.to_datetime(merged_df['date_grouped'])
    merged_df['date_grouped'] = merged_df['date_grouped'].map(ddt.datetime.toordinal)

    # TODO - add rest of independent variables
    X = merged_df[['date_grouped','monthly_gross','monthly_admissions', 'number_of_cinemas', 'monthly_gross_ratio_rank_1', 'monthly_gross_ratio_rank_2', 'monthly_gross_ratio_rank_3', 'monthly_gross_ratio_rank_4',
                   'monthly_gross_ratio_rank_5', 'monthly_gross_ratio_rank_6', 'monthly_gross_ratio_rank_7', 'monthly_gross_ratio_rank_8', 'monthly_gross_ratio_rank_9', 'monthly_gross_ratio_rank_10',
                   'monthly_gross_ratio_rank_11', 'monthly_gross_ratio_rank_12', 'monthly_gross_ratio_rank_13', 'monthly_gross_ratio_rank_14', 'monthly_gross_ratio_rank_15', 'frequency_academy_awards',
                   'frequency_cinema_showings', 'frequency_cinemas_near_me', 'frequency_films', 'frequency_films_near_me']]
    Y = merged_df['gdp']

    # initiating linear regression
    reg = LinearRegression()
    reg.fit(X, Y)

    Intercept = reg.intercept_
    Coefficients = reg.coef_

    print(Intercept)  # -2255.443730866293
    print(Coefficients)  # [ 3.46684283e-03 -3.36704042e-08  2.38061825e-07  2.02875346e-04, -2.06275364e+02 -2.13966644e+02 -1.96083756e+02 -2.02908698e+02, -2.11805028e+02 -2.61624826e+02 -1.93369650e+02 -2.22989899e+02, -1.25068905e+02 -3.61970849e+02 -1.98135846e+02  2.58697635e+00, -3.07260330e+02 -3.03507628e+02 -1.55415957e+02 -1.86969517e-03,  8.01293711e-02  3.56540510e-03 -5.27801170e-02 -7.05375150e-03]

    # TODO - WIP
    # statsmodel functionality, with more detail:
    X = add_constant(X)  # to add constant value in the model
    model = OLS(Y, X).fit()  # fitting the model
    predictions = model.summary()

    # summary of the OLS regression
    predictions


def create_gdp_df():
    ### creating our gdp df

    gdp_df = pd.read_csv(os.path.join(os.getcwd(), 'input', gdp_file))

    # filter on gdp df for 1. Time period we want and 2. Columns we want
    # filter on Title, gross value
    gdp_date_column = 'Month'
    monthly_gdp = 'Monthly GDP (A-T)'
    gdp_refine_df = gdp_df[[gdp_date_column, monthly_gdp]]

    # clean date column format to match DD-MM-YY format, similar to the box office df
    gdp_refine_df = gdp_refine_df.rename(columns={gdp_date_column: "date", monthly_gdp: "gdp"})

    gdp_refine_df['date'] = pd.to_datetime(gdp_refine_df['date'], format='%Y%b')
    gdp_refine_df['date_grouped'] = pd.to_datetime(gdp_refine_df['date']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

    return gdp_refine_df.drop(columns=['date'])

def create_box_office_weightings_df():

    ### creating box office df
    box_office_df = pd.read_csv(os.path.join(os.getcwd(), 'output', box_office_file))

    weekend_gross_column = 'Weekend Gross'
    box_office_date_column = 'date'
    no_of_cinemas = 'Number of cinemas'
    distributor = 'Distributor'
    title = 'Title'
    site_average = 'Site average'

    box_office_refine_df = box_office_df.drop(columns=[title, distributor, no_of_cinemas, site_average], axis='columns')
    #box_office_refine_df = box_office_df[[weekend_gross_column, box_office_date_column]]

    box_office_refine_df = box_office_refine_df.rename(columns={'Weekend Gross': 'weekend_gross'})

    box_office_refine_df['date'] = pd.to_datetime(box_office_refine_df['date'], format='%d-%m-%Y')

    box_office_refine_df['rank'] = box_office_refine_df.groupby('date')['weekend_gross'].rank(ascending=False)

    box_office_refine_df['total_weekend_gross'] = box_office_refine_df.groupby('date')['weekend_gross'].transform(np.sum)

    box_office_refine_df['weekend_gross_ratio'] = box_office_refine_df['weekend_gross'] / box_office_refine_df['total_weekend_gross']

    # aggregate data by month (to match frequency of GDP data)
    box_office_refine_df['date_grouped'] = pd.to_datetime(box_office_refine_df['date']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

    box_office_refine_df = box_office_refine_df.drop(columns=['date', 'total_weekend_gross', 'weekend_gross', 'Unnamed: 0', 'Unnamed: 0.1'])
    box_office_refine_df_by_month = box_office_refine_df.groupby(by=['rank', 'date_grouped'])['weekend_gross_ratio'].mean().reset_index()

    def categorize_row(row, i):
        if row['rank'] == i:
            return row['weekend_gross_ratio']

    # flattening dataframe
    for i in range(1, 16):
        box_office_refine_df_by_month['monthly_gross_ratio_rank_%s' % i] = box_office_refine_df_by_month.apply(lambda row: categorize_row(row, i), axis=1)

    box_office_refine_df_by_month = box_office_refine_df_by_month.drop(columns=['rank', 'weekend_gross_ratio'])

    # read https://stackoverflow.com/questions/52899858/collapsing-rows-with-nan-entries-in-pandas-dataframe
    box_office_final_df = box_office_refine_df_by_month.groupby('date_grouped', as_index=False).first()

    return box_office_final_df

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


    box_office_refine_df['date'] = pd.to_datetime(box_office_refine_df['date'], format='%d-%m-%Y')

    # aggregate data by month (to match frequency of GDP data)
    box_office_refine_df['date_grouped'] = pd.to_datetime(box_office_refine_df['date']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

    box_office_grouped_df = box_office_refine_df.groupby('date_grouped')['weekend_gross', 'number_of_cinemas'].sum()
    box_office_grouped_df = box_office_grouped_df.reset_index()

    box_office_grouped_df = box_office_grouped_df.rename(columns={'weekend_gross': 'monthly_gross'})

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
    monthly_admission_df = monthly_admission_df.rename(columns={'monthly admissions': 'monthly_admissions'})

    return monthly_admission_df


def create_google_trends_df():
    '''
    Load all google trend .csv's for key words - from 2004-2023
    :return: dataframe containing keywords
    '''

    # generating dataframes
    academy_awards_df = pd.read_csv(os.path.join(os.getcwd(), 'google_trends', academy_awards + csv_extension), skiprows=[0,1])\
        .rename(columns={'Academy Awards: (United Kingdom)': 'frequency_%s' % academy_awards})
    cinema_showings_df = pd.read_csv(os.path.join(os.getcwd(), 'google_trends', cinema_showings + csv_extension), skiprows=[0,1])\
        .rename(columns={'%s: (United Kingdom)' % cinema_showings.replace('_', ' '): 'frequency_%s' % cinema_showings})
    cinemas_near_me_df = pd.read_csv(os.path.join(os.getcwd(), 'google_trends', cinemas_near_me + csv_extension), skiprows=[0,1])\
        .rename(columns={'%s: (United Kingdom)' % cinemas_near_me.replace('_', ' '): 'frequency_%s' % cinemas_near_me})
    films_df = pd.read_csv(os.path.join(os.getcwd(), 'google_trends', films + csv_extension), skiprows=[0,1])\
        .rename(columns={'%s: (United Kingdom)' % films: 'frequency_%s' % films})
    films_near_me_df = pd.read_csv(os.path.join(os.getcwd(), 'google_trends', films_near_me + csv_extension), skiprows=[0,1])\
        .rename(columns={'%s: (United Kingdom)' % films_near_me.replace('_', ' '): 'frequency_%s' % films_near_me })

    # merge dataframes
    google_trends_df = pd.merge(pd.merge(pd.merge(pd.merge(academy_awards_df, cinema_showings_df, on='Month'), cinemas_near_me_df, on='Month'), films_df, on='Month'), films_near_me_df, on='Month')
    google_trends_df = google_trends_df.rename(columns={'Month': 'date_grouped'})
    google_trends_df['date_grouped'] = pd.to_datetime(google_trends_df['date_grouped']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
    return google_trends_df


if __name__ == '__main__':
    # Setting up config to avoid truncation of columns or column names:
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    # collecting independent variables
    google_trends_df = create_google_trends_df()  # google trends dataset

    gdp_df = create_gdp_df()  # monthly gdp dataset

    box_office_df = create_box_office_df()  # bfi box office dataset

    box_office_weightings_df = create_box_office_weightings_df()  # bfi box office dataset, with weightings

    monthly_admission_df = create_monthly_admission_df()  # monthly admissions dataset

    # creating regressions
    univariate_regression_box_office_gdp(gdp_df, box_office_df)

    univariate_regression_monthly_admission_gdp(gdp_df, monthly_admission_df)

    multivariate_linear_regression(gdp_df, box_office_df, monthly_admission_df, box_office_weightings_df, google_trends_df)