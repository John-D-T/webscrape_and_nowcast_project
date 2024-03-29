import datetime as ddt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels import IV2SLS
import scipy.stats as scs
# pip install scikit-learn (to use sklearn)
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
import statsmodels.stats.api as sms
import statsmodels.api as sm
from common.latex_file_generator import save_table_as_latex

from common.constants import ScrapingProjectConstants as c
from common.latex_file_generator import save_model_as_image
from regression_analysis.machine_learning_code import nowcast_regression_revamped
from regression_analysis.multicollinearity_checker import checking_all_independent_variables_for_collinearity
from twitter_scraper.sentiment_analysis_on_tweets import sentiment_analysis

"""
PYTHON 3.7 (64 BIT) - Found to be more compatible. No issues when installing scipy and scikit learn

pip install seaborn scipy scikit-learn statsmodel linearmodels

Linear regression notes:
https://towardsdatascience.com/demystifying-ml-part1-basic-terminology-linear-regression-a89500a9e
https://medium.com/analytics-vidhya/the-pitfalls-of-linear-regression-and-how-to-avoid-them-b93626e1a020
https://towardsdatascience.com/regression-plots-in-python-with-seaborn-118472b12e3d
https://codeburst.io/multiple-linear-regression-sklearn-and-statsmodels-798750747755
"""

def univariate_regression_box_office_gdp(gdp_df, box_office_df):

    # clearing out existing graphs
    plt.clf()

    # creating our dataframe to pass into regression
    merged_df = pd.merge(box_office_df, gdp_df, on=['date_grouped'])

    sns.regplot(x="monthly_gross", y="gdp", data=merged_df).set(title='Univariate regression of Weekend Gross vs GDP')

    plt.clf()
    sns.regplot(x="monthly_gross", y="gdp", data=merged_df, order=2)

    # convert date to numerical value
    import datetime as ddt
    merged_df['date_grouped'] = pd.to_datetime(merged_df['date_grouped'])
    merged_df['date_grouped'] = merged_df['date_grouped'].map(ddt.datetime.toordinal)

    X = merged_df[['date_grouped', 'monthly_gross']]
    Y = merged_df['gdp']

    # initiating linear regression
    reg = LinearRegression()
    reg.fit(X, Y)

    # statsmodel functionality, with more detail:
    X = add_constant(X)  # to add constant value in the model
    model = OLS(Y, X).fit()  # fitting the model

    # summary of the OLS regression - https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
    save_model_as_image(model=model, file_name='univariate_regression_box_office_gdp')

def univariate_regression_monthly_admission_gdp(gdp_df, monthly_admission_df):

    # clearing out existing graphs
    plt.clf()

    # creating our time series df to pass into regression
    merged_df = pd.merge(monthly_admission_df, gdp_df, on=['date_grouped'])

    sns.regplot(x="monthly_admissions", y="gdp", data=merged_df).set(title='Univariate regression of Monthly Admissions vs GDP')

    plt.clf()
    sns.regplot(x="monthly_admissions", y="gdp", data=merged_df, order=2)

    # convert date to numerical value
    import datetime as ddt
    merged_df['date_grouped'] = pd.to_datetime(merged_df['date_grouped'])
    merged_df['date_grouped'] = merged_df['date_grouped'].map(ddt.datetime.toordinal)

    X = merged_df[['date_grouped', 'monthly_admissions']]
    Y = merged_df['gdp']

    # initiating linear regression
    reg = LinearRegression()
    reg.fit(X, Y)

    # statsmodel functionality, with more detail:
    X = add_constant(X)  # to add constant value in the model
    model = OLS(Y, X).fit()  # fitting the model

    # summary of the OLS regression - https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
    predictions = model.summary()

    # https://economics.stackexchange.com/questions/11774/outputting-regressions-as-table-in-python-similar-to-outreg-in-stata
    prediction_latex = predictions.as_latex()
    beginningtex = """\\documentclass{report}
    \\usepackage{booktabs}
    \\begin{document}"""
    endtex = "\end{document}"

    f = open('univariate_regression_monthly_admission_gdp.tex', 'w+')
    f.write(beginningtex)
    f.write(prediction_latex)
    f.write(endtex)
    f.close()

def multivariate_linear_regression_pre_covid(gdp_df, weather_df, box_office_df, monthly_admissions_df, box_office_weightings_df, google_trends_df, twitter_scrape_df):
    '''
    Preparing regression input
    '''

    # clearing out existing graphs
    plt.clf()

    merged_df = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(box_office_df, gdp_df, on=['date_grouped']), box_office_weightings_df, on=['date_grouped']), google_trends_df, on=['date_grouped']), weather_df, on=['date_grouped']), twitter_scrape_df, on=['date_grouped'])

    merged_df['weighted_ranking'] = merged_df['monthly_gross_ratio_rank_1'] * 1 + merged_df['monthly_gross_ratio_rank_2'] * 2 \
                                    + merged_df['monthly_gross_ratio_rank_3'] * 3 + merged_df['monthly_gross_ratio_rank_4'] * 4 \
                                    + merged_df['monthly_gross_ratio_rank_5'] * 5 + merged_df['monthly_gross_ratio_rank_6'] * 6 \
                                    + merged_df['monthly_gross_ratio_rank_7'] * 7 + merged_df['monthly_gross_ratio_rank_8'] * 8 \
                                    + merged_df['monthly_gross_ratio_rank_9'] * 9 + merged_df['monthly_gross_ratio_rank_10'] * 10 \
                                    + merged_df['monthly_gross_ratio_rank_11'] * 11 + merged_df['monthly_gross_ratio_rank_12'] * 12 \
                                    + merged_df['monthly_gross_ratio_rank_13'] * 13 + merged_df['monthly_gross_ratio_rank_14'] * 14 \
                                    + merged_df['monthly_gross_ratio_rank_15'] * 15

    features = checking_all_independent_variables_for_collinearity(df=merged_df)

    merged_df['date_grouped'] = pd.to_datetime(merged_df['date_grouped'])

    # Set the cutoff date, based on when covid started in the UK
    cutoff_date = pd.to_datetime('2020-02-01')

    # Filter the DataFrame
    merged_df = merged_df[merged_df['date_grouped'] < cutoff_date]

    merged_df = merged_df.sort_values(by='date_grouped')

    merged_df['date_grouped_ordinal'] = merged_df['date_grouped'].map(ddt.datetime.toordinal)

    # Add lags for the dependent variable
    merged_df['gdp_lag1'] = merged_df['gdp'].shift(1)

    # have to filter out null values in gdp_lag1 - losing dimensionality?
    merged_df = merged_df.dropna(subset=["gdp_lag1"])
    features.append('gdp_lag1')
    x_ols = merged_df[features]
    y = merged_df['gdp']
    y_with_date = merged_df[['gdp', 'date_grouped']]

    features_alt = features
    features_alt.append('date_grouped')
    x_ols_alt = merged_df[features_alt]

    x_ols = add_constant(x_ols)    # to add constant value in the model, to tell us to fit for the b in 'y = mx + b'

    '''
    Now plotting regression
    '''


    # OLS Regression using linearmodels - Has robust covariance
    ols_model = IV2SLS(dependent=y, exog=x_ols, endog=None, instruments=None).fit()
    # Leaving this out as we're going for a nowcast
    # # Summary of the OLS regression - https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
    # save_model_as_image(model=ols_model, file_name='multivariate_ols_regression', lin_reg=True)

    '''
    Now checking residuals
    '''

    # QQ Plot - https://towardsdatascience.com/q-q-plots-explained-5aa8495426c0
    fig = plt.figure()
    ax = fig.add_subplot()
    scs.probplot(ols_model.resids, dist='norm', plot=plt)

    ax.get_lines()[0].set_markerfacecolor('black')
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[1].set_color('black')
    plt.title('QQ Plot')
    plt.show()

    plt.clf()

    # Plot outliers and check if any residuals are above 4 or < -4
    merged_df['residuals'] = ols_model.resids
    max_residual = merged_df['residuals'].max()  # 1.164928414420217
    min_residual = merged_df['residuals'].min()  # -1.3439102445690594

    # Finding residuals which are > 0.6 and < -0.6
    filtered_merged_df = merged_df[(merged_df.residuals > 0.6) | (merged_df.residuals < -0.6)]
    filtered_merged_df = filtered_merged_df[['date_grouped', 'residuals']]
    plt.plot(filtered_merged_df['date_grouped'], filtered_merged_df['residuals'], 'o', color='black')

    # Check for serial correlation using a Durbin Watson test - https://www.statology.org/durbin-watson-test-python/
    dw_test = durbin_watson(ols_model.resids) # 2.8126133956116295

    # Homocedasticity test
        # Ho = Homocedasticity = P > 0.05
        # Ha = There's no homocedasticity = p <=0.05

    model = sm.OLS(y, x_ols).fit()

    stat, p, f, fp = sms.het_breuschpagan(model.resid, model.model.exog)

    print(f'Test stat: {stat}') # 17.91253203099973
    print(f'p-Value: {p}') # 0.012370934841518211
    print(f'F-Value: {f}') # 2.81236654211901
    print(f'f_p_value: {fp}') # 0.009855514708688591

    plt.scatter(y=model.resid, x=model.predict(), color='black', alpha=0.5, s=20)
    plt.hlines(y=0, xmin=0, xmax=4, color='orange')
    plt.xlim(80,100)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Plotted residuals')
    plt.show()

    # Generate Latex Table with all results
    rows = [['Model', 'durbin watson', 'min residual', 'max residual', 'het breuschpagan p value'],
            ['Pre covid LR', dw_test, min_residual, max_residual, p]]

    save_table_as_latex(file_name='linear_reg_violation_pre_covid', rows=rows, header_count=5,
                        caption="Linear Regression violation checks (pre covid)")

    '''
    Nowcasting model
    '''
    var_variables = features
    var_variables.append('gdp')
    var_variables.append('date_grouped')
    var_df = merged_df[var_variables]

    nowcast_regression_revamped(var_df, x_ols_alt, y, y_with_date)

    return rows


def multivariate_linear_regression_incl_covid(gdp_df, weather_df, box_office_df, box_office_weightings_df,
                                              google_trends_df, twitter_scrape_df, rows, covid_check=False):
    '''
    Preparing regression input
    '''

    # clearing out existing graphs
    plt.clf()

    merged_df = pd.merge(pd.merge(pd.merge(pd.merge(box_office_df, gdp_df, on=['date_grouped']), box_office_weightings_df, on=['date_grouped']), google_trends_df, on=['date_grouped']), weather_df, on=['date_grouped'])

    merged_df['weighted_ranking'] = merged_df['monthly_gross_ratio_rank_1'] * 1 + merged_df['monthly_gross_ratio_rank_2'] * 2 \
                                    + merged_df['monthly_gross_ratio_rank_3'] * 3 + merged_df['monthly_gross_ratio_rank_4'] * 4 \
                                    + merged_df['monthly_gross_ratio_rank_5'] * 5 + merged_df['monthly_gross_ratio_rank_6'] * 6 \
                                    + merged_df['monthly_gross_ratio_rank_7'] * 7 + merged_df['monthly_gross_ratio_rank_8'] * 8 \
                                    + merged_df['monthly_gross_ratio_rank_9'] * 9 + merged_df['monthly_gross_ratio_rank_10'] * 10 \
                                    + merged_df['monthly_gross_ratio_rank_11'] * 11 + merged_df['monthly_gross_ratio_rank_12'] * 12 \
                                    + merged_df['monthly_gross_ratio_rank_13'] * 13 + merged_df['monthly_gross_ratio_rank_14'] * 14 \
                                    + merged_df['monthly_gross_ratio_rank_15'] * 15

    features = checking_all_independent_variables_for_collinearity(df = merged_df, covid=True)

    merged_df['date_grouped'] = pd.to_datetime(merged_df['date_grouped'])

    # rename columns to fix issue where the underscores for monthly_gross and frequency_academy_awards mess up the syntax
    # merged_df.rename(columns={"monthly_gross": "monthly gross"}, inplace=True)

    # Add dummy variable for covid lockdown
    list_of_months = [pd.to_datetime('2020-03-01'), pd.to_datetime('2020-04-01'), pd.to_datetime('2020-05-01'),
                      pd.to_datetime('2020-06-01'), pd.to_datetime('2020-07-01'), pd.to_datetime('2020-09-01'),
                      pd.to_datetime('2020-10-01'), pd.to_datetime('2020-11-01'), pd.to_datetime('2020-12-01'),
                      pd.to_datetime('2021-01-01'), pd.to_datetime('2021-02-01'), pd.to_datetime('2021-03-01'),
                      pd.to_datetime('2021-04-01'), pd.to_datetime('2021-05-01')]
    merged_df['cinema_lockdown'] = merged_df['date_grouped'].apply(lambda x: 1 if x in list_of_months else 0)

    #merged_df['date_grouped'] = merged_df['date_grouped'].map(ddt.datetime.toordinal)

    # Add lags for the dependent variable
    merged_df['gdp_lag1'] = merged_df['gdp'].shift(1)

    # have to filter out null values in gdp_lag1 - losing dimensionality?
    merged_df = merged_df.dropna(subset=["gdp_lag1"])

    features.extend(['gdp_lag1', 'cinema_lockdown'])
    x_ols = merged_df[features]
    y = merged_df['gdp']
    y_with_date = merged_df[['gdp', 'date_grouped']]

    features_alt = features
    features_alt.append('date_grouped')
    x_ols_alt = merged_df[features_alt]

    x_ols = add_constant(x_ols)    # to add constant value in the model, to tell us to fit for the b in 'y = mx + b'

    # OLS Regression using linearmodels - Has robust covariance
    ols_model = IV2SLS(dependent=y, exog=x_ols, endog=None, instruments=None).fit()
    # Leaving this out as we're going for a nowcast
    # Summary of the OLS regression - https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
    # save_model_as_image(model=ols_model, file_name='multivariate_ols_regression_incl_covid', lin_reg=True)

    '''
    Now checking residuals
    '''

    # QQ Plot - https://towardsdatascience.com/q-q-plots-explained-5aa8495426c0
    fig = plt.figure()
    ax = fig.add_subplot()
    scs.probplot(ols_model.resids, dist='norm', plot=plt)

    ax.get_lines()[0].set_markerfacecolor('black')
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[1].set_color('black')
    plt.title('QQ Plot - incl. covid')
    plt.show()

    plt.clf()

    # Plot outliers and check if any residuals are above 4 or < -4
    # Note that for extreme residuals, check the QQplot to see if it's one outlier exclusively
    merged_df['residuals'] = ols_model.resids
    max_residual = merged_df['residuals'].max() # 8.078761020132063
    min_residual = merged_df['residuals'].min() # -12.15981483406162

    # Check for serial correlation using a Durbin Watson test - https://www.statology.org/durbin-watson-test-python/
    dw_test = durbin_watson(ols_model.resids) # 1.164954221660367

    # Homocedasticity test
        # Ho = Homocedasticity = P > 0.05
        # Ha = There's no homocedasticity = p <=0.05

    model = sm.OLS(y, x_ols).fit()

    stat, p, f, fp = sms.het_breuschpagan(model.resid, model.model.exog)

    print(f'Test stat: {stat}') # 10.959761738829663
    print(f'p-Value: {p}') # 0.05218365231539978
    print(f'F-Value: {f}') # 2.2695840450750473
    print(f'f_p_value: {fp}') # 0.05063914958172162

    plt.scatter(y=model.resid, x=model.predict(), color='black', alpha=0.5, s=20)
    plt.hlines(y=0, xmin=0, xmax=4, color='orange')
    plt.xlim(80,100)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.show()

    # Generate Latex Table with all results
    rows_incl_covid = [['Model', 'durbin watson', 'min residual', 'max residual', 'het breuschpagan p value'],
                        ['Incl covid LR', dw_test, min_residual, max_residual, p]]

    rows_incl_covid.append(rows[1])

    save_table_as_latex(file_name='linear_reg_violation_incl_covid', rows=rows_incl_covid, header_count=5,
                        caption="Linear Regression violation checks (including covid)")

    '''
    Nowcasting model
    '''
    var_variables = features
    var_variables.append('gdp')
    var_variables.append('date_grouped')
    var_df = merged_df[var_variables]
    nowcast_regression_revamped(var_df, x_ols_alt, y, y_with_date, covid=True)

class GeneratingDataSourceDataframes():

    def generate_weather_df(self):
        ### creating our weather_analysis df

        weather_df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'weather_analysis', c.weather_file))

        weather_df = weather_df.drop(columns=[' win', ' spr', ' sum', ' aut', ' ann'], axis=1)

        # Transpose columns
        weather_df_transpose = weather_df.set_index('year').stack().reset_index()

        # rename df column
        weather_df_transpose = weather_df_transpose.rename(columns={"level_1": "month", 0: "average_temperature"})

        # generate date column from year and month column
        weather_df_transpose['date_grouped'] = weather_df_transpose['year'].astype(str) + '-' + weather_df_transpose['month'].astype(str)

        weather_df_transpose = weather_df_transpose.drop(columns=['year', 'month'])

        return weather_df_transpose


    def create_twitter_scrape_df(self):
        ### creating our twitter df
        column_name = 'tweet'
        twitter_scrape_df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'twitter_scraper', c.twitter_odeon_file))

        twitter_scrape_df = sentiment_analysis(twitter_scrape_df, column_name)
        # filter on twitter df for 1. Time period we want and 2. Columns we want
        date_column = 'date'
        sentiment = 'sentiment'
        twitter_sentiment_df = twitter_scrape_df[[date_column, sentiment]]
        twitter_sentiment_df['date'] = pd.to_datetime(twitter_sentiment_df['date'], format='%Y-%m-%d')

        # Set the cutoff date, based on when covid started in the UK
        cutoff_date = pd.to_datetime('2020-02-01')

        # Filter the DataFrame
        twitter_sentiment_df = twitter_sentiment_df[twitter_sentiment_df['date'] < cutoff_date]

        # Group by month?
        # grouped_df = twitter_sentiment_df.groupby(pd.Grouper(key='date', freq='M')).agg({'sentiment': pd.Series.mode})
        grouped_df = twitter_sentiment_df.groupby(pd.Grouper(key='date', freq='M')).agg({'sentiment': 'mean'}).reset_index()
        grouped_df['date_grouped'] = pd.to_datetime(grouped_df['date']).apply(
            lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

        return grouped_df[['date_grouped', 'sentiment']]

    def create_gdp_df(self):
        ### creating our gdp df

        gdp_df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'auxilliary_data', c.gdp_file))

        # filter on gdp df for 1. Time period we want and 2. Columns we want
        # filter on Title, gross value
        gdp_date_column = 'Month'
        monthly_gdp = 'Monthly GDP (A-T)'
        gdp_refine_df = gdp_df[[gdp_date_column, monthly_gdp]]

        # clean date column format to match DD-MM-YY format, similar to the box office df
        gdp_refine_df = gdp_refine_df.rename(columns={gdp_date_column: "date", monthly_gdp: "gdp"})

        gdp_refine_df['date'] = pd.to_datetime(gdp_refine_df['date'], format='%Y%b')
        gdp_refine_df['date_grouped'] = pd.to_datetime(gdp_refine_df['date']).apply(
            lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

        return gdp_refine_df.drop(columns=['date'])

    def create_box_office_weightings_df(self):

        ### creating box office df
        box_office_df = pd.read_csv(
            os.path.join(os.path.dirname(os.getcwd()), 'bfi_data_compile', 'output', c.box_office_file))

        no_of_cinemas = 'Number of cinemas'
        distributor = 'Distributor'
        title = 'Title'
        site_average = 'Site average'

        box_office_refine_df = box_office_df.drop(columns=[title, distributor, no_of_cinemas, site_average],
                                                  axis='columns')

        box_office_refine_df = box_office_refine_df.rename(columns={'Weekend Gross': 'weekend_gross'})

        box_office_refine_df['date'] = pd.to_datetime(box_office_refine_df['date'], format='%d-%m-%Y')

        box_office_refine_df['rank'] = box_office_refine_df.groupby('date')['weekend_gross'].rank(ascending=False)

        box_office_refine_df['total_weekend_gross'] = box_office_refine_df.groupby('date')['weekend_gross'].transform(
            np.sum)

        box_office_refine_df['weekend_gross_ratio'] = box_office_refine_df['weekend_gross'] / box_office_refine_df[
            'total_weekend_gross']

        # aggregate data by month (to match frequency of GDP data)
        box_office_refine_df['date_grouped'] = pd.to_datetime(box_office_refine_df['date']).apply(
            lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

        box_office_refine_df = box_office_refine_df.drop(
            columns=['date', 'total_weekend_gross', 'weekend_gross', 'Unnamed: 0', 'Unnamed: 0.1'])
        box_office_refine_df_by_month = box_office_refine_df.groupby(by=['rank', 'date_grouped'])[
            'weekend_gross_ratio'].mean().reset_index()

        def categorize_row(row, i):
            if row['rank'] == i:
                return row['weekend_gross_ratio']

        # flattening dataframe
        for i in range(1, 16):
            box_office_refine_df_by_month['monthly_gross_ratio_rank_%s' % i] = box_office_refine_df_by_month.apply(
                lambda row: categorize_row(row, i), axis=1)

        box_office_refine_df_by_month = box_office_refine_df_by_month.drop(columns=['rank', 'weekend_gross_ratio'])

        # read https://stackoverflow.com/questions/52899858/collapsing-rows-with-nan-entries-in-pandas-dataframe
        box_office_final_df = box_office_refine_df_by_month.groupby('date_grouped', as_index=False).first()

        # Add rows for missing months
        missing_box_office_df = pd.DataFrame({'date_grouped': ['2021-1', '2021-2', '2021-3', '2021-4', '2020-4', '2020-5', '2020-6'],
                                              'monthly_gross_ratio_rank_1': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_2': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_3': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_4': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_5': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_6': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_7': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_8': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_9': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_10': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_11': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_12': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_13': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_14': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067],
                                              'monthly_gross_ratio_rank_15': [0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
                                                                             0.067]})

        box_office_final_df = box_office_final_df.append(missing_box_office_df, ignore_index=True).sort_values(by=['date_grouped'])

        return box_office_final_df

    def create_box_office_df(self):
        box_office_file = 'compiled_top_15_box_office.csv'

        ### creating box office df
        box_office_df = pd.read_csv(
            os.path.join(os.path.dirname(os.getcwd()), 'bfi_data_compile', 'output', box_office_file))

        weekend_gross_column = 'Weekend Gross'
        box_office_date_column = 'date'
        no_of_cinemas = 'Number of cinemas'
        distributor = 'Distributor'

        box_office_refine_df = box_office_df[[weekend_gross_column, distributor, box_office_date_column, no_of_cinemas]]

        box_office_refine_df = box_office_refine_df.rename(
            columns={'Weekend Gross': 'weekend_gross', 'Number of cinemas': 'number_of_cinemas',
                     'Distributor': 'distributor'})

        box_office_refine_df['date'] = pd.to_datetime(box_office_refine_df['date'], format='%d-%m-%Y')

        # aggregate data by month (to match frequency of GDP data)
        box_office_refine_df['date_grouped'] = pd.to_datetime(box_office_refine_df['date']).apply(
            lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

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

        # Add rows for missing months
        missing_box_office_df = pd.DataFrame({'date_grouped': ['2021-1', '2021-2', '2021-3', '2021-4', '2020-4', '2020-5', '2020-6'],
                            'monthly_gross': [0, 0, 0, 0, 0, 0, 0],
                            'number_of_cinemas': [0, 0, 0, 0, 0, 0, 0]})

        box_office_grouped_df = box_office_grouped_df.append(missing_box_office_df, ignore_index=True).sort_values(by=['date_grouped'])

        return box_office_grouped_df

    def create_monthly_admission_df(self):
        '''
        CSV taken from https://www.cinemauk.org.uk/the-industry/facts-and-figures/uk-cinema-admissions-and-box-office/monthly-admissions/
        :return:
        '''
        monthly_admissions_file = 'monthly_admissions_uk.csv'

        # creating monthly admissions df
        monthly_admission_df = pd.read_csv(
            os.path.join(os.path.dirname(os.getcwd()), 'auxilliary_data', monthly_admissions_file), thousands=',')
        monthly_admission_df = monthly_admission_df.rename(columns={'monthly admissions': 'monthly_admissions'})

        return monthly_admission_df

    def create_google_trends_df(self):
        '''
        Load all google trend .csv's for key words - from 2004-2023
        :return: dataframe containing keywords
        '''

        # generating dataframes
        baftas_df = pd.read_csv(
            os.path.join(os.path.dirname(os.getcwd()), 'google_trends_scraper', c.baftas + c.csv_extension),
            skiprows=[0, 1]) \
            .rename(columns={'%s: (United Kingdom)' % c.baftas: 'frequency_%s' % c.baftas})
        academy_awards_df = pd.read_csv(
            os.path.join(os.path.dirname(os.getcwd()), 'google_trends_scraper', c.academy_awards + c.csv_extension),
            skiprows=[0, 1]) \
            .rename(columns={'Academy Awards: (United Kingdom)': 'frequency_%s' % c.academy_awards})
        cinema_showings_df = pd.read_csv(
            os.path.join(os.path.dirname(os.getcwd()), 'google_trends_scraper', c.cinema_showings + c.csv_extension),
            skiprows=[0, 1]) \
            .rename(
            columns={'%s: (United Kingdom)' % c.cinema_showings.replace('_', ' '): 'frequency_%s' % c.cinema_showings})
        cinemas_near_me_df = pd.read_csv(
            os.path.join(os.path.dirname(os.getcwd()), 'google_trends_scraper', c.cinemas_near_me + c.csv_extension),
            skiprows=[0, 1]) \
            .rename(
            columns={'%s: (United Kingdom)' % c.cinemas_near_me.replace('_', ' '): 'frequency_%s' % c.cinemas_near_me})
        films_df = pd.read_csv(
            os.path.join(os.path.dirname(os.getcwd()), 'google_trends_scraper', c.films + c.csv_extension), skiprows=[0, 1]) \
            .rename(columns={'%s: (United Kingdom)' % c.films: 'frequency_%s' % c.films})
        films_near_me_df = pd.read_csv(
            os.path.join(os.path.dirname(os.getcwd()), 'google_trends_scraper', c.films_near_me + c.csv_extension),
            skiprows=[0, 1]) \
            .rename(columns={'%s: (United Kingdom)' % c.films_near_me.replace('_', ' '): 'frequency_%s' % c.films_near_me})

        # merge dataframes
        google_trends_df = pd.merge(pd.merge(pd.merge(
            pd.merge(pd.merge(academy_awards_df, cinema_showings_df, on='Month'), cinemas_near_me_df, on='Month'),
            films_df, on='Month'), films_near_me_df, on='Month'), baftas_df, on='Month')
        google_trends_df = google_trends_df.rename(columns={'Month': 'date_grouped'})
        google_trends_df['date_grouped'] = pd.to_datetime(google_trends_df['date_grouped']).apply(
            lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
        return google_trends_df

def clean_up():
    folder_path = os.path.join(os.getcwd())
    extensions = ['.aux', '.txt', '.tex', '.log']

    for filename in os.listdir(folder_path):
        if any(filename.endswith(ext) for ext in extensions):
            os.remove(os.path.join(folder_path, filename))

if __name__ == '__main__':
    # Setting up config to avoid truncation of columns or column names:
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    # collecting independent variables
    df_generator = GeneratingDataSourceDataframes()

    weather_df = df_generator.generate_weather_df()

    twitter_scrape_df = df_generator.create_twitter_scrape_df()

    google_trends_df = df_generator.create_google_trends_df()  # google trends dataset

    gdp_df = df_generator.create_gdp_df()  # monthly gdp dataset

    box_office_df = df_generator.create_box_office_df()  # bfi box office dataset

    box_office_weightings_df = df_generator.create_box_office_weightings_df()  # bfi box office dataset, with weightings

    monthly_admission_df = df_generator.create_monthly_admission_df()  # monthly admissions dataset

    # creating regressions
    univariate_regression_box_office_gdp(gdp_df, box_office_df)

    univariate_regression_monthly_admission_gdp(gdp_df, monthly_admission_df)

    rows = multivariate_linear_regression_pre_covid(weather_df, gdp_df, box_office_df, monthly_admission_df, box_office_weightings_df, google_trends_df, twitter_scrape_df)

    multivariate_linear_regression_incl_covid(weather_df, gdp_df, box_office_df, box_office_weightings_df,
                                              google_trends_df, twitter_scrape_df, rows)

    clean_up()

