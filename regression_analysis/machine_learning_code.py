from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from common.latex_file_generator import save_table_as_latex
from common.visualisations import plot_importance_features, plot_nowcast, plot_var_nowcast
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

# Metrics
from regression_analysis.machine_learning.easymetrics import diebold_mariano_test

"""
VENV 3.7

pip install tabulate
pip install texttable
pip install latextable
"""


def nowcast_regression(var_df, X, Y, y_with_date, covid=False):
    """
    Function to nowcast, using machine learning techniques
    """
    y = Y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=100)
    var_train, var_test = train_test_split(var_df, test_size=0.3, shuffle=False)

    # Seeing the split across training and testing datasets
    print('Number of records in the original dataset: ', len(y))
    print('Number of records in the training dataset: ', len(y_train))
    print('Number of records in the testing dataset: ', len(y_test))

    lr_model = LinearRegression().fit(x_train, y_train)
    gbr_model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
    rfr_model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
    ridge_model_alpha_1 = Ridge(alpha=1).fit(x_train, y_train)
    lasso_model_alpha_1 = Lasso(alpha=1).fit(x_train, y_train)
    # higher the alpha value, more restriction on the coefficients; low alpha > more generalization,

    var_variable = plot_var_nowcast(var_df=var_df, var_train=var_train, var_test=var_test)
    var_df = var_df.set_index('date_grouped')

    #forecaster_model = ForecasterAutoreg(regressor=regressor, lags=20)
    #TODO - ForecasterAutoreg - recursive forecasting - https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html


    train_score_ols = lr_model.score(x_train, y_train)
    test_score_ols = lr_model.score(x_test, y_test)

    train_score_gbr = gbr_model.score(x_train, y_train)
    test_score_gbr = gbr_model.score(x_test, y_test)

    train_score_rfr = rfr_model.score(x_train, y_train)
    test_score_rfr = rfr_model.score(x_test, y_test)

    train_score_ridge = ridge_model_alpha_1.score(x_train, y_train)
    test_score_ridge = ridge_model_alpha_1.score(x_test, y_test)

    train_score_lasso = lasso_model_alpha_1.score(x_train, y_train)
    test_score_lasso = lasso_model_alpha_1.score(x_test, y_test)

    covid_features = ['constant', 'monthly gross', 'frequency_cinemas_near_me',
                                             'frequency_baftas',
                                             'average_temperature', 'weighted_ranking',
                                             'gdp_lag1', 'cinema_lockdown']
    non_covid_features = ['constant', 'monthly gross', 'frequency_cinemas_near_me',
                          'frequency_baftas',
                          'average_temperature', 'sentiment',
                          'weighted_ranking', 'gdp_lag1']

    # Lin Reg - get importance of each feature
    plot_importance_features(model=lr_model, color='maroon', covid_features=covid_features,
                             non_covid_features=non_covid_features, model_name='Lin Reg nowcast', coef=True,
                             covid=covid)

    # GBR - get importance of each feature
    plot_importance_features(model=gbr_model, color='limegreen', covid_features=covid_features,
                             non_covid_features=non_covid_features, model_name='GBR nowcast', coef=False,
                             covid=covid)

    # RFR - get importance of each feature
    plot_importance_features(model=rfr_model, color='gold', covid_features=covid_features,
                             non_covid_features=non_covid_features, model_name='RFR nowcast', coef=False,
                             covid=covid)

    # Ridge - get importance of each feature
    plot_importance_features(model=ridge_model_alpha_1, color='blue',covid_features=covid_features, coef=True,
                             non_covid_features=non_covid_features, model_name='Ridge nowcast',
                             covid=covid)

    # Lasso - get importance of each feature
    plot_importance_features(model=lasso_model_alpha_1, color='black', covid_features=covid_features, coef=True,
                             non_covid_features=non_covid_features, model_name='Lasso nowcast',
                             covid=covid)

    plt.clf()

    # Prepare nowcast graph:
    # https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
    x_test_full = x_test.reset_index().merge(y_with_date.reset_index(), how='inner', on='index')
    y_test_full = y_test.to_frame().reset_index().merge(y_with_date.reset_index(), how='inner', on='index')
    y_test_full = y_test_full.sort_values('date_grouped')
    x_test_full = x_test_full.sort_values('date_grouped')
    covid_nowcast = ['const', 'monthly_gross', 'frequency_cinemas_near_me',
                     'frequency_baftas',
                     'average_temperature', 'weighted_ranking',
                     'gdp_lag1', 'cinema_lockdown']
    non_covid_nowcast = ['const', 'monthly_gross', 'frequency_cinemas_near_me',
                         'frequency_baftas',
                         'average_temperature', 'sentiment',
                         'weighted_ranking', 'gdp_lag1']
    # LR nowcast
    y_pred_lr = plot_nowcast(model=lr_model, x_test_full=x_test_full, y_test_full=y_test_full,
                             covid_features=covid_nowcast,
                             non_covid_features=non_covid_nowcast, color='maroon', model_label='linear regression',
                             model_name='lr',
                             covid=covid)

    # GBR nowcast
    y_pred_gbr = plot_nowcast(model=gbr_model, x_test_full=x_test_full, y_test_full=y_test_full,
                              covid_features=covid_nowcast,
                              non_covid_features=non_covid_nowcast, color='limegreen',
                              model_label='gradient boosting regression', model_name='gbr',
                              covid=covid)

    # RFR nowcast
    y_pred_rfr = plot_nowcast(model=rfr_model, x_test_full=x_test_full, y_test_full=y_test_full,
                              covid_features=covid_nowcast,
                              non_covid_features=non_covid_nowcast, color='gold',
                              model_label='random forest regression', model_name='rfr',
                              covid=covid)

    # Ridge nowcast
    y_pred_ridge = plot_nowcast(model=ridge_model_alpha_1, x_test_full=x_test_full, y_test_full=y_test_full,
                              covid_features=covid_nowcast,
                              non_covid_features=non_covid_nowcast, color='blue',
                              model_label='ridge regression', model_name='ridge',
                              covid=covid)

    # Lasso nowcast
    y_pred_lasso = plot_nowcast(model=lasso_model_alpha_1, x_test_full=x_test_full, y_test_full=y_test_full,
                              covid_features=covid_nowcast,
                              non_covid_features=non_covid_nowcast, color='black',
                              model_label='lasso regression', model_name='lasso',
                              covid=covid)


    # Calculating RMSE (Root mean squared error) for each model
    # https://stackoverflow.com/questions/69844967/calculation-of-mse-and-rmse-in-linear-regression
    rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))

    rmse_gbr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_gbr))

    rmse_rfr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfr))

    rmse_lasso = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso))

    rmse_ridge = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge))

    # TODO - figure out calculating DM-test (Diebold-Mariano Test) to compare models
    # https://www.kaggle.com/code/jorgesandoval/xgboost-vs-lightgbm-using-diebold-mariano-test/notebook
    # DM-test - https://academic.oup.com/ej/pages/top_cited_papers
    # - https://medium.com/@philippetousignant/comparing-forecast-accuracy-in-python-diebold-mariano-test-ad109026f6ab#:~:text=In%20conclusion%2C%20the%20Diebold%2DMariano,when%20choosing%20a%20forecasting%20method.
    dm_test = diebold_mariano_test(y_test, y_pred_lr, y_pred_gbr, h=1, crit="MSE")


    # TODO - VAR test
    # https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
    # https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/

    # ADF Test on each column
    # TODO - Get ADF test working, which checks for stationarity (which is what we want)
    # By the way, if a series is found to be non-stationary, you make it stationary by differencing the series once and repeat the test again until it becomes stationary.
    for name, column in df_train.iteritems():
        adfuller_test(column, name=column.name)
        print('\n')

    # Generate Latex Table with all results
    rows = [['Model', 'train score', 'test score', 'RMSE'],
            ['LR', train_score_ols, test_score_ols, rmse_lr],
            ['GBR', train_score_gbr, test_score_gbr, rmse_gbr],
            ['RFR', train_score_rfr, test_score_rfr, rmse_rfr],
            ['Ridge', train_score_ridge, test_score_ridge, rmse_ridge],
            ['Lasso', train_score_lasso, test_score_lasso, rmse_lasso]
            # ['VAR', 'ESA', '21', '2002']]
            ]

    if covid:
        save_table_as_latex(caption="A comparison of nowcasting models (including covid)", file_name='model_comparison_incl_covid', rows=rows, header_count=4)
    else:
        save_table_as_latex(caption="A comparison of nowcasting models (pre-covid)", file_name='model_comparison_pre_covid', rows=rows, header_count=4)

