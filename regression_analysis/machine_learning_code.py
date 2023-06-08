from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from tabulate import tabulate
from texttable import Texttable
import latextable

# Metrics
from regression_analysis.machine_learning.easymetrics import diebold_mariano_test

"""
VENV 3.7

pip install tabulate
pip install texttable
pip install latextable
"""


def nowcast_regression(X, Y, y_with_date, features):
    """
    Function to nowcast, using machine learning techniques
    """
    y = Y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=100)

    # Seeing the split across training and testing datasets
    print('Number of records in the original dataset: ', len(y))
    print('Number of records in the training dataset: ', len(y_train))
    print('Number of records in the testing dataset: ', len(y_test))

    model_train = LinearRegression().fit(x_train, y_train)
    gbr_model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
    rfr_model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
    #forecaster_model = ForecasterAutoreg(regressor=regressor, lags=20)
    #TODO - ForecasterAutoreg - recursive forecasting - https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html


    train_score_ols = model_train.score(x_train, y_train) # 0.9958670475825846
    test_score_ols = model_train.score(x_test, y_test) # 0.9947080976845063

    train_score_gbr = gbr_model.score(x_train, y_train) #
    test_score_gbr = gbr_model.score(x_test, y_test) #

    train_score_rfr = rfr_model.score(x_train, y_train) #
    test_score_rfr = rfr_model.score(x_test, y_test) #

    # Lin Reg - get importance of each feature
    fig, ax = plt.subplots()
    importance = model_train.coef_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    feature_list = ['constant', 'monthly gross', 'frequency_cinemas_near_me', 'frequency_baftas', 'average_temperature', 'sentiment',
                    'weighted_ranking', 'gdp_lag1']
    bars = ax.barh(feature_list, importance, color='maroon')
    ax.bar_label(bars)
    plt.title('Feature importance - GBR nowcast (pre-covid)')
    plt.show()

    # GBR - get importance of each feature
    fig, ax = plt.subplots()
    importance = gbr_model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    feature_list = ['constant', 'monthly gross', 'frequency_cinemas_near_me', 'frequency_baftas', 'average_temperature', 'sentiment',
                    'weighted_ranking', 'gdp_lag1']
    bars = ax.barh(feature_list, importance, color='limegreen')
    ax.bar_label(bars)
    plt.title('Feature importance - GBR nowcast (pre-covid)')
    plt.show()
    plt.clf()

    # RFR - get importance of each feature
    fig, ax = plt.subplots()
    importance = rfr_model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    feature_list = ['constant', 'monthly gross', 'frequency_cinemas_near_me', 'frequency_baftas', 'average_temperature', 'sentiment',
                    'weighted_ranking', 'gdp_lag1']
    bars = ax.barh(feature_list, importance, color='yellow')
    ax.bar_label(bars)
    plt.title('Feature importance - RFR nowcast (pre-covid)')
    plt.show()
    plt.clf()

    # https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
    x_test_full = x_test.reset_index().merge(y_with_date.reset_index(), how='inner', on='index')
    x_test_full['y_pred_lr'] = model_train.predict(x_test_full.drop(columns=['date_grouped', 'gdp', 'index']))
    y_pred = x_test_full['y_pred_lr'].to_numpy()
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

    # Prepare nowcast graph:
    y_test_full = y_test.to_frame().reset_index().merge(y_with_date.reset_index(), how='inner', on='index')
    # Plotting nowcast gdp vs actual gdp
    plt.figure()
    fig, ax = plt.subplots()
    y_test_full = y_test_full.sort_values('date_grouped')
    x_test_full = x_test_full.sort_values('date_grouped')
    plt.plot(y_test_full['date_grouped'], y_test_full['gdp_x'], '-o', label="actual gdp", markersize=3)
    plt.plot(x_test_full['date_grouped'], x_test_full['y_pred_lr'], '-o', label="linear regression", markersize=3, color='maroon')
    plt.grid()
    leg = plt.legend(loc='upper center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title('Nowcast test set - Linear Regression (pre-covid)')

    plt.clf()


    fig, ax = plt.subplots()
    x_test_full['y_pred_gbr'] = gbr_model.predict(x_test_full.drop(columns=['date_grouped', 'gdp', 'index', 'y_pred_lr']))
    y_pred_gbr = x_test_full['y_pred_gbr'].to_numpy()
    plt.plot(y_test_full['date_grouped'], y_test_full['gdp_x'], '-o', label="actual gdp", markersize=3)
    plt.plot(x_test_full['date_grouped'], x_test_full['y_pred_gbr'], '-o', label="gradient boosting regression", markersize=3, color='limegreen')
    leg = plt.legend(loc='upper center')
    plt.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title('Nowcast test set - GBR (pre-covid)')
    plt.show()

    plt.clf()


    fig, ax = plt.subplots()
    x_test_full['y_pred_rfr'] = gbr_model.predict(
        x_test_full.drop(columns=['date_grouped', 'gdp', 'index', 'y_pred_lr',  'y_pred_gbr']))
    y_pred_rfr = x_test_full['y_pred_rfr'].to_numpy()

    plt.plot(y_test_full['date_grouped'], y_test_full['gdp_x'], '-o', label="actual gdp", markersize=3)
    plt.plot(x_test_full['date_grouped'], x_test_full['y_pred_rfr'], '-o', label="gradient boosting regression",
             markersize=3, color='yellow')
    leg = plt.legend(loc='upper center')
    plt.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title('Nowcast test set - RFR (pre-covid)')
    plt.show()

    plt.clf()

    # Calculating RMSE (Root mean squared error) for each model
    # https://stackoverflow.com/questions/69844967/calculation-of-mse-and-rmse-in-linear-regression
    rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, y_pred)) #

    rmse_gbr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_gbr)) #

    rmse_rfr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfr)) #

    # Calculating DM-test (Diebold-Mariano Test) to compare models
    # https://www.kaggle.com/code/jorgesandoval/xgboost-vs-lightgbm-using-diebold-mariano-test/notebook
    # DM-test - https://academic.oup.com/ej/pages/top_cited_papers
    # - https://medium.com/@philippetousignant/comparing-forecast-accuracy-in-python-diebold-mariano-test-ad109026f6ab#:~:text=In%20conclusion%2C%20the%20Diebold%2DMariano,when%20choosing%20a%20forecasting%20method.
    dm_test = diebold_mariano_test(y_test, y_pred, y_pred_gbr, h=1, crit="MSE")


    # TODO - VAR test
    # https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
    # https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/

    # Generate Latex Table with all results
    rows = [['Model', 'train score', 'test score', 'RMSE'],
            ['LR', train_score_ols, test_score_ols, rmse_lr],
            ['GBR', train_score_gbr, test_score_gbr, rmse_gbr],
            ['RFR', train_score_rfr, test_score_rfr, rmse_rfr]]
            #['VAR', 'ESA', '21', '2002']]

    table = Texttable()
    table.set_cols_align(["c"] * 4)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    print('Tabulate Table:')
    print(tabulate(rows, headers='firstrow'))

    print('\nTexttable Table:')
    print(table.draw())

    print('\nTabulate Latex:')
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))

    print('\nTexttable Latex:')
    print(latextable.draw_latex(table, caption="A comparison of rocket features."))



