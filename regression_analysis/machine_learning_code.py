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
from statsmodels.tsa.api import VAR

# Metrics
from regression_analysis.machine_learning.easymetrics import diebold_mariano_test

"""
VENV 3.7

pip install tabulate
pip install texttable
pip install latextable
"""


# def nowcast_regression(var_df, X, Y, y_with_date, covid=False):
#     """
#     Function to nowcast, using machine learning techniques
#     """
#     y = Y.to_numpy()
#
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=100)
#     #var_train, var_test = train_test_split(var_df, test_size=0.3, shuffle=False)
#
#     # Seeing the split across training and testing datasets
#     print('Number of records in the original dataset: ', len(y))
#     print('Number of records in the training dataset: ', len(y_train))
#     print('Number of records in the testing dataset: ', len(y_test))
#
#     lr_model = LinearRegression().fit(x_train, y_train)
#     gbr_model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
#     rfr_model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
#     ridge_model_alpha_1 = Ridge(alpha=1).fit(x_train, y_train)
#     lasso_model_alpha_1 = Lasso(alpha=1).fit(x_train, y_train)
#     # higher the alpha value, more restriction on the coefficients; low alpha > more generalization,
#
#     # var_variable = plot_var_nowcast(var_df=var_df, var_train=var_train, var_test=var_test)
#     # var_df = var_df.set_index('date_grouped')
#
#     #forecaster_model = ForecasterAutoreg(regressor=regressor, lags=20)
#     #TODO - ForecasterAutoreg - recursive forecasting - https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html
#
#
#     train_score_ols = lr_model.score(x_train, y_train)
#     test_score_ols = lr_model.score(x_test, y_test)
#
#     train_score_gbr = gbr_model.score(x_train, y_train)
#     test_score_gbr = gbr_model.score(x_test, y_test)
#
#     train_score_rfr = rfr_model.score(x_train, y_train)
#     test_score_rfr = rfr_model.score(x_test, y_test)
#
#     train_score_ridge = ridge_model_alpha_1.score(x_train, y_train)
#     test_score_ridge = ridge_model_alpha_1.score(x_test, y_test)
#
#     train_score_lasso = lasso_model_alpha_1.score(x_train, y_train)
#     test_score_lasso = lasso_model_alpha_1.score(x_test, y_test)
#
#     covid_features = ['constant', 'monthly gross', 'frequency_cinemas_near_me',
#                                              'frequency_baftas',
#                                              'average_temperature', 'weighted_ranking',
#                                              'gdp_lag1', 'cinema_lockdown']
#     non_covid_features = ['constant', 'monthly gross', 'frequency_cinemas_near_me',
#                           'frequency_baftas',
#                           'average_temperature', 'sentiment',
#                           'weighted_ranking', 'gdp_lag1']
#
#     # Lin Reg - get importance of each feature
#     plot_importance_features(model=lr_model, color='maroon', covid_features=covid_features,
#                              non_covid_features=non_covid_features, model_name='Lin Reg nowcast', coef=True,
#                              covid=covid)
#
#     # GBR - get importance of each feature
#     plot_importance_features(model=gbr_model, color='limegreen', covid_features=covid_features,
#                              non_covid_features=non_covid_features, model_name='GBR nowcast', coef=False,
#                              covid=covid)
#
#     # RFR - get importance of each feature
#     plot_importance_features(model=rfr_model, color='gold', covid_features=covid_features,
#                              non_covid_features=non_covid_features, model_name='RFR nowcast', coef=False,
#                              covid=covid)
#
#     # Ridge - get importance of each feature
#     plot_importance_features(model=ridge_model_alpha_1, color='blue',covid_features=covid_features, coef=True,
#                              non_covid_features=non_covid_features, model_name='Ridge nowcast',
#                              covid=covid)
#
#     # Lasso - get importance of each feature
#     plot_importance_features(model=lasso_model_alpha_1, color='black', covid_features=covid_features, coef=True,
#                              non_covid_features=non_covid_features, model_name='Lasso nowcast',
#                              covid=covid)
#
#     plt.clf()
#
#     # Prepare nowcast graph:
#     # https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
#     x_test_full = x_test.reset_index().merge(y_with_date.reset_index(), how='inner', on='index')
#     y_test_full = y_test.to_frame().reset_index().merge(y_with_date.reset_index(), how='inner', on='index')
#     y_test_full = y_test_full.sort_values('date_grouped')
#     x_test_full = x_test_full.sort_values('date_grouped')
#     covid_nowcast = ['const', 'monthly_gross', 'frequency_cinemas_near_me',
#                      'frequency_baftas',
#                      'average_temperature', 'weighted_ranking',
#                      'gdp_lag1', 'cinema_lockdown']
#     non_covid_nowcast = ['const', 'monthly_gross', 'frequency_cinemas_near_me',
#                          'frequency_baftas',
#                          'average_temperature', 'sentiment',
#                          'weighted_ranking', 'gdp_lag1']
#     # LR nowcast
#     y_pred_lr = plot_nowcast(model=lr_model, x_test_full=x_test_full, y_test_full=y_test_full,
#                              covid_features=covid_nowcast,
#                              non_covid_features=non_covid_nowcast, color='maroon', model_label='linear regression',
#                              model_name='lr',
#                              covid=covid)
#
#     # GBR nowcast
#     y_pred_gbr = plot_nowcast(model=gbr_model, x_test_full=x_test_full, y_test_full=y_test_full,
#                               covid_features=covid_nowcast,
#                               non_covid_features=non_covid_nowcast, color='limegreen',
#                               model_label='gradient boosting regression', model_name='gbr',
#                               covid=covid)
#
#     # RFR nowcast
#     y_pred_rfr = plot_nowcast(model=rfr_model, x_test_full=x_test_full, y_test_full=y_test_full,
#                               covid_features=covid_nowcast,
#                               non_covid_features=non_covid_nowcast, color='gold',
#                               model_label='random forest regression', model_name='rfr',
#                               covid=covid)
#
#     # Ridge nowcast
#     y_pred_ridge = plot_nowcast(model=ridge_model_alpha_1, x_test_full=x_test_full, y_test_full=y_test_full,
#                               covid_features=covid_nowcast,
#                               non_covid_features=non_covid_nowcast, color='blue',
#                               model_label='ridge regression', model_name='ridge',
#                               covid=covid)
#
#     # Lasso nowcast
#     y_pred_lasso = plot_nowcast(model=lasso_model_alpha_1, x_test_full=x_test_full, y_test_full=y_test_full,
#                               covid_features=covid_nowcast,
#                               non_covid_features=non_covid_nowcast, color='black',
#                               model_label='lasso regression', model_name='lasso',
#                               covid=covid)
#
#
#     # Calculating RMSE (Root mean squared error) for each model
#     # https://stackoverflow.com/questions/69844967/calculation-of-mse-and-rmse-in-linear-regression
#     rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))
#
#     rmse_gbr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_gbr))
#
#     rmse_rfr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfr))
#
#     rmse_lasso = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso))
#
#     rmse_ridge = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge))
#
#     # TODO - figure out calculating DM-test (Diebold-Mariano Test) to compare models
#     # https://www.kaggle.com/code/jorgesandoval/xgboost-vs-lightgbm-using-diebold-mariano-test/notebook
#     # DM-test - https://academic.oup.com/ej/pages/top_cited_papers
#     # - https://medium.com/@philippetousignant/comparing-forecast-accuracy-in-python-diebold-mariano-test-ad109026f6ab#:~:text=In%20conclusion%2C%20the%20Diebold%2DMariano,when%20choosing%20a%20forecasting%20method.
#     dm_test = diebold_mariano_test(y_test, y_pred_lr, y_pred_gbr, h=1, crit="MSE")
#
#     # Generate Latex Table with all results
#     rows = [['Model', 'train score', 'test score', 'RMSE'],
#             ['LR', train_score_ols, test_score_ols, rmse_lr],
#             ['GBR', train_score_gbr, test_score_gbr, rmse_gbr],
#             ['RFR', train_score_rfr, test_score_rfr, rmse_rfr],
#             ['Ridge', train_score_ridge, test_score_ridge, rmse_ridge],
#             ['Lasso', train_score_lasso, test_score_lasso, rmse_lasso]
#             # ['VAR', 'ESA', '21', '2002']]
#             ]
#
#     if covid:
#         save_table_as_latex(caption="A comparison of nowcasting models (including covid)", file_name='model_comparison_incl_covid', rows=rows, header_count=4)
#     else:
#         save_table_as_latex(caption="A comparison of nowcasting models (pre-covid)", file_name='model_comparison_pre_covid', rows=rows, header_count=4)


def nowcast_regression_revamped(var_df, x, y, y_with_date, covid=False):
    """
    Function to nowcast, using machine learning techniques
    """

    var_df = var_df.loc[:, ~var_df.columns.duplicated()].copy()

    # TODO - see if I need to add 'const' back
    covid_nowcast_features = ['monthly_gross', 'frequency_cinemas_near_me',
                     'frequency_baftas',
                     'average_temperature', 'weighted_ranking',
                     'gdp_lag1', 'cinema_lockdown']
    non_covid_nowcast_features = ['monthly_gross', 'frequency_cinemas_near_me',
                         'frequency_baftas',
                         'average_temperature', 'sentiment',
                         'weighted_ranking', 'gdp_lag1']
    # Create list for each model, to eventually contain predictions and be passed into a df
    pred_lr_list = []
    pred_gbr_list = []
    pred_rfr_list = []
    pred_lasso_1_list = []
    pred_ridge_1_list = []
    pred_lasso_01_list = []
    pred_ridge_01_list = []
    pred_var_list = []
    # TODO - add pred_var_list to this list when I figure out VAR
    list_of_predictors = [pred_lr_list, pred_gbr_list, pred_rfr_list, pred_lasso_1_list, pred_ridge_1_list,
                          pred_lasso_01_list, pred_ridge_01_list, pred_var_list]

    # Loop to iteratively generate a prediction df
    # Loop through each month - going over the last 5 years/60 months
    x_row_count = len(x.index)
    for i in range(x_row_count-60, x_row_count):
        # TODO - if approaching 2020, add an if statement to catch it and remove 'sentiment' from the features
        if x.iloc[i]['date_grouped'] > pd.to_datetime('2020-02-01'):
            # TODO - will also need to add lockdown dummy to x?
            x = x.drop['sentiment']
            features = covid_nowcast_features
        else:
            features = non_covid_nowcast_features
        # Training using the 48 months (4 years) before then)
        # Dropping date grouped column to avoid error in .fit() - TypeError: The DType <class 'numpy.dtype[datetime64]'> could not be promoted by <class 'numpy.dtype[float64]'>
        x_train = x.iloc[i-48:i].drop(columns=['date_grouped'])
        y_train = y.iloc[i-48:i].drop(columns=['date_grouped'])
        x_test = x.iloc[[i]]
        y_test = y.iloc[i]

        lr_model = LinearRegression().fit(x_train, y_train)
        gbr_model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
        rfr_model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
        ridge_model_alpha_1 = Ridge(alpha=1).fit(x_train, y_train)
        lasso_model_alpha_1 = Lasso(alpha=1).fit(x_train, y_train)
        # TODO - what does this mean: Ill-conditioned matrix (rcond=1.38778e-17): result may not be accurate.
        ridge_model_alpha_01 = Ridge(alpha=0.1).fit(x_train, y_train)
        lasso_model_alpha_01 = Lasso(alpha=0.1).fit(x_train, y_train)
        # higher the alpha value, more restriction on the coefficients; low alpha > more generalization

        # https://towardsdatascience.com/vector-autoregressive-for-forecasting-time-series-a60e6f168c70
        var_df_pred = var_df
        var_train, var_test = var_df_pred.iloc[(i-48):i], var_df_pred.iloc[[i]]
        # if 'date_grouped' in var_train: # TODO - clean up this if-else code
        #     var_train_diff = var_train.drop(columns=['date_grouped'])
        # else:
        var_train_diff = var_train
        var_train_diff.diff().dropna(inplace=True)
        if 'date_grouped' in var_train_diff:
            var_train_diff.set_index('date_grouped', inplace=True)
        var_model = VAR(var_train_diff)
        res = var_model.select_order(maxlags=1)
        #print(res.summary())
        var_model = VAR(endog=var_train_diff)
        var_model = var_model.fit(1)

        list_of_models = [lr_model, gbr_model, rfr_model, lasso_model_alpha_1, ridge_model_alpha_1, lasso_model_alpha_01, ridge_model_alpha_01, var_model]

        # Loop through each model
        for a, b in zip(list_of_predictors, list_of_models):
            if b == var_model:
                lag_order = b.k_ar
                input_data = var_train_diff.values[-lag_order:]
                pred = b.forecast(y=input_data, steps=1)
                # var doesn't like the date column, so the next few lines are not clean code
                if 'date_grouped' in var_df_pred:
                    var_df_pred.set_index('date_grouped', inplace=True)
                pred = (pd.DataFrame(pred, index=var_test.index, columns=var_df_pred.columns + '_pred'))

                if 'date_grouped' not in var_test:
                    var_test = var_test.reset_index()
                pred = pred.reset_index()
                if 'date_grouped' not in pred:
                    pred = pred.set_index(keys='index')
                    pred['date_grouped'] = var_test[['date_grouped']]

                predicted_value = pred[['date_grouped', 'gdp_pred']].values.tolist()[0]
                a.append(predicted_value)

            else:
                # Append prediction to each relevant dataframe
                x_test['y_pred'] = b.predict(x_test[features])
                # Append this predicted value - date pair into a list
                predicted_value = x_test[['date_grouped', 'y_pred']].values.tolist()[0]
                a.append(predicted_value)

    # Pass newly created lists into empty dfs:
    pred_lr_df = pd.DataFrame(data=pred_lr_list, columns=('date_grouped', 'pred_gdp_lr'))
    pred_gbr_df = pd.DataFrame(data=pred_gbr_list, columns=('date_grouped', 'pred_gdp_gbr'))
    pred_rfr_df = pd.DataFrame(data=pred_rfr_list, columns=('date_grouped', 'pred_gdp_rfr'))
    pred_lasso_1_df = pd.DataFrame(data=pred_lasso_1_list, columns=('date_grouped', 'pred_gdp_lasso_1'))
    pred_lasso_01_df = pd.DataFrame(data=pred_lasso_01_list, columns=('date_grouped', 'pred_gdp_lasso_01'))
    pred_ridge_1_df = pd.DataFrame(data=pred_ridge_1_list, columns=('date_grouped', 'pred_gdp_ridge_1'))
    pred_ridge_01_df = pd.DataFrame(data=pred_ridge_01_list, columns=('date_grouped', 'pred_gdp_ridge_01'))
    pred_var_df = pd.DataFrame(data=pred_var_list, columns=('date_grouped', 'pred_gdp_var'))

    # Group all dfs to pass into nowcast plot
    complete_df = pd.merge(pred_lr_df, pred_gbr_df, on=['date_grouped'])
    complete_df = pd.merge(complete_df, pred_rfr_df, on=['date_grouped'])
    complete_df = pd.merge(complete_df, pred_lasso_1_df, on=['date_grouped'])
    complete_df = pd.merge(complete_df, pred_lasso_01_df, on=['date_grouped'])
    complete_df = pd.merge(complete_df, pred_ridge_1_df, on=['date_grouped'])
    complete_df = pd.merge(complete_df, pred_ridge_01_df, on=['date_grouped'])
    complete_df = pd.merge(complete_df, pred_var_df, on=['date_grouped'])


    # Merge with y_with_date
    complete_df = pd.merge(complete_df, y_with_date, on=['date_grouped'])

    # Derived column of %differential between gdp and predicted gdp. Do so for all models
    complete_df['lr_pct_difference'] = (complete_df['pred_gdp_lr'] - complete_df['gdp']) / complete_df['pred_gdp_lr']
    complete_df['gbr_pct_difference'] = (complete_df['pred_gdp_gbr'] - complete_df['gdp']) / complete_df['pred_gdp_gbr']
    complete_df['rfr_pct_difference'] = (complete_df['pred_gdp_rfr'] - complete_df['gdp']) / complete_df['pred_gdp_rfr']
    complete_df['lasso_1_pct_difference'] = (complete_df['pred_gdp_lasso_1'] - complete_df['gdp']) / complete_df['pred_gdp_lasso_1']
    complete_df['lasso_01_pct_difference'] = (complete_df['pred_gdp_lasso_01'] - complete_df['gdp']) / complete_df['pred_gdp_lasso_01']
    complete_df['ridge_1_pct_difference'] = (complete_df['pred_gdp_ridge_1'] - complete_df['gdp']) / complete_df['pred_gdp_ridge_1']
    complete_df['ridge_01_pct_difference'] = (complete_df['pred_gdp_ridge_01'] - complete_df['gdp']) / complete_df['pred_gdp_ridge_01']
    complete_df['var_pct_difference'] = (complete_df['pred_gdp_var'] - complete_df['gdp']) / complete_df['pred_gdp_var']


    # TODO - Plot nowcast graph at the end of the loop
    fig, ax = plt.subplots()

    plt.plot(complete_df['date_grouped'], complete_df['lr_pct_difference'], '-o', label="lr_pct_difference gdp", markersize=3)
    plt.plot(complete_df['date_grouped'], complete_df['gbr_pct_difference'], '-o', label="gbr_pct_difference gdp", markersize=3)
    plt.plot(complete_df['date_grouped'], complete_df['rfr_pct_difference'], '-o', label="rfr_pct_difference gdp", markersize=3)
    plt.plot(complete_df['date_grouped'], complete_df['lasso_1_pct_difference'], '-o', label="lasso_1_pct_difference gdp", markersize=3)
    plt.plot(complete_df['date_grouped'], complete_df['lasso_01_pct_difference'], '-o', label="lasso_01_pct_difference gdp", markersize=3)
    plt.plot(complete_df['date_grouped'], complete_df['ridge_1_pct_difference'], '-o', label="ridge_1_pct_difference gdp", markersize=3)
    plt.plot(complete_df['date_grouped'], complete_df['ridge_01_pct_difference'], '-o', label="ridge_01_pct_difference gdp", markersize=3)
    plt.plot(complete_df['date_grouped'], complete_df['var_pct_difference'], '-o', label="var_pct_difference gdp", markersize=3)

    leg = plt.legend(loc='upper center')
    plt.grid()
    plt.title('GDP Nowcast model comparison: 2015 - 2020')
    # TODO - work on naming conventions (make it covid agnostic)
    # if covid:
    #     plt.title('Nowcast test set - %s (including covid)' % model_name)
    # else:
    #     plt.title('Nowcast test set - %s (pre-covid)' % model_name)
    plt.show()



    # Calculating RMSE (Root mean squared error) for each model
    # https://stackoverflow.com/questions/69844967/calculation-of-mse-and-rmse-in-linear-regression
    # Get y_pred_lr LIST from pred_lr_df DATAFRAME
    y_pred_lr = pred_lr_df['pred_gdp_lr'].values.tolist()
    y_pred_gbr = complete_df['pred_gdp_gbr'].values.tolist()
    y_pred_rfr = complete_df['pred_gdp_rfr'].values.tolist()
    y_pred_lasso_1 = complete_df['pred_gdp_lasso_1'].values.tolist()
    y_pred_lasso_01 = complete_df['pred_gdp_lasso_01'].values.tolist()
    y_pred_ridge_1 = complete_df['pred_gdp_ridge_1'].values.tolist()
    y_pred_ridge_01 = complete_df['pred_gdp_ridge_1'].values.tolist()
    y_pred_var = complete_df['pred_gdp_var'].values.tolist()

    # Create a custom y_test to cover the date range y_pred_lr covered
    y_test = y.iloc[x_row_count-60: x_row_count].values.tolist()

    # Calculate RMSE for all models
    rmse_lr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))

    rmse_gbr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_gbr))

    rmse_rfr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfr))

    rmse_lasso_1 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso_1))
    rmse_lasso_01 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso_01))

    rmse_ridge_1 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge_1))
    rmse_ridge_01 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge_01))

    # TODO - figure out calculating DM-test (Diebold-Mariano Test) to compare models
    # https://www.kaggle.com/code/jorgesandoval/xgboost-vs-lightgbm-using-diebold-mariano-test/notebook
    # DM-test - https://academic.oup.com/ej/pages/top_cited_papers
    # - https://medium.com/@philippetousignant/comparing-forecast-accuracy-in-python-diebold-mariano-test-ad109026f6ab#:~:text=In%20conclusion%2C%20the%20Diebold%2DMariano,when%20choosing%20a%20forecasting%20method.
    dm_test = diebold_mariano_test(y_test, y_pred_lr, y_pred_gbr, h=1, crit="MSE")

    # Generate Latex Table with all results
    rows = [['Model', 'train score', 'test score', 'RMSE'],
            ['LR', rmse_lr],
            ['GBR', rmse_gbr],
            ['RFR', rmse_rfr],
            ['Ridge Alpha 1', rmse_ridge_1],
            ['Lasso Alpha 1', rmse_lasso_1],
            ['Ridge Alpha 0.1', rmse_ridge_01],
            ['Lasso Alpha 0.1', rmse_lasso_01]
            # ['VAR', 'ESA', '21', '2002']]
            ]

    # TODO - add some feature importance capture later
    if covid:
        save_table_as_latex(caption="A comparison of nowcasting models (including covid)", file_name='nowcast_model_comparison_incl_covid', rows=rows, header_count=4)
    else:
        save_table_as_latex(caption="A comparison of nowcasting models (pre-covid)", file_name='nowcast_model_comparison_pre_covid', rows=rows, header_count=4)
