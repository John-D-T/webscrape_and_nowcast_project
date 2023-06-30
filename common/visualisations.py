import matplotlib.pyplot as plt
from regression_analysis.machine_learning.var_utilities import granger_casuality_test, adf
from statsmodels.tsa.api import VAR
import pandas as pd



# def plot_importance_features(model, color, covid_features, non_covid_features, model_name, coef, covid=False):
    # fig, ax = plt.subplots()
    # if coef:
    #     importance = model.coef_
    # else:
    #     importance = model.feature_importances_
    # for i, v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i, v))
    #
    # # plot feature importance
    # if covid:
    #     feature_list = covid_features
    # else:
    #     feature_list = non_covid_features
    # bars = ax.barh(feature_list, importance, color=color)
    # ax.bar_label(bars)
    # if covid:
    #     title = 'Feature importance - %s (including covid)' % model_name
    #     plt.title(title)
    # else:
    #     title = 'Feature importance - %s (pre-covid)' % model_name
    #     plt.title(title)
    # plt.show()
def plot_importance_features(list_of_feature_importance_coef):
    # TODO - convert list_of_feature_importance_coef to a df?

    for coefficient_and_date in list_of_feature_importance_coef:
        feature_df = pd.DataFrame(data=coefficient_and_date, columns=('date_grouped', 'feature', 'coefficient'))
        fig, ax = plt.subplots()

        plt.plot(feature_df['date_grouped'], feature_df['coefficient'], '-o', label=feature_df['feature'][0],
                 markersize=3)
        # Setting max number of ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))


def append_to_importance_feature_coef(model, list_of_feature_importance_coef, date):
    importance = model.coef_
    feature = model.feature_names_in_
    for a, b, c in zip(feature, importance, list_of_feature_importance_coef):
        print('Feature: %s, Score: %.5f' % (a, b))
        c.append([date, a, b])

    return list_of_feature_importance_coef


def plot_nowcast(model, x_test_full, y_test_full, covid_features, non_covid_features, color, model_label, model_name,
                 covid=False):
    fig, ax = plt.subplots()

    if covid:
        x_test_full['y_pred_%s' % model_name] = model.predict(x_test_full[covid_features])
    else:
        x_test_full['y_pred_%s' % model_name] = model.predict(x_test_full[non_covid_features])

    y_pred = x_test_full['y_pred_%s' % model_name].to_numpy()

    plt.plot(y_test_full['date_grouped'], y_test_full['gdp_x'], '-o', label="actual gdp", markersize=3)
    plt.plot(x_test_full['date_grouped'], x_test_full['y_pred_%s' % model_name], '-o', label=model_label, markersize=3, color=color)
    leg = plt.legend(loc='upper center')
    plt.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if covid:
        plt.title('Nowcast test set - %s (including covid)' % model_name)
    else:
        plt.title('Nowcast test set - %s (pre-covid)' % model_name)
    plt.show()

    return y_pred

def plot_var_nowcast(var_df, var_train, var_test):
    # https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
    # https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/

    # OPTION 2
    # https://cprosenjit.medium.com/multivariate-time-series-forecasting-using-vector-autoregression-3e5c9b85e42a

    # Select the required columns
    data_var = var_df[['monthly_gross',
         'weighted_ranking',
         'sentiment',
         'average_temperature',
         'frequency_baftas',
         'frequency_cinemas_near_me',
         'gdp',
         'date_grouped'
         ]]
    # Create a date-time index
    data_var.index = pd.DatetimeIndex(var_df['date_grouped'])
    data_var

    # Organize by date
    prediction_start_month = \
        (var_train["date_grouped"].iloc[-1] +
         pd.DateOffset(months=1)) \
            .strftime('%Y-%m-%d')

    forecastingPeriod = 12
    var_train = var_train.drop(columns=['date_grouped'])

    # ADF TEST - differentiate if needed, in order to achieve stationarity
    df_differenced = var_train.diff().dropna()

    var_model = VAR(df_differenced)
    order = var_model.select_order()
    # An asterix indicates the right order of the VAR model. More specifically, we're choosing the optimal lag that minimizes AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) out-of-sample error prediction
    print(order.summary())
    # It appears the optimal lag structure is 3, with the most minimums (using the last 3 month's GDP data when determining next month's GDP)

    var_model_fitted = var_model.fit(3)
    # Fetch the lag order
    lag_order = var_model_fitted.k_ar
    # Produce forecasts for desired number of steps ahead
    predictions = var_model_fitted.forecast(df_differenced.values[-lag_order:], forecastingPeriod)
    # Converts NumPy multidimensional array into Pandas DataFrame
    predictions_df = pd.DataFrame(predictions)
    # Assign the column headers
    predictions_df.columns = \
        ['monthly_gross',
         'weighted_ranking',
         'sentiment',
         'average_temperature',
         'frequency_baftas',
         'frequency_cinemas_near_me',
         'const',
         'gdp'
         ]
    gdp_var_prediction_df = predictions_df['gdp']
    gdp_var_prediction_df

    # Create a DateTimeIndex
    prediction_date_range = pd.date_range(
        prediction_start_month,
        periods=forecastingPeriod,
        freq='MS')
    # Assign the DateTimeIndex as DataFrame index
    gdp_var_prediction_df.index = prediction_date_range
    gdp_var_prediction_df

    # Plotting VAR forecast
    fig, ax = plt.subplots()
    plt.figure(figsize=(14, 8))
    # Plotting the Actuals
    plt.plot(df_differenced.index, df_differenced.gdp, label='Actuals')
    # Plotting the Forecasts
    plt.plot(gdp_var_prediction_df.index, gdp_var_prediction_df, label='Forecasts')
    plt.legend(loc='best')
    plt.title("VAR title TBD - Forecasting")
    plt.show()

    # TODO - Figure out a way to plot the forecast vs the real GDP
    # TODO - incorporate granger and adf test at beginning
    # TODO - remove gdp lag1 from everything!!!
    # # TODO - check https://medium.com/mlearning-ai/how-i-used-statsmodels-vector-autoregression-var-to-forecast-on-multivariate-training-data-fc867eb6de8b
    # TODO - check if graph Y axis is %change in GDP
    # TODO - note that we don't use gdp lag as that is how VAR works.

    # https://taufik-azri.medium.com/forecasting-through-economic-uncertainty-multivariable-time-series-analysis-with-var-and-prophet-e6b801962acb
    # https://github.com/fickaz/time-series-for-business/blob/master/Forecasting.ipynb

    # GRANGER TEST - WIP
    # # statsmodels.tools.sm_exceptions.InfeasibleTestError: The Granger causality test statistic cannot be compute because the VAR has a perfect fit of the data.
    #granger_df = granger_casuality_test(data=var_df, variables=var_df.columns)


    return ''


if __name__ == '__main__':
    pass