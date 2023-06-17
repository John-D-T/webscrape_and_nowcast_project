import matplotlib.pyplot as plt
from regression_analysis.machine_learning.var_utilities import granger_casuality_test, adf
from statsmodels.tsa.api import VAR
import pandas as pd



def plot_importance_features(model, color, covid_features, non_covid_features, model_name, coef, covid=False):
    fig, ax = plt.subplots()
    if coef:
        importance = model.coef_
    else:
        importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    if covid:
        feature_list = covid_features
    else:
        feature_list = non_covid_features
    bars = ax.barh(feature_list, importance, color=color)
    ax.bar_label(bars)
    if covid:
        title = 'Feature importance - %s (including covid)' % model_name
        plt.title(title)
    else:
        title = 'Feature importance - %s (pre-covid)' % model_name
        plt.title(title)
    plt.show()


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
    # https://taufik-azri.medium.com/forecasting-through-economic-uncertainty-multivariable-time-series-analysis-with-var-and-prophet-e6b801962acb
    # https://github.com/fickaz/time-series-for-business/blob/master/Forecasting.ipynb

    # GRANGER TEST - WIP
    # # statsmodels.tools.sm_exceptions.InfeasibleTestError: The Granger causality test statistic cannot be compute because the VAR has a perfect fit of the data.
    #granger_df = granger_casuality_test(data=var_df, variables=var_df.columns)

    # ADF TEST - differentiate if needed
    # 1st order differencing
    df_differenced = var_train.diff().dropna()

    for name, column in df_differenced.iteritems():
        adf(column, variable=column.name)
        print('\n')


    var_model = VAR(df_differenced)
    x = var_model.select_order(maxlags=8)
    x.summary()
    nobs = len(df_differenced.index)
    # An asterix indicates the right order of the VAR model. More specifically, we're choosing the optimal lag that minimizes AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) out-of-sample error prediction
    print(x.summary())
    var_model_fitted = var_model.fit(3)

    # Get the lag order
    lag_order = var_model_fitted.k_ar

    # Separate input data for forecasting
    # the goal is to forecast based on the last 4 inputs (since the lag is 4)
    forecast_input = df_differenced.values[-lag_order:]
    # Forecast
    ## we insert the last four values and inform the model to predict the next 10 values

    fc = var_model_fitted.forecast(y=forecast_input, steps=nobs)

    ## organize the output into a clear DataFrame layout, add '_f' suffix at each column indicating they are the forecasted values
    df_forecast = pd.DataFrame(fc, index=var_df.index[-nobs:], columns=var_df.columns + '_f')
    df_forecast

    # get a copy of the forecast
    df_fc = df_forecast.copy()

    # get column name from the original dataframe
    columns = var_train.columns

    # Roll back from the 1st order differencing
    # we take the cumulative sum (from the top row to the bottom) for each of the forecasting data,
    ## and the add to the previous step's original value (since we deduct each row from the previous one)
    ## we rename the new forecasted column with the prefix 'forecast'

    for col in columns:
        df_fc[str(col) +'_forecast'] = var_train[col].iloc[-1] + df_fc[str(col) +'_f'].cumsum()

    ## if you perform second order diff, make sure to get the difference from the last row and second last row of df_train
    for col in columns:
        df_fc[str(col) + '_first_differenced'] = (var_train[col].iloc[-1] - var_train[col].iloc[-2]) + df_fc[
            str(col) + '_f'].cumsum()
        df_fc[str(col) + '_forecast'] = var_train[col].iloc[-1] + df_fc[str(col) + '_first_differenced'].cumsum()
    df_results = df_fc

    # Plotting forecast (WIP) - for some reason gdp keeps going up, all the way to 600.
    # TODO - fix this. It's because we're basing all forecasts on the first 3 GDP values. Of course it's going to not have any bumps/fluctuations
    # TODO - check https://medium.com/mlearning-ai/how-i-used-statsmodels-vector-autoregression-var-to-forecast-on-multivariate-training-data-fc867eb6de8b
    fig = var_model_fitted.plot_forecast(10)
    fig, ax = plt.subplots()

    var_test.reset_index()['gdp'].plot(color='k', label='Actual')
    df_results.reset_index()['gdp_forecast'].plot(color='r', label='Predicted')

    plt.title('VAR Model: Title TBD')
    ax.legend()

    return ''


if __name__ == '__main__':
    pass