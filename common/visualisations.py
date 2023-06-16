import matplotlib.pyplot as plt
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

def plot_var_nowcast(var_model, var_df):
    # TODO - Plot the forecast for just GDP -
    # https://taufik-azri.medium.com/forecasting-through-economic-uncertainty-multivariable-time-series-analysis-with-var-and-prophet-e6b801962acb
    # https://github.com/fickaz/time-series-for-business/blob/master/Forecasting.ipynb

    # Plots all points
    var_model.plot()
    var_model.plot_forecast(20)

    # Get the lag order
    lag_order = var_model.k_ar
    print(lag_order)
    # Separate input data for forecasting
    # the goal is to forecast based on the last 4 inputs (since the lag is 4)
    forecast_input = var_df.values[-lag_order:]
    # Forecast
    # We insert the last four values and inform the model to predict the next 10 values
    fc = var_model.forecast(y=forecast_input, steps=nobs)
    # organize the output into a clear DataFrame layout, add ‘_f’ suffix # at each column indicating they are the forecasted values
    df_forecast = pd.DataFrame(fc, index=tsdf.index[-nobs:], columns=tsdf.columns + ‘_f’)




if __name__ == '__main__':
    pass