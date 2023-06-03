from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

"""
VENV 3.7
"""


def nowcast_regression(X, Y, y_with_date):
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

    intercept = model_train.intercept_
    coef = model_train.coef_

    train_score = model_train.score(x_train, y_train) # 0.9874287562139491
    test_score = model_train.score(x_test, y_test) # 0.9844758436829304

    # https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
    x_test_full = x_test.reset_index().merge(y_with_date.reset_index(), how='inner', on='index')
    x_test_full['y_pred'] = model_train.predict(x_test_full.drop(columns=['date_grouped', 'gdp', 'index']))

    y_pred = x_test['y_pred'].to_numpy()
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    # TODO - Add a title header

    # Prep:
    plt.clf()
    y_test_full = y_test.to_frame().reset_index().merge(y_with_date.reset_index(), how='inner', on='index')
    # Plotting nowcast gdp vs actual gdp
    plt.figure()
    y_test_full = y_test_full.sort_values('date_grouped')
    x_test_full = x_test_full.sort_values('date_grouped')
    plt.plot(y_test_full['date_grouped'], y_test_full['gdp_x'], '-o')
    plt.plot(x_test_full['date_grouped'], x_test_full['y_pred'], '-o')
    # TODO - Add labels and title

    # Calculating RMSE - Root mean squared error https://stackoverflow.com/questions/69844967/calculation-of-mse-and-rmse-in-linear-regression
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    # TODO - WIP boosting the model
    from sklearn.ensemble import GradientBoostingRegressor
    gbr_model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
    gbr_train_score = gbr_model.score(x_train, y_train)
    gbr_test_score = gbr_model.score(x_test, y_test)

    from sklearn.ensemble import RandomForestRegressor
    rfr_model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
    rfr_model.score(x_train, y_train)
    rfr_model.score(x_test, y_test)

    #ForecasterAutoreg - recursive forecasting - https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html





