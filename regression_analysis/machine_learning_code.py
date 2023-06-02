from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
"""
VENV 3.7
"""


def nowcast_regression(X, Y):
    """
    Function to nowcast, using machine learning techniques
    """

    x = X.to_numpy()
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
    predictions = model_train.predict(x_test)
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

