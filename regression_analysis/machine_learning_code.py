from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston
"""
VENV 3.7
"""

def nowcast_regression(X, Y):
    """
    Function to nowcast, using machine learning techniques
    """

    # x = df['']
    # y = df['']
    x = X.to_numpy()
    y = Y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=100)

    # Seeing the split across training and testing datasets
    print('Number of records in the original dataset: ', len(y))
    print('Number of records in the training dataset: ', len(y_train))
    print('Number of records in the testing dataset: ', len(y_test))

    model = LinearRegression().fit(x_train, y_train)
    intercept = model.intercept_
    coef = model.coef_

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    # TODO - consider other techniques to improve the model score