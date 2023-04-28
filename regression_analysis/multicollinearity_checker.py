"""
PYTHON 3.7 (64 BIT)

read: https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc

Also find a python function to find correlation between all independent variables - CORRELATION MATRIX
"""

import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor

def checking_all_independent_variables_for_collinearity(df):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    # clearing out existing graphs
    pyplot.clf()

    # TODO - debug and figure out all independent variables we want to get
    X_variables = df[['', '', '']]
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_variables.columns
    vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]

    plt.matshow(df.corr())
    plt.show()