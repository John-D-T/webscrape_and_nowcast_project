"""
PYTHON 3.7 (64 BIT)

read: https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc

Also find a python function to find correlation between all independent variables - CORRELATION MATRIX
https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas

pip install Jinja2
"""

import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor

def checking_all_independent_variables_for_collinearity(df):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    # clearing out existing graphs
    pyplot.clf()

    # calculating VIF

    # TODO - debug and figure out additional independent variables I want to add
    X_variables = df[['monthly_admissions', 'frequency_academy_awards', 'frequency_cinema_showings', 'frequency_cinemas_near_me', 'monthly_gross', 'number_of_cinemas']]
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_variables.columns
    vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]

    # correlation matrix:
    rs = np.random.RandomState(0)
    df = pd.DataFrame(rs.rand(10, 10))
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm')

    return ''