"""
PYTHON 3.7 (64 BIT)

read: https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc

Also find a python function to find correlation between all independent variables - CORRELATION MATRIX
https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas

pip install Jinja2
pip install kaleido
"""

import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.figure_factory as ff

from statsmodels.stats.outliers_influence import variance_inflation_factor

def checking_all_independent_variables_for_collinearity(df):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    # clearing out existing graphs
    pyplot.clf()

    df['box_office_disparity'] = df['monthly_gross_ratio_rank_1'] - df['monthly_gross_ratio_rank_5']

    # calculating VIF
    list_of_columns = ['frequency_academy_awards', 'frequency_cinema_showings', 'frequency_cinemas_near_me', 'frequency_films',
         'monthly_gross', 'number_of_cinemas', 'sentiment', 'average_temperature', 'box_office_disparity']


    X_variables = df[list_of_columns]
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_variables.columns
    vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]

    # Saving VIF data from all independent variables to a .png file
    fig = ff.create_table(vif_data)
    fig.update_layout(
        autosize=False,
        width=500,
        height=200,
    )
    fig.write_image("vif_df_all_variables.png", scale=2)
    fig.show()

    list_of_columns_reduced = ['monthly_gross', 'frequency_cinemas_near_me',
                               'box_office_disparity', 'frequency_academy_awards', 'average_temperature',
                               'sentiment']

    X_variables_reduced = df[list_of_columns_reduced]
    vif_data_reduced = pd.DataFrame()
    vif_data_reduced["feature"] = X_variables_reduced.columns
    vif_data_reduced["VIF"] = [variance_inflation_factor(X_variables_reduced.values, i) for i in range(len(X_variables_reduced.columns))]

    # Saving VIF data from 3 independent variables to a .png file (after seeing high correlation in other variables
    fig = ff.create_table(vif_data_reduced)
    fig.update_layout(
        autosize=False,
        width=500,
        height=200,
    )
    fig.write_image("vif_df_reduced_ind_variables.png", scale=2)
    fig.show()

    # correlation matrix (related to, but different from the VIF values)
    rs = np.random.RandomState(0)
    df = pd.DataFrame(rs.rand(10, 9), columns=list_of_columns)
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm')

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.4f',
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    plt.savefig('correlation_matrix.png', bbox_inches='tight', pad_inches=0.0)

    return ''