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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

def checking_all_independent_variables_for_collinearity(df, covid=False):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)

    # clearing out existing graphs
    pyplot.clf()

    # calculating VIF
    if covid:
        list_of_columns = ['frequency_baftas', 'frequency_cinema_showings', 'frequency_cinemas_near_me', 'frequency_films',
                           'monthly_gross', 'number_of_cinemas', 'average_temperature', 'weighted_ranking']
    else:
        list_of_columns = ['frequency_baftas', 'frequency_cinema_showings', 'frequency_cinemas_near_me', 'frequency_films',
                           'monthly_gross', 'number_of_cinemas', 'sentiment', 'average_temperature', 'weighted_ranking']

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
    if covid:
        fig.write_image("vif_df_all_variables_incl_covid.png", scale=2)
    else:
        fig.write_image("vif_df_all_variables.png", scale=2)
    fig.show()

    # PCA
    # https://towardsdatascience.com/a-visual-learners-guide-to-explain-implement-and-interpret-principal-component-analysis-cc9b345b75be
    pca_df = df[list_of_columns]
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(pca_df)
    n_components = 6
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_df)
    principal_components = [('PC%s' % i) for i in range(1,n_components+1)]
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    plt.figure(figsize=(10, n_components))
    plt.bar(principal_components, explained_variance, color='#D3E7EE')
    plt.plot(principal_components, cumulative_variance, 'o-', linewidth=2, color='#C6A477')

    # add cumulative variance as the annotation
    for i, j in zip(principal_components, cumulative_variance):
        plt.annotate(str(round(j, 2)), xy=(i, j))

    pca_component_df = pd.DataFrame(pca.components_, columns=pca_df.columns)

    # Seaborn visualisation
    customPalette = sns.color_palette("blend:#D3E7EE,#C6A477", as_cmap=True)
    plt.figure(figsize=(24, 3))
    sns.heatmap(pca_component_df, cmap=customPalette, annot=True)

    # Calculating VIF for reduced variables
    if covid:
        list_of_columns_reduced = ['monthly_gross', 'frequency_cinemas_near_me',
                                   'frequency_baftas', 'average_temperature',
                                   'weighted_ranking']
    else:
        list_of_columns_reduced = ['monthly_gross', 'frequency_cinemas_near_me',
                                   'frequency_baftas', 'average_temperature',
                                   'sentiment','weighted_ranking']

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
    if covid:
        fig.write_image("vif_df_reduced_ind_variables_incl_covid.png", scale=2)
    else:
        # fig.write_image("vif_df_reduced_ind_variables.png", scale=2)
        fig.write_image("vif_df_reduced_ind_variables_no_sent.png", scale=2)
    fig.show()

    # correlation matrix (related to, but different from the VIF values)
    rs = np.random.RandomState(0)
    df = pd.DataFrame(rs.rand(10, len(list_of_columns)), columns=list_of_columns)
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm')

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.4f',
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    if covid:
        plt.savefig('correlation_matrix_incl_covid.png', bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig('correlation_matrix.png', bbox_inches='tight', pad_inches=0.0)

    return list_of_columns_reduced