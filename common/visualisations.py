import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    pass