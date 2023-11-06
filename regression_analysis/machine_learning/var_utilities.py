from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def granger_casuality_test(data, variables, verbose=True):
    maxlag = 12
    test = 'ssr_chi2test'

    df = pd.DataFrame(np.zeros((len(data.columns), len(data.columns))), columns=data.columns, index=data.columns)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in data.columns]
    df.index = [var + '_y' for var in data.columns]

    grangerdf = df

    return grangerdf

def adf(data,  variable=None, signif=0.05, verbose=False):

    r = adfuller(data, autolag='AIC')
    output = {'test_statistic':np.round(r[0], 4), 'pvalue':np.round(r[1], 4), 'n_lags':np.round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']

    # Print Summary
    print(' Augmented Dickey-Fuller Test on ', variable)
    print(' Null Hypothesis: Variable is Non-Stationary.')
    print(' Significance Level         = ', signif)
    print(' Test Statistic             = ', output["test_statistic"])
    print(' No. Lags Chosen (lowest AIC)= ', output["n_lags"])

    for key,val in r[4].items():
        print(' Critical value', key, np.round(val, 3))

    if p_value <= signif:
        print("p-Value = ", p_value, ". P value is less than critical value. Reject Null H. Series is Stationary.")
    else:
        print("p-Value = ", p_value, ".  P value is not less than critical value. Weak evidence to reject the Null H. Series is Non-Stationary")