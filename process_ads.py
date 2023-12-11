# Power regression while sales depend on TV ads

import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv("./data/ads_data.csv", index_col=0)

# print(df.info())

# Take a look to see what sales could possibly depend on
# look for high correlation
# print(df.corr())

# Multi regression with OLS method
results = smf.ols(
    'np.log(sales) ~ np.log(TV) + radio + newspaper', data=df).fit()
print(results.summary())
