# Power regression while sales depend on TV ads

import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv("./data/ads_data.csv", index_col=0)

INDEPENDENT_VARS = df["TV"]
DEPENDENT_VARS = df["sales"]


print(df.info())

# Take a look to see what sales could possibly depend on
# look for high correlation
print(df.corr())

# Multi regression with OLS method
results = smf.ols('sales ~ TV + radio + newspaper', data=df).fit()
print(results.summary())


# def model(x, intercept, slope):
#     return (np.e * intercept)*x**slope


# plot.title("sales dependence on TV ads")
# plot.xlabel("TV ads")
# plot.ylabel("sales")
# plot.yscale("log")

# plot.scatter(INDEPENDENT_VARS, DEPENDENT_VARS)
# plot.scatter(INDEPENDENT_VARS, model(INDEPENDENT_VARS, *results.params))
# plot.show()
