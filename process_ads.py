# Power regression while sales depend on TV ads

import pandas as pd
import matplotlib.pyplot as plot
import numpy
from scipy import stats

df = pd.read_csv("./data/ads_data.csv")
df.rename(index={0: "No."})

INDEPENDENT_VARS = df["TV"]
DEPENDENT_VARS = df["sales"]

slope, intercept, r, p, std_err = stats.linregress(
    numpy.log(INDEPENDENT_VARS), numpy.log(DEPENDENT_VARS))

print({"slope": slope, "intercept": intercept, "r": r, "p": p,
      "std_err": std_err, "euler_number": intercept*numpy.e})


def model(x):
    return (numpy.e * intercept)*x**slope


plot.title("sales dependence on TV ads")
plot.xlabel("TV ads")
plot.ylabel("sales")
plot.yscale("log")

plot.scatter(INDEPENDENT_VARS, DEPENDENT_VARS)
plot.scatter(INDEPENDENT_VARS, model(INDEPENDENT_VARS))
plot.show()
