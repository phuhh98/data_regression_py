import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

DATA_PATH = "./data/accessories_data.csv"

df = pd.read_csv(DATA_PATH, index_col=0)

filterDf = df.drop(["category", "id", "name", "description", "fulfillment_type",
                    "brand", "pay_later", "current_seller", "date_created", "vnd_cashback", "has_video", "favourite_count"], axis="columns")


def calDeductPercent(originalPrice, price):
    return (1 - price/originalPrice)*100


calDeductPercentWithArr = np.frompyfunc(calDeductPercent, 2, 1)

filterDf.dropna()

ORIGINAL_PRICE = np.array(filterDf["original_price"])
PRICE = np.array(filterDf["price"])

PRICE_DEDUCT_PERCENTAGE = calDeductPercentWithArr(ORIGINAL_PRICE, PRICE)

filterDf.insert(2, "price_deduct_perentage", PRICE_DEDUCT_PERCENTAGE)

filterDf['quantity_sold'].replace(0, 1, inplace=True)
filterDf['review_count'].replace(0, 1, inplace=True)
filterDf['price_deduct_perentage'].replace(0, 0.000000000001, inplace=True)


# Overview of supplied data
# print(filterDf.info())

results = smf.ols(
    'np.log(quantity_sold) ~ np.log(review_count) * rating_average * price_deduct_perentage', data=filterDf).fit()

print(results.summary())
fModel = open("./results/models.pkl", "w")
results.save("./results/models.pkl")
fModel.close()

for index in range(len(results.summary().tables)):
    f = open("./results/table_{number}.html".format(number=index), "w")
    f.write(results.summary().tables[index].as_html())
    f.close()

# Overview correlation of quantity_sold vs other aspect to find where to make calculation
print(filterDf.corr())

# filterDf.plot(kind='scatter', x='price_deduct_perentage', y='quantity_sold')
# plt.yscale("log")

# filterDf.plot(kind='scatter', x='review_count', y='quantity_sold')
# plt.yscale("log")

# filterDf.plot(kind='scatter', x='rating_average', y='quantity_sold')
# plt.yscale("log")

# plt.show()
