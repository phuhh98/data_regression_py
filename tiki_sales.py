import pandas as pd
import numpy
import matplotlib.pyplot as plt

DATA_PATH = "./data/accessories_data.csv"

df = pd.read_csv(DATA_PATH)
df.rename(index={0: "No."})

filterDf = df.drop(["category", "No.", "id", "name", "description", "fulfillment_type",
                    "brand", "pay_later", "current_seller", "date_created", "vnd_cashback", "has_video", "favourite_count"], axis="columns")


def calDeductPercent(originalPrice, price):
    return (1 - price/originalPrice)*100


calDeductPercentWithArr = numpy.frompyfunc(calDeductPercent, 2, 1)

ORIGINAL_PRICE = numpy.array(filterDf["original_price"])
PRICE = numpy.array(filterDf["price"])

PRICE_DEDUCT_PERCENTAGE = calDeductPercentWithArr(ORIGINAL_PRICE, PRICE)

filterDf.insert(2, "price_deduct_perentage", PRICE_DEDUCT_PERCENTAGE)

# Overview of supplied data
print(filterDf.info())

# Overview correlation of quantity_sold vs other aspect to find where to make calculation
print(filterDf.corr())

filterDf.plot(kind='scatter', x='price_deduct_perentage', y='quantity_sold')
filterDf.plot(kind='scatter', x='review_count', y='quantity_sold')
filterDf.plot(kind='scatter', x='rating_average', y='quantity_sold')

plt.show()
