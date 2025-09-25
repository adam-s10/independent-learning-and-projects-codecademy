import pandas as pd
import numpy as np

# ----------Calculating Column Statistics----------
orders = pd.read_csv('orders.csv')
print(orders.head(10))

most_expensive = orders.price.max()  # max value in price column
num_colors = orders.shoe_color.nunique()  # number of unique values in shoe_color column

# ----------Calculating Aggregate Functions----------
# In general, the following syntax is used to create aggregates
# df.groupby('column1').column2.measurement()

pricey_shoes = orders.groupby('shoe_type').price.max()  # group by shoe type and then get the max price of each type
print(pricey_shoes)
print(type(pricey_shoes))

pricey_shoes = orders.groupby('shoe_type').price.max().reset_index()  # return a dataframe rather than series
print(pricey_shoes)
print(type(pricey_shoes))

# calculate the 25th percentile of price of shoes based off color
cheap_shoes = orders.groupby('shoe_color').price.apply(lambda x: np.percentile(x, 25)).reset_index()
print(cheap_shoes)

# calculate the number of shoes of each type with each color were sold
shoe_counts = orders.groupby(['shoe_type', 'shoe_color']).id.count().reset_index()  # note you don't have to use id to use .count()
print(shoe_counts)

# ----------Pivot Tables----------
shoe_counts_pivot = shoe_counts.pivot(
  columns='shoe_color',  # column to be pivoted
  index='shoe_type',  # column to be rows
  values='id'  # column to be values (this is misleading as id is technically the count of shoe type and color sold)
).reset_index()
print(shoe_counts_pivot)

# ----------Review----------
user_visits = pd.read_csv('page_visits.csv')
print(user_visits.head())

click_source = user_visits.groupby('utm_source').id.count().reset_index()
print(click_source)

click_source_by_month = user_visits.groupby(['utm_source', 'month']).id.count().reset_index()

click_source_by_month_pivot = click_source_by_month.pivot(
  columns='month',
  index='utm_source',
  values='id'
).reset_index()

print(click_source_by_month_pivot)
