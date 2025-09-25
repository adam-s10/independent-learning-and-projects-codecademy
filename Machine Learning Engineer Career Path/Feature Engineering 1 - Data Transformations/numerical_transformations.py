import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ----------Numerical Transformation Introduction----------
coffee = pd.read_csv('starbucks_customers.csv')
print(coffee.columns)
print(coffee.info())

# ----------Centering Your Data----------
ages = coffee.age

# visualize the range in our data
min_age = ages.min()
max_age = ages.max()
print(min_age)
print(max_age)
print(max_age - min_age)

# get the mean of your feature
mean_age = ages.mean()
print(mean_age)
# return new series of all values in column minus mean of column
centered_ages = ages - mean_age
print(centered_ages)

# visualize data
plt.hist(centered_ages)
plt.title('Starbucks Age Data Centered')
plt.xlabel('Distance from Mean')
plt.ylabel('Count')
plt.show()

# ----------Standardizing our Data----------
# Useful when working with features of vastly different scales ie one datapoint is 1-10 and another is 1-10000
std_dev_age = np.std(ages)

## standardize ages
ages_standardized = (ages - mean_age) / std_dev_age

## print the results
print(np.mean(ages_standardized))
print(np.std(ages_standardized))

# ----------Standardizing our Data with Sklearn----------
scaler = StandardScaler()
# takes array and returns it as one column. -1 asks numpy to create number of rows based on our data and 1 asks it to return it as one column
ages_reshaped = np.array(ages).reshape(-1,1)
ages_scaled = scaler.fit_transform(ages_reshaped)
print(np.mean(ages_scaled))
print(np.std(ages_scaled))

# ----------Min-max Normalization----------
# Uses formula Xnorm = (X - Xmin) / (Xmax - Xmin). Does not work well with outliers
# get spent feature
spent = coffee.spent

# find the max spent
max_spent = spent.max()

# find the min spent
min_spent = spent.min()

# find the difference
spent_range = max_spent - min_spent

# normalize your spent feature
spent_normalized = (spent - min_spent) / spent_range

#print your results
print(spent_normalized)

# ----------Min-max Normalization with Sklearn----------
spent_reshaped = np.array(spent).reshape(-1,1)
mmscaler = MinMaxScaler()

reshaped_scaled = mmscaler.fit_transform(spent_reshaped)
print(np.min(reshaped_scaled))
print(np.max(reshaped_scaled))

# ----------Binning our Data----------
print(ages.min())
print(ages.max())

age_bins = [12, 20, 30, 40, 71]

coffee['binned_ages'] = pd.cut(coffee.age, age_bins, right=False)

print(coffee.binned_ages.head(10))

coffee.binned_ages.value_counts().plot(kind='bar')
plt.title('Starbucks Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# ----------Natural Log Transformation----------
# do not use on left-skewed data, only use on right-skewed data
# read in csv file
cars = pd.read_csv('cars.csv')

# set you price variable
prices = cars.sellingprice

# plot a histogram of prices
plt.hist(prices, bins=150, color='g')
plt.xticks(rotation=45)
plt.title('Number of Cars by Selling Price')
plt.xlabel('Selling Price')
plt.ylabel('Number of Cars')
plt.show()

# log transform prices
log_prices = np.log(prices)

# plot a histogram of log_prices
plt.hist(log_prices, bins=150)
plt.title('Log of Cars Selling Price')
plt.show()

