import pandas as pd

from category_encoders import BinaryEncoder, HashingEncoder, TargetEncoder

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# import the data
cars = pd.read_csv('cars.csv')

# check variable types
print(cars.dtypes)


# ----------Ordinal Encoding----------
# Ordinal data is data that has an order or hierarchy between its values
print(cars['condition'].value_counts())
# This shows the following order of best to worst
# - Excellent
# - New
# - Like New
# - Good
# - Fair

# We can convert the values to number using map
# create dictionary of label:values in order
rating_dict = {'Excellent':5, 'New':4, 'Like New':3, 'Good':2, 'Fair':1}

#create a new column
cars['condition_rating'] = cars['condition'].map(rating_dict)

# We can convert the values using scikit-learn (won't work with nan values)

# create encoder and set category order
encoder = OrdinalEncoder(categories=[['Excellent', 'New', 'Like New', 'Good', 'Fair']])

# reshape our feature
condition_reshaped = cars['condition'].values.reshape(-1,1)

# create new variable with assigned numbers
cars['condition_rating'] = encoder.fit_transform(condition_reshaped)


# ----------Label Encoding----------
# Used for nominal data (data that does not have an inherent order)
print(cars['color'].nunique())
# #OUTPUT
# 19

print(cars['color'].value_counts()[:5])
# #OUTPUT
# black     2015
# white     1931
# gray      1506
# silver    1503
# blue       869

# Convert from object type to category type
# convert feature to category type
cars['color'] = cars['color'].astype('category')

# save new version of category codes
cars['color'] = cars['color'].cat.codes

# print to see transformation
print(cars['color'].value_counts()[:5])
# #OUTPUT
# 2     2015
# 18    1931
# 8     1506
# 15    1503
# 3      869

# Transform using Sklearn
# create encoder
encoder = LabelEncoder()

# create new variable with assigned numbers
cars['color'] = encoder.fit_transform(cars['color'])

# Label encoding can lead to issues where the model puts higher precedence over certain values. Eg 'black' was encoded
# to 2 while 'white' was encoded to 18. The model can assign 9 times more weight to 'white' than to 'black' in the example


# ----------One-Hot Encoding----------
# use pandas .get_dummies method to create one new column for each color
ohe = pd.get_dummies(cars['color'])

# join the new columns back onto our cars dataframe
cars = cars.join(ohe)


# ----------Binary Encoding----------
# Useful when we want to one-hot encode values, but it would produce many columns. Eg, if we wanted to one-hot encode the
# colors column of the cars df, there are 19 unique values and would create 19 columns. Instead, we can create 5 columns
# to represent 1-19 in binary. 5 columns because 19 in binary is 10011

#this will create a new data frame with the color column removed and replaced with our 5 new binary feature columns
colors = BinaryEncoder(cols = ['color'], drop_invariant=True).fit_transform(cars)  # set drop_invariant to true to not get 6th column of 0's


# ----------Hashing----------
# Hashing can reduce dimensionality but can cause issues due to collisions, this is when 2 unique categories end up with the same value.
# This could be a solution to your project and dataset if you are not as interested in assessing the impact of any particular categorical value.
#
# For this example, maybe you aren’t interested in knowing which color car had an impact on your final prediction, but you want to be able
# to get the best performance from your model. This encoding solution may be a good approach.

# instantiate our encoder
encoder = HashingEncoder(cols=['color'], n_components=5)

# do a fit transform on our color column and set to a new variable
hash_results = encoder.fit_transform(cars['color'])


# ----------Target Encoding----------
# A Bayesian encoder used to transform categorical data into a hashed numerical values. This can be used on data being
# prepared for regression-based supervised learning. Some drawbacks to this approach are overfitting and unevenly distributed
# values that could lead to extremes. This would work on our color feature by replacing each color with a blend of the mean
# price of that car color and the mean price of all the cars. Had it been predicting something categorical, it would’ve used
# a Bayesian target statistic.

# instantiate our encoder
encoder = TargetEncoder(cols = 'color')

# set the results of our fit_transform to a variable
# the output will be its own pandas series
encoder_results = encoder.fit_transform(cars['color'], cars['sellingprice'])

print(encoder_results.head())
#   color
# 0 11761.881473
# 1 18007.276995
# 2 8458.251232
# 3 14769.292595
# 4 12691.099747


# ----------Encoding date-time variables----------
print(cars['saledate'].dtypes)
# # OUTPUT
# dtype('O')

cars['saledate'] = pd.to_datetime(cars['saledate'])
# #OUTPUT
# datetime64[ns, tzlocal()]

# create new variable for month
cars['month'] = cars['saledate'].dt.month

# create new variable for day of the week
cars['dayofweek'] = cars['saledate'].dt.day

# create new variable for difference between cars model year and year sold
cars['yearbuild_sold'] = cars['saledate'].dt.year - cars['year']


