import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv('honeyproduction.csv')

print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index()
# extract year and total production columns
X = prod_per_year['year']
X = X.values.reshape(-1, 1)
y = prod_per_year['totalprod']

# plot and display X and y values
plt.scatter(X, y)

# use linear regression and fit to data
regr = linear_model.LinearRegression()
regr.fit(X, y)

# gradient and y-intercept of line of best fit
print(regr.coef_[0])
print(regr.intercept_)

# create list of y values predicted for given x values and plot the line
y_predict = regr.predict(X)
plt.plot(X, y_predict)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.show()

# create x values for future years (2013-2050)
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)

# use model to predict what the production would be for the future years
future_predict = regr.predict(X_future)
plt.scatter(X, y)
plt.plot(X, y_predict)
plt.plot(X_future, future_predict)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.show()
