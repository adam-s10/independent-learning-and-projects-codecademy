import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
# Boston housing dataset
california = fetch_california_housing()

df = pd.DataFrame(california.data, columns=california.feature_names)
print(df.columns)

# Set the x-values to the nitrogen oxide concentration:
X = df[['AveRooms']]
# Y-values are the prices:
y = california.target

# Can we do linear regression on this?
line_fitter = LinearRegression()
line_fitter.fit(X, y)


plt.scatter(X, y, alpha=0.4)
# Plot line here:
plt.plot(X, line_fitter.predict(X))

plt.title("California Housing Dataset")
plt.xlabel("House Age")
plt.ylabel("House Price ($)")
plt.show()
