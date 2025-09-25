import pandas as pd

visits = pd.read_csv('visits.csv',
                     parse_dates=[1])
cart = pd.read_csv('cart.csv',
                   parse_dates=[1])
checkout = pd.read_csv('checkout.csv',
                       parse_dates=[1])
purchase = pd.read_csv('purchase.csv',
                       parse_dates=[1])

print(visits.head())
print(cart.head())
print(checkout.head())
print(purchase.head())

visits_cart_left = visits.merge(cart, how='left')
print(visits_cart_left.shape)
# Number of users who did not add an item to the cart
num_cart_empty = len(visits_cart_left[visits_cart_left.cart_time.isnull()])
print(num_cart_empty)

percent_cart_empty = float(num_cart_empty) / len(visits_cart_left)
print('Percent of users who did not add an item to the cart {:.4f}'.format(percent_cart_empty))

cart_checkout_left = cart.merge(checkout, how='left')
# Number of users who did not check out with an item in the cart
num_non_checkout = len(cart_checkout_left[cart_checkout_left.checkout_time.isnull()])
percent_non_checkout = float(num_non_checkout) / len(cart_checkout_left)
print('Percent of users who added an item but did not chekout {:.4f}'.format(percent_non_checkout))

all_data = visits.merge(cart, how='left').merge(checkout, how='left').merge(purchase, how='left')
print(all_data.head())

num_non_purchase = len(all_data[(~all_data.checkout_time.isnull()) & (all_data.purchase_time.isnull())])
percent_non_purchase = float(num_non_purchase) / len(all_data)
print('Number of users that checked out but did not purchase {:.4f}'.format(percent_non_purchase))
# Percent of users who did not add an item to the cart was the weakest link with a high 0.8260. This would suggest that the styles are not what people are looking for or their price. Cool T-Shirts Inc. could incorporate sales

all_data['time_to_purchase'] = all_data.purchase_time - all_data.visit_time
print(all_data.time_to_purchase)
mean_time_to_purchase = all_data.time_to_purchase.mean()
print(mean_time_to_purchase)
