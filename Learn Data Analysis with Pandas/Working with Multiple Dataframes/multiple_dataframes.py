import pandas as pd

# ----------Inner Merge----------

sales = pd.read_csv('sales.csv')
print(sales)
targets = pd.read_csv('targets.csv')
print(targets)

sales_vs_targets = pd.merge(sales, targets)
print(sales_vs_targets)

crushing_it = sales_vs_targets[sales_vs_targets.revenue > sales_vs_targets.target]

men_women = pd.read_csv('men_women_sales.csv')

all_data = sales.merge(targets).merge(men_women)
print(all_data)

# ----------Merge on Specific Column----------
orders = pd.read_csv('orders.csv')
print(orders)
products = pd.read_csv('products.csv')
print(products)

orders_products = pd.merge(
  orders,
  products.rename(columns={'id':'product_id'})
)
print(orders_products)

orders_products = pd.merge(
  orders,
  products,
  left_on='product_id',
  right_on='id',
  suffixes=['_orders', '_products']
)
print(orders_products)


# ----------Mismatched Merges----------
orders = pd.DataFrame([
    [1, 3, 2, 1, '2017-01-01'],
    [2, 2, 2, 3, '2017-01-01'],
    [3, 5, 1, 1, '2017-01-01'],
    [4, 2, 3, 2, '2016-02-01'],
    [5, 3, 3, 3, '2017-02-01']],
    columns=['id', 'product_id', 'customer_id', 'quantity', 'timestamp']
)

products = pd.DataFrame([
    [1, 'thing-a-ma-jig', 5],
    [2, 'whatcha-ma-call-it', 10],
    [3, 'doo-hickey', 7],
    [4, 'gizmo', 3]],
    columns=['product_id', 'description', 'price']
)

print(orders)
print(products)

merged_df = orders.merge(products)
print(merged_df)  # product_id of 5 has been dropped as it didn't exist in the products dataframe


# ----------Outer Merge----------
store_a = pd.read_csv('store_a.csv')
print(store_a)
store_b = pd.read_csv('store_b.csv')
print(store_b)

store_a_b_outer = store_a.merge(store_b, how='outer')
print(store_a_b_outer)


# ----------Left and Right Merge----------

# left merge means that all data from table on left (store_a) is kept and only matching values in table on right are added to the new dataframe
store_a_b_left = store_a.merge(store_b, how='left')
store_b_a_left = store_b.merge(store_a, how='left')  # same could be accomplished using store_a.merge(store_b, how='right')

print(store_a_b_left)
print(store_b_a_left)


# ----------Concatenate DataFrames----------
bakery = pd.read_csv('bakery.csv')
print(bakery)
ice_cream = pd.read_csv('ice_cream.csv')
print(ice_cream)

menu = pd.concat([bakery, ice_cream])  # concat will only work if the columns are the same in both dataframes
print(menu)

# ----------Review----------
visits = pd.read_csv('visits.csv',
                        parse_dates=[1])
checkouts = pd.read_csv('checkouts.csv',
                        parse_dates=[1])

print(visits)
print(checkouts)

v_to_c = visits.merge(checkouts)
v_to_c['time'] = v_to_c['checkout_time'] - v_to_c['visit_time']
print(v_to_c.time.mean())

