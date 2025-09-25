import pandas as pd
# ----------Create Data----------
# create a dataframe containing product information using dictionary
df1 = pd.DataFrame({
  'Product ID': [1, 2, 3, 4],
  # add Product Name and Color here
  'Product Name': ['t-shirt', 't-shirt', 'skirt', 'skirt'],
  'Color': ['blue', 'green', 'red', 'black']
})

print(df1)
print()

# create a dataframe using a list of lists
df2 = pd.DataFrame([
  [1, 'San Diego', 100],
  [2, 'Los Angeles', 120],
  # Fill in rows 3 and 4
  [3, 'San Francisco', 90],
  [4, 'Sacramento', 115]
],
  columns=[  # must add column names. This allows us to control the order that data will appear in
    #add column names here
    'Store ID',
    'Location',
    'Number of Employees'
  ])

print(df2)
print()

# ----------Read data----------
# read from file in current directory as default
df = pd.read_csv('sample.csv')
print(df)
print()

# Inspect a dataframe
#load the CSV below:
df = pd.read_csv('imdb.csv')
print(df.head())
print(df.info())
print()

# ----------Select Columns----------
df = pd.DataFrame([
  ['January', 100, 100, 23, 100],
  ['February', 51, 45, 145, 45],
  ['March', 81, 96, 65, 96],
  ['April', 80, 80, 54, 180],
  ['May', 51, 54, 54, 154],
  ['June', 112, 109, 79, 129]],
  columns=['month', 'clinic_east',
           'clinic_north', 'clinic_south',
           'clinic_west']
)

# you can select by columns similar to a dictionary (df['column']), or by df.column
clinic_north = df.clinic_north  # you can only use this notation if the column name follows python's rules for variable names
clinic_north2 = df['clinic_north']
print(type(clinic_north))  # when selecting by column, a series is returned
print(type(df))
print()

# ----------Selecting Multiple Columns----------
clinic_north_south = df[['clinic_north', 'clinic_south']]
print(type(clinic_north_south))  # returns a dataframe instead of a series
print()

# ----------Select Rows----------
march = df.iloc[2]

# ----------Selecting multiple rows----------
april_may_june = df.iloc[3:7]  # select from row three to row 6 (up to but not including 7)
print(april_may_june)
print()
april_may_june = df.iloc[-3:]  # select from the third last row until the end
print(april_may_june)
print()

# ----------Select rows with logic----------
january = df[df.month == 'January']  # select row where month is January
print(january)
print()

march_april = df[(df.month == 'March') | (df.month == 'April')]  # select where month is March or April (multiple conditions must be in parentheses)
print(march_april)
print()

january_february_march = df[df.month.isin([  # select where Month appears in provided list
  'January',
  'February',
  'March'
])]

print(january_february_march)
print()

# ----------Setting Indices----------
df2 = df.loc[[1, 3, 5]]
print(df2)
print()

df3 = df2.reset_index()  # this will create an 'index' column of the old indices. It should be dropped if not needed
print(df3)
print()

df2.reset_index(drop=True, inplace=True)  # drops the 'index' column and directly manipulates the already declared dataframe
print(df2)
print()

# ----------Review----------
orders = pd.read_csv('shoefly.csv')
print(orders.head())

emails = orders.email

frances_palmer = orders[(orders.first_name == 'Frances') & (orders.last_name == 'Palmer')]

comfy_shoes = orders[orders.shoe_type.isin([
  'clogs',
  'boots',
  'ballet flats'
])]
