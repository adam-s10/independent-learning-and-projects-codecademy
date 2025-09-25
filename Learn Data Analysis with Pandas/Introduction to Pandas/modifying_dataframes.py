import pandas as pd

# ----------Adding a Column----------
df = pd.DataFrame([
  [1, '3 inch screw', 0.5, 0.75],
  [2, '2 inch nail', 0.10, 0.25],
  [3, 'hammer', 3.00, 5.50],
  [4, 'screwdriver', 2.50, 3.00]
],
  columns=['Product ID', 'Description', 'Cost to Manufacture', 'Price']
)

# Add columns here
df['Sold in Bulk?'] = ['Yes', 'Yes', 'No', 'No']  # remember, list must be equal in size to the existing df
print(df)

df['Is taxed?'] = 'Yes'  # if all rows will have the same value
print(df)

df['Margin'] = df['Price'] - df['Cost to Manufacture']  # add a column by performing a function on existing columns
print(df)

# ----------Performing Column Operations----------
df = pd.DataFrame([
  ['JOHN SMITH', 'john.smith@gmail.com'],
  ['Jane Doe', 'jdoe@yahoo.com'],
  ['joe schmo', 'joeschmo@hotmail.com']
],
columns=['Name', 'Email'])

# Add columns here
df['Lowercase Name'] = df.Name.apply(str.lower)
print(df)

# ----------Applying a lambda to a column----------
df = pd.read_csv('employees.csv')

# Add columns here
get_last_name = lambda x: x.split()[-1]

df['last_name'] = df.name.apply(get_last_name)

print(df)

# ----------Applying a lambda to a row----------
total_earned = lambda row: (row.hourly_wage * 40) + ((row.hourly_wage * 1.5) * (row.hours_worked - 40)) \
    if row.hours_worked > 40 \
    else row.hourly_wage * row.hours_worked

df['total_earned'] = df.apply(total_earned, axis=1)
print(df)

# ----------Renaming columns----------
df = pd.read_csv('imdb.csv')
# Rename columns here
df.columns = ['ID', 'Title', 'Category', 'Year Released', 'Rating']  # rename all columns
print(df)

df.rename(columns={'name': 'movie_title'}, inplace=True)  # rename individual column or multiple (depending on dict size)
print(df)

# ----------Review----------
orders = pd.read_csv('shoefly2.csv')
print(orders.head())

orders['shoe_source'] = orders['shoe_material'].apply(
  lambda x: 'vegan' if x != 'leather' else 'animal'
)

orders['salutation'] = orders.apply(
  lambda row: f'Dear Mr. {row.last_name}' if row['gender'] == 'male' else f'Dear Ms. {row.last_name}', axis=1
)
