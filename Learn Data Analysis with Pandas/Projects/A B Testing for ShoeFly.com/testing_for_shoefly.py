import pandas as pd

ad_clicks = pd.read_csv('ad_clicks.csv')
print(ad_clicks.head())

num_views_from_source = ad_clicks.groupby('utm_source').user_id.count().reset_index()
print(num_views_from_source)

ad_clicks['is_click'] = ~ad_clicks.ad_click_timestamp.isnull()
print(ad_clicks.head())

clicks_by_source = ad_clicks.groupby(['utm_source', 'is_click']).user_id.count().reset_index()

print(clicks_by_source.head())

clicks_pivot = clicks_by_source.pivot(
  columns='is_click',
  index='utm_source',
  values='user_id'
)
print(clicks_pivot.head())

clicks_pivot['percent_clicked'] = clicks_pivot[True] / (clicks_pivot[False] + clicks_pivot[True])
print(clicks_pivot.head())

print(ad_clicks.groupby('experimental_group').user_id.count().reset_index())

print(ad_clicks.groupby(['experimental_group', 'is_click']).user_id.count().reset_index().pivot(
  columns='experimental_group',
  index='is_click',
  values='user_id'
))

a_clicks = ad_clicks[ad_clicks.experimental_group == 'A']
b_clicks = ad_clicks[ad_clicks.experimental_group == 'B']

a_pivot = a_clicks.groupby(['day', 'is_click']).user_id.count().reset_index().pivot(
  columns='is_click',
  index='day',
  values='user_id'
)

b_pivot = b_clicks.groupby(['day', 'is_click']).user_id.count().reset_index().pivot(
  columns='is_click',
  index='day',
  values='user_id'
)

a_pivot['percent_clicked'] = a_pivot[True] / (a_pivot[False] + a_pivot[True])
b_pivot['percent_clicked'] = b_pivot[True] / (b_pivot[False] + b_pivot[True])

print(a_pivot)
print(b_pivot)