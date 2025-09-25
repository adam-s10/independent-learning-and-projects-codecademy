import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

all_tweets = pd.read_json('random_tweets.json', lines=True)
print(len(all_tweets))
print(all_tweets.columns)
print(all_tweets.loc[0]['text'])
print(all_tweets.loc[0]['user'])
print(all_tweets.loc[0]['user']['location'])

median_retweet_count = np.median(all_tweets['retweet_count'])
print(median_retweet_count)
all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > median_retweet_count, 1, 0)  # come back and change values
print(all_tweets['is_viral'].value_counts())

all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)

# For the rest of this project, we will be using these three features, but we encourage you to create your own. Here are some potential ideas for more features.
#
# The number of hashtags in the tweet. You can find this by looking at the text of the tweet and using the .count() function with # as a parameter.
# The number of links in the tweet. Using a similar strategy to the one above, use .count() to count the number of times http appears in the tweet.
# The number of words in the tweet. Call .split() on the text of a tweet. This will give you a list of the words in the tweet. Find the length of that list.
# The average length of the words in the tweet.
all_tweets['num_hashtags'] = all_tweets.apply(lambda tweet: tweet['text'].count('#'), axis=1)
all_tweets['num_links'] = all_tweets.apply(lambda tweet: tweet['text'].count('http'), axis=1)
all_tweets['num_words'] = all_tweets.apply(lambda tweet: len(tweet['text'].split()), axis=1)


labels = all_tweets[['is_viral']]
data_original = all_tweets[['tweet_length', 'followers_count', 'friends_count']]
data_expanded = all_tweets[['tweet_length', 'followers_count', 'friends_count', 'num_hashtags', 'num_links', 'num_words']]

scaled_data_original = scale(data_original, axis=0)
scaled_data_expanded = scale(data_expanded, axis=0)
print(scaled_data_original[0])

# train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size=.2, random_state=1)
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(train_data, train_labels)
# print(classifier.score(test_data, test_labels))

scores_original = []
scores_expanded = []
for k in range(1, 201):
    train_data, test_data, train_labels, test_labels = train_test_split(scaled_data_original, labels, test_size=.2,
                                                                        random_state=1)
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_data, train_labels)
    scores_original.append(classifier.score(test_data, test_labels))

    train_data, test_data, train_labels, test_labels = train_test_split(scaled_data_expanded, labels, test_size=.2,
                                                                        random_state=1)
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_data, train_labels)
    scores_expanded.append(classifier.score(test_data, test_labels))

plt.xlabel('k')
plt.ylabel('Accuracy Scores')
plt.plot(range(1,201), scores_original)
plt.show()

plt.plot(range(1, 201), scores_expanded)
plt.show()