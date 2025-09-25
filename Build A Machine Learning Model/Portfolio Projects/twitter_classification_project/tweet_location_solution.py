import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

london_tweets = pd.read_json('london.json', lines=True)
new_york_tweets = pd.read_json('new_york.json', lines=True)
paris_tweets = pd.read_json('paris.json', lines=True)
random_tweets = pd.read_json('random_tweets.json', lines=True)

print(london_tweets.columns)
print(london_tweets.head())

new_york_text = new_york_tweets['text'].to_list()
paris_text = paris_tweets['text'].to_list()
london_text = london_tweets['text'].to_list()

all_tweets = new_york_text + paris_text + london_text
# tweets from New York = 0, Paris = 1, London = 2
labels = [0] * len(new_york_text) + [1] * len(paris_text) + [2] * len(london_text)

train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size=.2, random_state=1)
print(len(train_data), len(test_data))

counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)
print(train_data[3])
print(train_counts[3])

classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
print(classifier.score(test_counts, test_labels))
predictions = classifier.predict(test_counts)
print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))

tweet = 'What a great day to be out in Green Park!'  # Specific location in London classified as 2 (London)
tweet = 'What a great day to be outside!'  # Less obvious as no location data included also classified as 2
tweet_vectorized = counter.transform([tweet])
print(classifier.predict(tweet_vectorized))
