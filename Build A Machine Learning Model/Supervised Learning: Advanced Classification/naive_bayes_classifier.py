from reviews import neg_list, pos_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# can't run as pickle files are invalid for some reason....

# ---------- Formatting the data for Scikit-Learn ----------

review = "This crib was amazing"
counter = CountVectorizer()
counter.fit(neg_list + pos_list)
print(counter.vocabulary_)

review_counts = counter.transform([review])
print(review_counts.toarray())

training_counts = counter.transform(neg_list + pos_list)

# ---------- Using Scikit-Learn ----------

review = "This crib was fantastic, unbelievably useful!"
review_counts = counter.transform([review])

classifier = MultinomialNB()
training_labels = [0 if i <= 999 else 1 for i in range(2000)]

classifier.fit(training_counts, training_labels)
print(classifier.predict(review_counts))
print(classifier.predict_proba(review_counts))