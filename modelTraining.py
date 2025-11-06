from sklearn.feature_extraction.text import TfidfVectorizer

import pickle as pkl

from sklearn.linear_model import LogisticRegression

import pandas as pd

# Reading (and cleaning) the CSV file

df = pd.read_csv('products_cleaned.csv')

X = df['Product_Title']

Y = df['Category_Label']

# TF-IDF vectorization
 
vectorizer = TfidfVectorizer()
 
X_tfidf = vectorizer.fit_transform(X)

# Model training

model = LogisticRegression(max_iter=1000)
 
model.fit(X_tfidf, Y)

# with open("MLModel.pkl", "wb") as file:
#     pkl.dump(model, file)

