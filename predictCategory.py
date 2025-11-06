from sklearn.feature_extraction.text import TfidfVectorizer

import pickle as pkl

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading the data

df = pd.read_csv('products_cleaned.csv')

X = df['Product_Title']

Y = df['Category_Label']

# TF-IDF vectorization
 
vectorizer = TfidfVectorizer()
 
X_tfidf = vectorizer.fit_transform(X)

# Loading the model

with open("MLModel.pkl", "rb") as file:
    model = pkl.load(file)

y_pred = model.predict(X_tfidf)

print(f"\nLogistic Regression - Classification Report:")
print(classification_report(Y, y_pred),'\n')
print("Accuracy:\n", accuracy_score(Y, y_pred),'\n')
print("Confusion Matrix:\n", confusion_matrix(Y, y_pred))