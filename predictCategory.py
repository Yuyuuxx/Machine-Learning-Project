from sklearn.feature_extraction.text import TfidfVectorizer

import pickle as pkl

import pandas as pd

from sklearn.linear_model import LogisticRegression

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

while True:
 
    user_input = input("Enter a product: ")
    
    user_tfidf = vectorizer.transform([user_input])

    prediction = model.predict(user_tfidf)[0]

    print(f"\nPredicted Category: {prediction}")
    
    if user_input.lower() == 'exit':
 
        print("Exiting category classifier.")
 
        break


