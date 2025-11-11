# 🧠 Product Category Classifier using Machine Learning  

### 📋 Project Overview  
This project trains a **machine learning model** that can automatically predict the **category of a product** based on its **title or description**.  

We’ll use a dataset of products, clean it up, turn the text into numbers with **TF-IDF**, train a **Logistic Regression** model, and finally allow the user to enter their own product names to get instant predictions!  

---

## 🧩 Step 1: Importing the Required Libraries
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
import pandas as pd
from sklearn.linear_model import LogisticRegression
```

### 💡 Explanation  
In this step, we import all the tools we’ll need:
- `pandas` → for working with tables and CSV files  
- `TfidfVectorizer` → converts text into numeric form  
- `LogisticRegression` → our machine learning model  
- `pickle` → used to save and load our trained model later  

📝 *Think of this step as setting up your toolbox before starting the project.*

---

## 🧹 Step 2: Loading and Cleaning the Data
```python
# Loading data
df = pd.read_csv('products.csv')

# Dropping missing values
df.dropna(axis=0, how='any', inplace=True)

# Standardizing data
df["Category_Label"] = df['Category_Label'].str.lower()

# Checking the unique amount of categories
print("Number of unique categories:", df["Category_Label"].nunique())

# Sampling the dataframe
print("\n", df.sample(5))
```

### 💡 Explanation  
Here we:
1. **Load** our dataset from a CSV file.  
2. **Remove any empty rows** (this helps keep the data clean).  
3. **Make all category names lowercase** so similar labels (like “Shoes” vs “shoes”) are treated the same.  
4. **Check how many categories** we have and print a few sample rows to understand what the data looks like.  

📝 *Clean data means better learning — just like a clean desk helps you work better.*

---

## ⚙️ Step 3: Turning Text into Numbers and Training the Model
```python
# Separating the data into products and categories
X = df["Product_Title"]
Y = df["Category_Label"]

# Vectorizing data
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Initiating the model
model = LogisticRegression(max_iter=1000)

# Training the model
model.fit(X_tfidf, Y)
```

### 💡 Explanation  
1. `X` = product names (our input text)  
   `Y` = product categories (our labels)  
2. `TfidfVectorizer` turns each product title into a list of numbers representing how important each word is.  
3. `LogisticRegression` is our classifier — it learns the connection between words and product categories.  
4. We train the model using the cleaned data by calling `model.fit(X_tfidf, Y)`.  

📝 *At this point, the model has “learned” how to tell categories apart based on the words in product titles.*

---

## 💾 Step 4: Saving and Loading the Model
```python
# Dumping the model into a pickle file to open in another script
with open("MLModel.pkl", "wb") as file:
    pkl.dump(model, file)

# Loading the model
with open("MLModel.pkl", "rb") as file:
    model = pkl.load(file)
```

### 💡 Explanation  
We **save** the trained model so we can reuse it later without training again.  
Then we **reload** it from the saved file to test that saving worked correctly.  

📝 *Saving the model is like saving a video game — you can continue where you left off without replaying everything.*

---

## 💬 Step 5: Testing the Model with User Input
```python
# Loop for user input and printing the model prediction
while True:
    user_input = input("Enter a product: ")

    if user_input.lower() == 'exit':
        print("Exiting category classifier.")
        break

    user_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_tfidf)[0]

    print(f"\nPredicted Category: {prediction}")
```

### 💡 Explanation  
Here’s where the fun begins! 🎉  
We let the user **type a product name**, such as `wireless headphones`, and the model instantly predicts a category like `electronics`.  

Type `exit` to stop the program.  

📝 *This is how you can turn your machine learning model into a real interactive tool!*

---

## 🧠 Summary  

✅ Loaded and cleaned product data  
✅ Turned text into numbers using **TF-IDF**  
✅ Trained a **Logistic Regression** model  
✅ Saved the model for future use  
✅ Built a simple interactive prediction tool  

---

## 🚀 Next Steps (for improvement)
Here’s how you can make this project even better:
- **Add accuracy testing** → use `train_test_split` and `accuracy_score` to measure how well your model performs.  
- **Save the vectorizer** too → so you can reuse it later for predictions.  
- **Use a web app** (like Flask or Streamlit) for a user-friendly interface.  
- **Tune parameters** → experiment with `ngram_range`, `max_features`, or `class_weight`.  
- **Visualize results** → show which words are most important for each category.
