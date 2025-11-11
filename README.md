# 🧠 Product Category Classifier using Machine Learning

## 📋 Project Overview

This project uses machine learning to automatically predict a product’s category based on its title or description.  

It takes a dataset of products, cleans the data, converts text into numbers using TF-IDF, and trains a Logistic Regression model to learn how different product titles relate to their categories.  

Once trained, the model can instantly predict a category for any new product name entered by the user.

## ⚙️ How It Works

- Loads and cleans a dataset of product titles and their categories  
- Converts the product text into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency)  
- Trains a Logistic Regression model to recognize patterns between words and categories  
- Saves the trained model for future use  
- Lets the user type in a product name and predicts its category in real time

## 💡 How to Use

1. Make sure you have Python installed on your system.  
2. Install the required libraries by typing `pip install pandas scikit-learn` in your terminal.  
3. Place your dataset file (CSV or Excel) in the project folder.  
4. Run the Python script to train the model.  
5. After training, enter any product name when prompted to get its predicted category.  
6. Type `exit` to close the program.

## 💬 Example

You can type something like *wireless headphones*, and the program might predict the category **electronics**.  
Type *exit* to close the program.

## 🚀 Key Features

- Simple and beginner-friendly machine learning workflow  
- Interactive text-based category prediction  
- Reusable saved model for future predictions  
- Uses popular Python libraries like pandas and scikit-learn

## 🔮 Possible Improvements

- Add model accuracy testing and validation  
- Save both the model and vectorizer for deployment  
- Build a web interface using Flask or Streamlit  
- Experiment with other algorithms or text preprocessing methods
