import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk import sent_tokenize
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from utils.wordembedding import generate_word_embeddings
from utils.text_preprocess import LemmaTokenizer
from utils.transformers import SentenceEmbeddingTransformer

# Load your machine learning model
pipeline_model = joblib.load('../best_model.pkl')

word2vec_model_file = '../models/word2vec_200.model'
model = Word2Vec.load(word2vec_model_file)

# Create the main window
root = tk.Tk()
root.title("Fake reviews detector")
#root.geometry("490x400")
default_font = ('Helvetica', 12)

# Create an entry field for custom data
custom_data_label = tk.Label(root, text="Enter Review Text:", font=default_font)
custom_data_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

custom_data_entry = tk.Text(root, height=10, width=50, font=default_font)  # Set height for multiple lines
custom_data_entry.grid(row=1, column=0, padx=10, pady=5)

verified_purchase_var = tk.IntVar(value=1)  # Default is 1 (Verified)

# Create a Checkbutton to toggle Verified Purchase
verified_purchase_checkbutton = tk.Checkbutton(root, text="Verified Purchase", font=default_font, variable=verified_purchase_var)
verified_purchase_checkbutton.grid(row=2, column=0, padx=10, pady=5)

# Label to display prediction results
prediction_result_label = tk.Label(root, text="", font=default_font, fg="blue")
prediction_result_label.grid(row=4, column=0, padx=10, pady=10)

# Function to predict custom data
def predict_custom_data():
    try:
        # Get custom data from entry field
        text = custom_data_entry.get("1.0", tk.END).strip()
        verified_purchase = verified_purchase_var.get()

        df = pd.DataFrame({
            'REVIEW_TEXT': [text],
            'VERIFIED_PURCHASE': [verified_purchase],
        })
        df['Sentence_Embeddings'] = df['REVIEW_TEXT'].apply(lambda x: generate_word_embeddings(x, model))
        predictions = pipeline_model.predict(df)
        prediction = 'Fake' if predictions[0] == '__label1__' else 'Real'
        prediction_result_label.config(text=f"Prediction: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create a button to predict custom data
custom_predict_button = tk.Button(root, text="Predict Review", font=default_font, command=predict_custom_data, bg="green", fg="white", width=20)
custom_predict_button.grid(row=3, column=0, padx=10, pady=10)

# Run the application
root.mainloop()
