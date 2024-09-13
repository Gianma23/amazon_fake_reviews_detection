import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from nltk import sent_tokenize
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from utils.text_preprocess import LemmaTokenizer
from SentenceEmbedding import SentenceEmbeddingTransformer

# Load your machine learning model
model = joblib.load('../best_model.pkl')

# Create the main window
root = tk.Tk()
root.title("Intrusion Detection System")

# Create an entry field for custom data
custom_data_label = tk.Label(root, text="Review to predict")
custom_data_label.grid(row=7, column=0)
custom_data_entry = tk.Entry(root)
custom_data_entry.grid(row=7, column=1)

def average_word_embeddings(sentence, model):
    words = LemmaTokenizer()(sentence)
    word_vectors = [model.wv[word] for word in words if word in model.wv]

    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(word_vectors, axis=0)


# Define a function to process each review
def process_review(review_text, model):
    sentences = sent_tokenize(review_text)
    sentence_embeddings = [average_word_embeddings(sentence, model) for sentence in sentences]
    return sentence_embeddings

# Function to predict custom data
def predict_custom_data():
    try:
        # Get custom data from entry field
        text = custom_data_entry.get()
        # Process the custom data
        processed_data = TfidfVectorizer(tokenizer=LemmaTokenizer(), token_pattern=None).transform([text])
        print(processed_data)

        # Make prediction
        prediction = model.predict([processed_data])
        # Display the result
        messagebox.showinfo("Prediction Result", f"Prediction: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create a button to predict custom data
custom_predict_button = tk.Button(root, text="Predict Custom Data", command=predict_custom_data)
custom_predict_button.grid(row=8, columnspan=2)

# Run the application
root.mainloop()
