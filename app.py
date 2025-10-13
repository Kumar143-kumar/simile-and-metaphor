import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress TensorFlow warnings/messages
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration and Initialization ---

# An object of the Flask class is our WSGI application.
app = Flask(__name__, template_folder='templates')

# --- Global Model and Vectorizer Variables ---
model = None
tfidf_vectorizer = None
# Target class names (must match LabelEncoder order from training)
TARGET_NAMES = ['Metaphor', 'Simile'] 
# Input dimension must match the max_features used in training
MAX_FEATURES = 5000 

import joblib

def load_model_and_vectorizer():
    """Loads the trained Keras model and TF-IDF vectorizer"""
    global model, tfidf_vectorizer, MAX_FEATURES
    model_path = 'simile_metaphor_classifier_ann.keras'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    try:
        model = tf.keras.models.load_model(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)

        # Set MAX_FEATURES dynamically to match the real vocabulary
        MAX_FEATURES = len(tfidf_vectorizer.vocabulary_)
        print(f"✅ Loaded model and vectorizer successfully. Feature size = {MAX_FEATURES}")

    except Exception as e:
        print(f"❌ Error loading model/vectorizer: {e}")
        model = None
        tfidf_vectorizer = None


# Load the model upon application start
load_model_and_vectorizer()

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the Home page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Renders the About page."""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Renders the Contact page."""
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles GET (showing the form) and POST (processing the prediction).
    """
    if request.method == 'POST':
        sentence = request.form.get('sentence')
        
        if not sentence:
            return render_template('predict.html', error="Please enter a sentence to classify.")
        
        if model is None or tfidf_vectorizer is None:
            return render_template('predict.html', error="Model not loaded. Check server logs.")

        try:
            # 1. Vectorize the input text using the loaded/initialized vectorizer
            X_new = tfidf_vectorizer.transform([sentence]).toarray()
            
            # Ensure the input shape is correct (MAX_FEATURES)
            if X_new.shape[1] != MAX_FEATURES:
                print(f"Debug: Model expects {MAX_FEATURES}, got {X_new.shape[1]}")
                return render_template('predict.html', error=f"Feature dimension mismatch. Expected {MAX_FEATURES}, got {X_new.shape[1]}.")

            # 2. Predict probability (output is a single probability for the positive class, Simile)
            probability = model.predict(X_new)[0][0]
            
            # 3. Determine the prediction and probabilities
            if probability >= 0.5:
                prediction_class = TARGET_NAMES[1]  # Simile (Index 1)
                prob_simile = probability
                prob_metaphor = 1 - probability
            else:
                prediction_class = TARGET_NAMES[0]  # Metaphor (Index 0)
                prob_metaphor = 1 - probability
                prob_simile = probability

            # Format probabilities as percentages
            prob_simile_pct = f"{prob_simile * 100:.2f}"
            prob_metaphor_pct = f"{prob_metaphor * 100:.2f}"

            # 4. Render the result page
            return render_template(
                'result.html',
                sentence=sentence,
                prediction=prediction_class,
                prob_simile=prob_simile_pct,
                prob_metaphor=prob_metaphor_pct
            )

        except Exception as e:
            print(f"Prediction Error: {e}")
            return render_template('predict.html', error=f"An error occurred during prediction: {e}")

    # GET request: show the prediction form
    return render_template('predict.html')


# Example of how to run the app:
if __name__ == '__main__':
    # Flask runs on port 5000 by default
    print("Flask app running at http://127.0.0.1:5000/")
    # If you are running this in a cloud environment, use host='0.0.0.0'
    app.run(debug=True, host='0.0.0.0', port=5000)
