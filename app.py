# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:52:13 2023

@author: acer
"""

import pickle
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the vectorizer and similarity scores
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    similarity_scores = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text1 = request.form['text1']
    text2 = request.form['text2']

    # Preprocess the new text inputs (if required)

    # Feature extraction
    text1_feature_new = vectorizer.transform([text1])
    text2_feature_new = vectorizer.transform([text2])

    # Calculate similarity
    similarity_score_new = cosine_similarity(text1_feature_new, text2_feature_new)[0, 0]

    # Convert similarity score to a value between 0 and 1
    normalized_similarity_score_new = (similarity_score_new + 1) / 2

    # Return the similarity score as JSON response
    return jsonify({'similarity_score': normalized_similarity_score_new})

if __name__ == '__main__':
    app.run(debug=True)
