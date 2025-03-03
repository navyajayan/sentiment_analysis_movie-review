# sentiment_analysis_movie-review

This project provides a Flask-based API for analyzing the sentiment of movie reviews using a trained machine learning model.  

## Features  
- Predicts movie review sentiment (Positive/Negative).  
- Uses TF-IDF vectorization and a trained classifier.  
- Simple REST API with Flask.

## Project Structure

app.py - Flask API for handling predictions
sentiment_model.pkl - Trained sentiment analysis model
tfidf_vectorizer.pkl - TF-IDF vectorizer
requirements.txt - Dependencies 

├── Data  
│   ├── IMDB Dataset.csv  # Movie review dataset  
├── sentiment_model.pkl  # Trained sentiment analysis model  
├── tfidf_vectorizer.pkl  # TF-IDF vectorizer  
├── app.py  # Flask API  
├── requirements.txt  # Required Python packages  
├── notebook.ipynb  # Jupyter Notebook for training & experimentation  
├── README.md  # Project documentation  

