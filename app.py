from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

app = Flask(__name__)

# Load the model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    print("Model loaded successfully!") #added line

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
    print("Vector loaded successfully!") #added line

# Text cleaning function (same as before)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    review_text = data['text']
    cleaned_review = clean_text(review_text)
    review_vector = vectorizer.transform([cleaned_review])
    print(f"review_vector shape: {review_vector.shape}") #added line
    prediction = model.predict(review_vector)[0]

    if prediction == 0:
        sentiment = 'negative'
    else:
        sentiment = 'positive'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)