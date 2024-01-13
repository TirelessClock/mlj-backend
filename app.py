from flask import Flask, request, jsonify
from flask_cors import CORS

import pickle
import time
import requests
import numpy as np
import spacy 
import nltk 
nltk.download('stopwords')

import string

from nltk.corpus import stopwords
from sklearn.neighbors import BallTree
from google.cloud import storage

app = Flask(__name__)
CORS(app)


def load_data_from_gcs():
    print("Loading data started...")
    client = storage.Client()
    bucket = client.bucket('tipsy-tickle-monster')
    blob = bucket.blob('data.pickle')
    data = pickle.loads(blob.download_as_bytes())
    print("Data loaded!")
    return data


def initialize():
    data = load_data_from_gcs()
    labels = data["labels"]
    tree = BallTree(data["vectors"])

    print('Initialization complete')
    return tree, labels


tree, labels = initialize()
nlp = spacy.load("en_core_web_md")
stop_words = set(stopwords.words('english'))

cached_lemmas = {}

def applyWordEmbedding(text):

    text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))

    text_list = text.split()
    total_weight = 0
    total_vector = np.zeros(300)

    for word in text_list:
        if word not in stop_words:
            doc = nlp(word)
            word = doc[0]
            if word not in cached_lemmas:
                lemma = word.lemma_
                cached_lemmas[word] = lemma
            else:
                lemma = cached_lemmas[word]

            lemma_doc = nlp(lemma)
            total_vector += lemma_doc.vector
            total_weight += 1

    if total_weight > 0:
        result = total_vector / total_weight
    else:
        result = np.zeros(300)

    return result.tolist()

def downloadimage(movie, year):
    api_key = '4cd79b30'
    api_url = f'http://omdbapi.com/?apikey={api_key}&t={movie}&y={year}'

    try:
        response = requests.get(api_url)
        poster_data = response.json()

        # Check if the request was successful
        if response.status_code == 200 and poster_data.get("Poster"):
            return poster_data["Poster"]
        else:
            print(f"Error: {poster_data.get('Error', 'Unknown error')}")
            return None

    except Exception as e:
        print(e)
        return None
    
def model(query):
    new_point = applyWordEmbedding(query)

    k = 5
    _, indices = tree.query([new_point], k)

    nearest_labels = [labels[i] for i in indices[0]]

    s = time.time()
    for item in nearest_labels:
        item["Image"] = downloadimage(f"{item['Title']}", f"{item['Year']}")
    
    e = time.time()
    print(f"Time taken: {e-s}")
    return nearest_labels

@app.route('/search', methods=['POST'])
def search():
    print("Search function called!")
    received_data = request.get_json()
    query = received_data["query"]
    print("Query: ", query)
    result = model(query)
    return jsonify(result=result)

if __name__ == '__main__' and not app.debug:
    app.run(host='0.0.0.0', port=8080)
    