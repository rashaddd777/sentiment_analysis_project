import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', download_dir='~/nltk_data')
    nltk.download('stopwords', download_dir='~/nltk_data')
    nltk.download('punkt_tab', download_dir='~/nltk_data')
    nltk.data.path.append('~/nltk_data')

stop_words = set(stopwords.words('english'))

def load_data(path, label):
    reviews = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
            reviews.append(f.read())
    return pd.DataFrame({'review': reviews, 'label': label})

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    try:
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in stop_words and word]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""