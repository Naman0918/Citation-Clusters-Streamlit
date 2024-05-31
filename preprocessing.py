import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

stop_words = ['the', 'of', 'and', 'to', 'in', 'a', 'for', 'is', 'The', 'with', 'are']
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words + list(stopwords.words('english'))]
    text = re.sub(r'[^\w\s]', '', ' '.join(tokens))
    return text
