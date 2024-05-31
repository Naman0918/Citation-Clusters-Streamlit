from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Initialize the vectorizer globally
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

def preprocess_data(data, preprocess_text_func):
    data['Abstracts_processed'] = data['Abstract'].apply(preprocess_text_func)
    data['Title_processed'] = data['Title'].apply(preprocess_text_func)
    return data

def vectorize_text(data, fit=True):
    if fit:
        X = vectorizer.fit_transform(data['Abstracts_processed'])
    else:
        X = vectorizer.transform(data['Abstracts_processed'])
    return X

def perform_clustering(X, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

def reduce_dimensions(X):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    return X_pca

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
