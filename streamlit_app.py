import streamlit as st
import pandas as pd
from preprocessing import preprocess_text
from clustering import load_data, preprocess_data, vectorize_text, perform_clustering, reduce_dimensions, vectorizer, accuracy  
import matplotlib.pyplot as plt

# Title of the Streamlit app
st.markdown("""
    <style>
    .title {
        color: cyan;  /* Change this to your desired color */
        font-size: 48px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: #ff7f0e;  /* Change this to your desired color */
        font-size: 24px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title of the Streamlit app
st.markdown('<div class="title">Cluster Analysis of Research Papers using Text Mining</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">This application categorizes research papers into clusters based on their abstract.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:

        data = load_data(uploaded_file)
        st.success("File successfully uploaded!")
        st.dataframe(data)

        # Preprocess and vectorize text
        data = preprocess_data(data, preprocess_text)
        X = vectorize_text(data, fit=True)
        kmeans = perform_clustering(X)
        data['Cluster'] = kmeans.labels_

        X_pca = reduce_dimensions(X)

        st.write("-----------------------------------------")
        plt.figure(figsize=(10, 6))
        for cluster in range(kmeans.n_clusters):
            plt.scatter(X_pca[data['Cluster'] == cluster, 0], X_pca[data['Cluster'] == cluster, 1], label=f'Cluster {cluster}')
        plt.legend()
        plt.title('Clusters Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        st.pyplot(plt)
        st.write("-----------------------------------------")


    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload an Excel file to display its contents.")

title = st.text_input("Enter abstract of research paper", "")

st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="center">', unsafe_allow_html=True)
with st.form("myform"):
    submitted  = st.form_submit_button("Submit")

if submitted:
    if title:
        # Preprocess and vectorize the input abstract
        processed_title = preprocess_text(title)
        vectorized_title = vectorizer.transform([processed_title])

        cluster = kmeans.predict(vectorized_title)[0]

        st.write("The abstract of the research paper is:")
        st.write("-----------------------------------------")
        st.write(title)
        st.write("-----------------------------------------")
        st.write(f"This abstract belongs to cluster: {cluster}")
        st.write("Accuracy of the cluster is :", accuracy(data['Cluster'], kmeans.labels_))
    else:
        st.error("Please enter an abstract to get the cluster.")

st.markdown('</div>', unsafe_allow_html=True)

