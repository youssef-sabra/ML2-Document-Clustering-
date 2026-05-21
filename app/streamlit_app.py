"""Minimal Streamlit app for interactive clustering exploration."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
# ensure src is on path
sys.path.insert(0, str(root / 'src'))

import streamlit as st
from nlp_clustering import eda_utils, preprocessing, features, clustering, evaluation, utils
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title='NLP Clustering', layout='wide')

st.sidebar.title('Settings')
dataset_name = st.sidebar.selectbox(
    'Dataset',
    ['20 Newsgroups', 'People Wikipedia'],
    index=0,
)
algorithm = st.sidebar.selectbox('Algorithm', ['KMeans', 'Agglomerative', 'GMM'])
n_clusters = st.sidebar.slider('Number of clusters', 2, 10, 4)
run_btn = st.sidebar.button('Run clustering')

st.title('NLP Clustering')

if dataset_name == '20 Newsgroups':
    texts, titles, df = eda_utils.load_20newsgroups_dataset(
        categories=eda_utils.DEFAULT_20NEWSGROUPS_CATEGORIES,
        max_docs_per_category=250,
    )
else:
    texts, titles, df = eda_utils.load_dataset()

st.write(f'Documents: {len(texts)}')
st.write(f'Dataset: {dataset_name}')
st.write(f'Categories: {sorted(set(titles))[:10]}')

if run_btn:
    with st.spinner('Preprocessing...'):
        texts_clean = preprocessing.preprocess_texts(texts)
    with st.spinner('Vectorizing...'):
        X_raw, X_reduced, X_cluster, vectorizer, svd, normalizer = features.vectorize_for_clustering(
            texts_clean,
            max_features=5000,
            n_components=100,
            min_df=2,
        )
    with st.spinner('Clustering...'):
        if algorithm == 'KMeans':
            model, labels = clustering.run_kmeans(X_cluster, n_clusters=n_clusters)
        elif algorithm == 'Agglomerative':
            model, labels = clustering.run_agglomerative(X_cluster, n_clusters=n_clusters)
        else:
            model, labels = clustering.run_gmm(X_cluster, n_components=n_clusters)
    sil = evaluation.silhouette(X_cluster, labels)
    st.metric('Silhouette score', f'{sil:.4f}')

    st.subheader('Cluster sizes')
    sizes = clustering.cluster_sizes(labels)
    st.table(pd.DataFrame.from_dict(sizes, orient='index', columns=['size']))

    st.subheader('Top terms per cluster')
    try:
        tops = evaluation.top_terms_from_labels(X_raw, labels, vectorizer, n_terms=10)
    except Exception:
        tops = {}
    for cid, terms in tops.items():
        st.write(f'Cluster {cid}: ' + ', '.join(terms))

    st.subheader('PCA visualization')
    arr = X_cluster.toarray() if hasattr(X_cluster, 'toarray') else X_cluster
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(arr)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=30)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA projection')
    st.pyplot(fig)
