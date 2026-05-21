"""Evaluation metrics and utilities for clustering.

Provides silhouette, Davies-Bouldin, and Calinski-Harabasz scores, plus helpers
to compute comparison tables and plots across different clusterings or values
of K (for KMeans).
"""

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _to_dense(X):
    return X.toarray() if hasattr(X, 'toarray') else X


def silhouette(X, labels):
    """Compute silhouette score for clustering labels.

    Returns NaN if score cannot be computed (e.g., single cluster).
    """
    if len(set(labels)) <= 1:
        return float('nan')
    arr = _to_dense(X)
    return float(silhouette_score(arr, labels))


def davies_bouldin(X, labels):
    """Compute Davies-Bouldin score (lower is better)."""
    if len(set(labels)) <= 1:
        return float('nan')
    arr = _to_dense(X)
    return float(davies_bouldin_score(arr, labels))


def calinski_harabasz(X, labels):
    """Compute Calinski-Harabasz score (higher is better)."""
    if len(set(labels)) <= 1:
        return float('nan')
    arr = _to_dense(X)
    return float(calinski_harabasz_score(arr, labels))


def compute_all_metrics(X, labels) -> Dict[str, float]:
    """Return a dictionary with the three clustering metrics."""
    return {
        'silhouette': silhouette(X, labels),
        'davies_bouldin': davies_bouldin(X, labels),
        'calinski_harabasz': calinski_harabasz(X, labels)
    }


def metrics_table_from_labelings(X, labelings: Dict[str, List[int]]) -> pd.DataFrame:
    """Given a dict name->labels, return a DataFrame of metrics per labeling."""
    rows = []
    for name, labels in labelings.items():
        m = compute_all_metrics(X, labels)
        m['name'] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index('name')
    return df


def evaluate_kmeans_range(X, k_range: List[int], random_state: int = 42) -> pd.DataFrame:
    """Run KMeans for each k and return a DataFrame with inertia and metrics.

    Columns: inertia, silhouette, davies_bouldin, calinski_harabasz
    """
    from .clustering import run_kmeans

    rows = []
    arr = _to_dense(X)
    for k in k_range:
        km, labels = run_kmeans(X, n_clusters=k, random_state=random_state)
        inertia = float(getattr(km, 'inertia_', float('nan')))
        sil = silhouette(X, labels)
        db = davies_bouldin(X, labels)
        ch = calinski_harabasz(X, labels)
        rows.append({'k': k, 'inertia': inertia, 'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch})
    df = pd.DataFrame(rows).set_index('k')
    return df


def plot_metrics(df: pd.DataFrame, metrics: List[str] = None, out_path: str = None):
    """Plot metric curves (index should be numeric like k).

    If `out_path` is provided, saves the figure.
    """
    sns.set(style='whitegrid')
    if metrics is None:
        metrics = [c for c in df.columns]
    plt.figure(figsize=(8, 5))
    for m in metrics:
        if m in df.columns:
            plt.plot(df.index, df[m], marker='o', label=m)
    plt.xlabel('k')
    plt.legend()
    plt.title('Clustering metrics vs k')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    plt.show()


def plot_comparison_table(df: pd.DataFrame, out_path: str = None):
    """Show a heatmap-like comparison of metrics for different labelings."""
    sns.set(style='white')
    plt.figure(figsize=(8, max(2, len(df) * 0.5)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='vlag', center=0)
    plt.title('Clustering metrics comparison')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    plt.show()


def top_terms_per_cluster(kmeans, vectorizer, n_terms: int = 10) -> Dict[int, list]:
    """Return top terms per cluster for centroid-based models like KMeans."""
    from .utils import top_terms_from_centroids
    return top_terms_from_centroids(kmeans, vectorizer, n_terms=n_terms)


def top_terms_from_labels(X_raw, labels, vectorizer, n_terms: int = 10) -> Dict[int, list]:
    """Return the most important TF-IDF terms for each cluster label.

    This averages raw TF-IDF scores inside each cluster, which is a simple way to
    describe clusters even when the model was trained on reduced features.
    """
    terms = vectorizer.get_feature_names_out()
    label_array = np.asarray(labels)
    out: Dict[int, list] = {}

    for label in sorted(set(label_array.tolist())):
        mask = label_array == label
        cluster_matrix = X_raw[mask]
        if cluster_matrix.shape[0] == 0:
            out[int(label)] = []
            continue

        scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_idx = scores.argsort()[::-1][:n_terms]
        out[int(label)] = [terms[idx] for idx in top_idx if scores[idx] > 0]
    return out

