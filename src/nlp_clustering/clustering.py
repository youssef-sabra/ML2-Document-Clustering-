from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy
from typing import Tuple, List, Dict
import numpy as np


def run_kmeans(X, n_clusters: int = 4, random_state: int = 42, n_init: int = 10) -> Tuple:
    """Fit KMeans and return (model, labels)."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    return model, labels


def kmeans_elbow(X, k_range: List[int], random_state: int = 42) -> Dict[int, float]:
    """Compute inertia (sum of squared distances) for K values to help elbow method.

    Returns a dict mapping k -> inertia.
    """
    out = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        out[k] = float(km.inertia_)
    return out


def kmeans_silhouette_scores(X, k_range: List[int], random_state: int = 42) -> Dict[int, float]:
    """Compute silhouette scores for a range of K (where applicable).

    Scores are NaN for k <= 1.
    """
    out = {}
    arr = X.toarray() if hasattr(X, 'toarray') else X
    for k in k_range:
        if k <= 1 or k >= arr.shape[0]:
            out[k] = float('nan')
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        out[k] = float(silhouette_score(arr, labels))
    return out


def run_agglomerative(X, n_clusters: int = 4, linkage: str = 'ward') -> Tuple:
    """Fit Agglomerative clustering and return (model, labels).

    Note: some linkages require dense arrays.
    """
    arr = X.toarray() if hasattr(X, 'toarray') else X
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(arr)
    return model, labels


def compute_dendrogram_linkage(X, method: str = 'ward', sample_size: int = 200):
    """Compute linkage matrix for dendrogram plotting.

    Returns a linkage matrix suitable for `scipy.cluster.hierarchy.dendrogram`.
    For performance, samples up to `sample_size` rows.
    """
    arr = X.toarray() if hasattr(X, 'toarray') else X
    n = min(sample_size, arr.shape[0])
    return hierarchy.linkage(arr[:n], method=method)


def run_gmm(X, n_components: int = 4, random_state: int = 42) -> Tuple:
    """Fit GaussianMixture and return (model, labels)."""
    arr = X.toarray() if hasattr(X, 'toarray') else X
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(arr)
    return model, labels


def gmm_bic_aic_scores(X, components: List[int], random_state: int = 42) -> Dict[int, Dict[str, float]]:
    """Compute BIC and AIC for different numbers of GMM components.

    Returns dict: n_components -> {'bic':..., 'aic':...}
    """
    arr = X.toarray() if hasattr(X, 'toarray') else X
    out = {}
    for k in components:
        g = GaussianMixture(n_components=k, random_state=random_state)
        g.fit(arr)
        out[k] = {'bic': float(g.bic(arr)), 'aic': float(g.aic(arr))}
    return out


def cluster_sizes(labels) -> Dict[int, int]:
    """Return sizes per cluster id."""
    uniq, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}


def interpret_kmeans_clusters(kmeans_model, vectorizer, n_terms: int = 10) -> Dict[int, List[str]]:
    """Return top `n_terms` for each KMeans cluster using centroid values.

    Wrapper around centroid inspection; returns dict cluster_id -> list of terms.
    """
    from .utils import top_terms_from_centroids
    return top_terms_from_centroids(kmeans_model, vectorizer, n_terms=n_terms)
