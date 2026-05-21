import random
import numpy as np
import os
from typing import Dict, List


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_outputs():
    os.makedirs('outputs', exist_ok=True)


def top_terms_from_centroids(model, vectorizer, n_terms: int = 10) -> Dict[int, List[str]]:
    """Return top `n_terms` per centroid for centroid-based models (KMeans).

    Args:
        model: fitted clustering model with `cluster_centers_` attribute.
        vectorizer: fitted vectorizer exposing `get_feature_names_out()`.
        n_terms: number of top terms to return per cluster.

    Returns:
        Dict mapping cluster index -> list of top term strings.
    """
    if not hasattr(model, 'cluster_centers_'):
        return {}
    centers = model.cluster_centers_
    try:
        terms = vectorizer.get_feature_names_out()
    except Exception:
        # fallback to feature names (older sklearn)
        terms = getattr(vectorizer, 'get_feature_names', lambda: [])()
    order = centers.argsort()[:, ::-1]
    top = {int(i): [terms[int(idx)] for idx in order[i, :n_terms]] for i in range(centers.shape[0])}
    return top
