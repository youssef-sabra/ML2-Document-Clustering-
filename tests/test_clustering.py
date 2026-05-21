from nlp_clustering import features, clustering, utils


def test_kmeans_reproducible():
    texts = ["apple orange banana.", "dog cat mouse.", "car bus train.", "plane rocket."]
    utils.set_seed(0)
    X_raw, X_used, vec, svd = features.vectorize(texts, max_features=100, n_components=None)
    model1, labels1 = clustering.run_kmeans(X_used, n_clusters=2, random_state=0)
    utils.set_seed(0)
    model2, labels2 = clustering.run_kmeans(X_used, n_clusters=2, random_state=0)
    assert len(labels1) == len(texts)
    assert (labels1 == labels2).all()
