import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from nlp_clustering import preprocessing, features, clustering, evaluation, visualization, utils, eda_utils
import joblib


def main():
    utils.set_seed(42)
    project_root = Path(__file__).resolve().parent
    outputs_dir = project_root / 'outputs'
    models_dir = outputs_dir / 'models'
    results_dir = outputs_dir / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    texts, labels, df = eda_utils.load_20newsgroups_dataset(
        categories=eda_utils.DEFAULT_20NEWSGROUPS_CATEGORIES,
        max_docs_per_category=250,
    )

    print(f"Documents: {len(texts)}")
    print(f"Categories: {sorted(set(labels))}")
    df.to_csv(outputs_dir / '20newsgroups_sample.csv', index=False)
    print("Preprocessing texts...")
    texts_clean = preprocessing.preprocess_texts(texts)

    print("Vectorizing (TF-IDF)...")
    X_raw, X_reduced, X_cluster, vectorizer, svd, normalizer = features.vectorize_for_clustering(
        texts_clean,
        max_features=10000,
        n_components=100,
        min_df=2,
    )

    print("Running KMeans (k=4)...")
    kmodel, klabels = clustering.run_kmeans(X_cluster, n_clusters=4)
    sil = evaluation.silhouette(X_cluster, klabels)

    print(f"KMeans silhouette score: {sil:.4f}")
    cluster_sizes = clustering.cluster_sizes(klabels)
    for cid in sorted(cluster_sizes):
        print(f"Cluster {cid} size: {cluster_sizes[cid]}")

    tops = evaluation.top_terms_from_labels(X_raw, klabels, vectorizer, n_terms=10)
    for cid, terms in tops.items():
        print(f"Cluster {cid}: {', '.join(terms)}")

    metrics_payload = evaluation.compute_all_metrics(X_cluster, klabels)
    metrics_payload['n_documents'] = len(texts)
    metrics_payload['n_clusters'] = 4
    with open(results_dir / 'kmeans_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)

    with open(results_dir / 'kmeans_top_terms.json', 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in tops.items()}, f, indent=2)

    joblib.dump(kmodel, models_dir / 'kmeans_model.joblib')
    joblib.dump(vectorizer, models_dir / 'tfidf_vectorizer.joblib')

    print("Saving visualizations to outputs/ ...")
    visualization.plot_pca(X_cluster, klabels, out_path='outputs/pca_kmeans.png')
    visualization.plot_tsne(X_cluster, klabels, out_path='outputs/tsne_kmeans.png')
    visualization.plot_dendrogram(X_cluster, out_path='outputs/dendrogram.png')

    print("Done. Check the outputs/ folder for plots.")


if __name__ == '__main__':
    main()
