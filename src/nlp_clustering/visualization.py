import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import seaborn as sns
from scipy.cluster import hierarchy
from wordcloud import WordCloud


def _ensure_dir(path):
    # Accept either a file path (e.g., outputs/plot.png) or a directory path (e.g., outputs/figs)
    if not path:
        return
    # If path looks like a file (has an extension), use its directory; otherwise treat as directory
    base, ext = os.path.splitext(path)
    if ext:
        dirpath = os.path.dirname(path)
    else:
        dirpath = path
    if not dirpath:
        return
    os.makedirs(dirpath, exist_ok=True)


def plot_pca(X, labels, out_path='outputs/pca.png'):
    arr = X.toarray() if hasattr(X, 'toarray') else X
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(arr)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, palette='tab10', s=30, legend='full')
    plt.title('PCA projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tsne(X, labels, out_path='outputs/tsne.png'):
    arr = X.toarray() if hasattr(X, 'toarray') else X
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    Z = tsne.fit_transform(arr)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, palette='tab10', s=30, legend='full')
    plt.title('t-SNE projection')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dendrogram(X, out_path='outputs/dendrogram.png', truncate_mode='level', p=5):
    arr = X.toarray() if hasattr(X, 'toarray') else X
    # For speed, limit input size
    n = min(200, arr.shape[0])
    linked = hierarchy.linkage(arr[:n], method='ward')
    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(linked, truncate_mode=truncate_mode, p=p)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index (or cluster size)')
    plt.ylabel('Distance')
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cluster_scatter(X, labels, out_path='outputs/cluster_scatter.png', method: str = 'pca'):
    """Plot a 2D scatter of clusters using PCA or t-SNE for dimensionality reduction.

    Args:
        X: document-term matrix (sparse or dense)
        labels: cluster labels
        out_path: where to save
        method: 'pca' or 'tsne'
    """
    arr = X.toarray() if hasattr(X, 'toarray') else X
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        Z = reducer.fit_transform(arr)
        xlabel, ylabel = 'PC1', 'PC2'
    else:
        reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        Z = reducer.fit_transform(arr)
        xlabel, ylabel = 'Dim1', 'Dim2'

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, palette='tab10', s=40, legend='full')
    plt.title(f'Cluster scatter ({method})')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_wordclouds_for_clusters(kmeans, vectorizer, out_dir='outputs/wordclouds', n_terms: int = 100, width: int = 600, height: int = 300):
    """Generate a wordcloud image per cluster from KMeans centroids/vocabulary.

    Saves files to `out_dir/cluster_{i}.png` and returns list of paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not hasattr(kmeans, 'cluster_centers_'):
        raise ValueError('kmeans model must have cluster_centers_')
    centers = kmeans.cluster_centers_
    terms = vectorizer.get_feature_names_out()
    paths = []
    for i, center in enumerate(centers):
        order = center.argsort()[::-1]
        top_idx = order[:n_terms]
        freqs = {terms[j]: float(center[j]) for j in top_idx}
        wc = WordCloud(width=width, height=height, background_color='white')
        img = wc.generate_from_frequencies(freqs)
        out_path = os.path.join(out_dir, f'cluster_{i}_wordcloud.png')
        img.to_file(out_path)
        paths.append(out_path)
    return paths
