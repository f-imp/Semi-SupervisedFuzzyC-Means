import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import numpy as np


def plot_embedding(X, y, filepath, c, n, title=None):
    colors = cm.rainbow(np.linspace(0, 1, num=c))
    fig = plt.figure(n, figsize=(6, 6))
    ax = fig.add_subplot(111)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.set_facecolor("white")
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(y[i]),
                color=colors[y[i]],
                fontdict={'weight': 'bold', 'size': 12})
    ax.set_title(title)
    ax.title.set_position([.5, 1.1])
    ax.title.set_fontsize(14)
    fig.tight_layout()
    fig.savefig(filepath, pad_inches=1)
    plt.close()
    return


def visualizing_clustering_result(n, data, seed, title, filepath, n_cluster, idx):
    Z_embedded = TSNE(n_components=2, perplexity=50, n_jobs=4, init='pca', learning_rate=200).fit_transform(
        data[0][idx])
    plot_embedding(Z_embedded, (data[1])[idx], filepath, c=n_cluster, n=n, title=title)
    return
