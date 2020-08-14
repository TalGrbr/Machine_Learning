import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture

from src.utils import get_data


def plot_kmeans_pca():
    n_clusters = 3
    x, _ = get_data()
    plot_gap_optimal_k(x)
    distortion_optimal_k()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(x)
    x['clusters'] = kmeans.predict(x)
    pca_model = PCA(n_components=2)
    pcs_2d = pd.DataFrame(pca_model.fit_transform(x.drop(['clusters'], axis=1)))
    pcs_2d.columns = ["PC1", "PC2"]
    x = pd.concat([x, pcs_2d], axis=1, join='inner')
    clusters = [x[x['clusters'] == i] for i in range(5)]
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(n_clusters):
        plt.scatter(x=clusters[i][pcs_2d.columns[0]],
                    y=clusters[i][pcs_2d.columns[1]],
                    c=colors[i])
    plot_silhouette_score(x.drop('clusters', axis=1), x['clusters'])


def plot_silhouette_score(x, cluster_labels):
    silhouette_avg = silhouette_score(x, cluster_labels)
    print("The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(x, cluster_labels)

    y_lower = 10
    n_clusters = len(set(cluster_labels))
    fig, ax1 = plt.subplots()
    colors = ['red', 'green', 'blue', 'yellow', 'black']
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()


def plot_gap_optimal_k(data):
    k, gap_df = gap_optimal_k(data, n_refs=5, max_clusters=15)
    plt.plot(gap_df.clusterCount, gap_df.gap, linewidth=3)
    plt.scatter(gap_df[gap_df.clusterCount == k].clusterCount, gap_df[gap_df.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title('Gap Values by Cluster Count')
    plt.show()
    return k


def gap_optimal_k(data, n_refs=3, max_clusters=15):
    gaps = np.zeros((len(range(1, max_clusters)),))
    results_df = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, max_clusters)):
        ref_disps = np.zeros(n_refs)

        for i in range(n_refs):
            random_reference = np.random.random_sample(size=data.shape)

            km = KMeans(k)
            km.fit(random_reference)

            ref_disp = km.inertia_
            ref_disps[i] = ref_disp

        km = KMeans(k)
        km.fit(data)

        orig_disp = km.inertia_

        gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)

        gaps[gap_index] = gap

        results_df = results_df.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return gaps.argmax() + 1, results_df


def distortion_optimal_k():
    x, _ = get_data()
    distortions = []
    k_options = range(1, 15)
    for k in k_options:
        kmean_model = KMeans(n_clusters=k)
        kmean_model.fit(x)
        distortions.append(sum(np.min(cdist(x, kmean_model.cluster_centers_, 'euclidean'), axis=1)) / x.shape[0])

    plt.plot(k_options, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def plot_gmm():  # Bad results!
    x, _ = get_data()
    cv_types = ['spherical', 'tied', 'diag', 'full']
    bic = []
    lowest_bic = np.infty

    for cv_type in cv_types:
        gmm = GaussianMixture(n_components=5, covariance_type=cv_type)
        gmm.fit(x)
        bic.append(gmm.bic(x))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

    plt.subplot(2, 1, 1)
    plt.bar(range(len(cv_types)), bic)
    plt.xticks(range(4), cv_types)
    plt.xlabel('Covariance Type')
    plt.ylabel('BIC score')
    plt.title('BIC score per model')

    plt.subplot(2, 1, 2)

    best_gmm.fit(x)
    x['clusters'] = best_gmm.predict(x)
    pca_model = PCA(n_components=2)
    pcs_2d = pd.DataFrame(pca_model.fit_transform(x.drop(['clusters'], axis=1)))
    pcs_2d.columns = ["PC1", "PC2"]
    x = pd.concat([x, pcs_2d], axis=1, join='inner')
    clusters = [x[x['clusters'] == i] for i in range(5)]
    colors = ['red', 'green', 'blue', 'yellow', 'black']
    for i in range(5):
        plt.scatter(x=clusters[i][pcs_2d.columns[0]],
                    y=clusters[i][pcs_2d.columns[1]],
                    c=colors[i])
    plot_silhouette_score(x.drop('clusters', axis=1), x['clusters'])
    plt.show()


def plot_dbscan():  # Bad results!
    x, _ = get_data()
    db = DBSCAN(eps=0.001, min_samples=2).fit(x)

    plot_silhouette_score(x, db.labels_)


def plot_hierarchical():
    n_clusters = 3
    x, _ = get_data()
    x['clusters'] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(x)
    pca_model = PCA(n_components=2)
    pcs_2d = pd.DataFrame(pca_model.fit_transform(x.drop(['clusters'], axis=1)))
    pcs_2d.columns = ["PC1", "PC2"]
    x = pd.concat([x, pcs_2d], axis=1, join='inner')
    clusters = [x[x['clusters'] == i] for i in range(5)]
    colors = ['red', 'green', 'blue']
    for i in range(n_clusters):
        plt.scatter(x=clusters[i][pcs_2d.columns[0]],
                    y=clusters[i][pcs_2d.columns[1]],
                    c=colors[i])
    plot_silhouette_score(x.drop('clusters', axis=1), x['clusters'])
