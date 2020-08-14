import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

from src.utils import get_data


def features_significant():
    x, _ = get_data()
    full_pca = PCA(n_components=len(x.columns))
    full_pca.fit(x)
    return full_pca.explained_variance_ratio_


def calc_pca():
    x, y = get_data()
    pca = PCA(n_components=3)
    pca.fit(x)
    x_transformed = pca.fit_transform(x)
    pca_df = pd.DataFrame(x_transformed)
    pca_df.columns = ["PC1", "PC2", 'PC3']
    x = pd.concat([x, pca_df], axis=1, join='inner')

    return x, y


def plot_pca():
    x, y = calc_pca()
    ax = sns.scatterplot(x=x['PC1'], y=x['PC2'], size=x['PC3'] * 1.6, hue=x['PC3'] * 1.6, alpha=0.5)
    ax.set(ylim=(-50, 80), xlim=(-110, 210))
    plt.show()

    print('PCA correlation test')
    pearson_correlation(x['PC1'], x['PC2'], x['PC3'], y)
    spearmanr_correlation(x['PC1'], x['PC2'], x['PC3'], y)


def calc_fast_ica():
    x, y = get_data()
    ica = FastICA(n_components=3)
    x_transformed = ica.fit_transform(x)
    x_transformed /= x_transformed.std(axis=0)
    fica_df = pd.DataFrame(x_transformed)
    fica_df.columns = ["PC1", "PC2", 'PC3']
    x = pd.concat([x, fica_df], axis=1, join='inner')

    return x, y


def plot_fast_ica():
    x, y = calc_fast_ica()
    ax = sns.scatterplot(x=x['PC1'], y=x['PC2'], size=x['PC3'] * 1.6, hue=x['PC3'] * 1.75, alpha=0.3)
    ax.set(ylim=(-5, 3), xlim=(-5, 7.5))
    plt.show()

    print('Fast ICA correlation test')
    pearson_correlation(x['PC1'], x['PC2'], x['PC3'], y)
    spearmanr_correlation(x['PC1'], x['PC2'], x['PC3'], y)


def pearson_correlation(pc1, pc2, pc3, quality):
    fig, axes = plt.subplots(3, 1)
    print('pearson for pc1')
    axes[0].grid(True)
    axes[0].set_title('Quality & PC1')
    sns.regplot(pc1, quality, scatter_kws={'s': 3}, line_kws={'color': 'g'}, ax=axes[0])
    print(stats.pearsonr(pc1, quality))
    print('pearson for pc2')
    axes[1].grid(True)
    axes[1].set_title('Quality & PC2')
    sns.regplot(pc2, quality, scatter_kws={'s': 3}, line_kws={'color': 'g'}, ax=axes[1])
    print(stats.pearsonr(pc2, quality))
    print('pearson for pc3')
    axes[2].grid(True)
    axes[2].set_title('Quality & PC3')
    sns.regplot(pc3, quality, scatter_kws={'s': 3}, line_kws={'color': 'g'}, ax=axes[2])
    print(stats.pearsonr(pc3, quality))
    plt.subplots_adjust(top=1.2, hspace=1.5)
    plt.show()


def spearmanr_correlation(pc1, pc2, pc3, quality):
    print('spearman for pc1')
    print(stats.spearmanr(pc1, quality))
    print('spearman for pc2')
    print(stats.spearmanr(pc2, quality))
    print('spearman for pc3')
    print(stats.spearmanr(pc3, quality))


def compare_pca_ica():
    x, y = get_data()
    features_significant_ratios = features_significant()

    ica = FastICA(n_components=3)
    ica_transformed = ica.fit_transform(x)
    ica_diff = ica.inverse_transform(ica_transformed) - x
    ica_results = (features_significant_ratios * ica_diff).sum(1)

    pca = PCA(n_components=3)
    pca.fit(x)
    x_transformed = pca.fit_transform(x)
    pca_diff = pca.inverse_transform(x_transformed) - x
    pca_results = (features_significant_ratios * pca_diff).sum(1)

    n = x.shape[0]
    print('Correlation between PCA results and ICA results: ' + str(stats.spearmanr(pca_results, ica_results)))
    t_static = (np.mean(pca_results) - np.mean(ica_results)) / \
               np.sqrt(np.std(pca_results) ** 2 / n + np.std(ica_results) ** 2 / n)

    p_value = 2 * (stats.t.cdf(t_static, n - 1))
    print('PCA ICA Student\'s t-test p_value result: ' + str(p_value))
