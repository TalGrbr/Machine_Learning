from src import learning
from src import clustering
from src import dimensionality_reduction


def main():
    print('---------------------------     MACHINE LEARNING       ---------------------------')
    learning.compare_svm_rfc()
    print('---------------------------        CLUSTERING          ---------------------------')
    clustering.plot_kmeans_pca()
    clustering.plot_hierarchical()
    print('---------------------------  DIMENSIONALITY REDUCTION  ---------------------------')
    dimensionality_reduction.plot_pca()
    dimensionality_reduction.plot_fast_ica()
    dimensionality_reduction.compare_pca_ica()


if __name__ == '__main__':
    main()
