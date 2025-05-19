import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Return AgglomerativeClustering with initialize
def agglomerative_clustering():
    agglo_clustering = AgglomerativeClustering(
        n_clusters=None,  # Automatically determine the number of clusters
        distance_threshold=0.7,  # Set similarity threshold (1 - threshold = distance)
        metric="precomputed",  # Using a pre-calculated similarity matrix
        linkage="complete"  # Calculate the distances between clusters: complete Linkage
    )
    return agglo_clustering

if __name__ == "__main__":
    clustering = agglomerative_clustering()
    print(clustering)