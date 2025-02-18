import matplotlib.pyplot as plt

import algorithm
import show_clustering as sc
from vector import load_from_json
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np


#from scipy.cluster.hierarchy import dendrogram, linkage
def show_silhouette_score_clustering(labels, coordinates, names):

    # Silhouette Score Plot
    n_clusters = len(set(labels))  # 클러스터 개수 자동 계산
    fig, axs = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)

    sil_avg = silhouette_score(coordinates, labels)
    sil_values = silhouette_samples(coordinates, labels)
    print("sil_avg = ", sil_avg, "\nsil_values = ", sil_values)

    y_lower = 1
    axs.set_title(f'Number of Clusters: {n_clusters}\nSilhouette Score: {round(sil_avg, 3)}')
    axs.set_xlabel("Silhouette coefficient values")
    axs.set_ylabel("Cluster label")
    axs.set_xlim([-0.4, 1])
    axs.set_ylim([0, len(coordinates) + (n_clusters + 1)])
    axs.set_yticks([])
    axs.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = plt.get_cmap('cool', 3)
    for i in range(n_clusters):
        ith_cluster_sil_values = sil_values[np.array(labels) == i]
        ith_cluster_sil_values.sort()
        print(ith_cluster_sil_values)

        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i

        axs.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, color = colors(i), alpha=0.7, label=f'Cluster {i + 1}')
        axs.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
        for k, j, name in zip(ith_cluster_sil_values, np.arange(y_lower, y_upper), names):
            plt.scatter(k, j, color='red')
            plt.text(k, j, name)
        y_lower = y_upper + 1

    axs.axvline(x=sil_avg, color="red", linestyle="--")
    plt.legend(loc = 'upper left')
    plt.show()



#데이터 전처리 -> 코사인 유사도 계산 -> 거리 행렬로 변환 -> 클러스터링 -> 거리행렬을 2d 행렬로 변환 -> 시각화 + 라벨링
if __name__ == "__main__":
    students = load_from_json("students_data.json")
    student_names = [student['name'] for student in students] # student name list

    # Student vector extraction / Calculate the Similarity Matrix
    student_vectors = [student["vector"] for student in students]
    similarity_matrix = algorithm.calculate_similarity_matrix(student_vectors)

    # Converting to distance matrix
    distance_matrix = 1 - similarity_matrix

    # Receives Agglomeration clustering with values set / clustering
    clustering = algorithm.agglomerative_clustering()
    agglo_clustering_labels = clustering.fit_predict(distance_matrix)

    # Converting multi dimension Distance Matrix to 2d Matrix
    mds_coordinates = algorithm.mds_scaling(distance_matrix)

    # show clustered result
    show_silhouette_score_clustering(agglo_clustering_labels, mds_coordinates, student_names)

