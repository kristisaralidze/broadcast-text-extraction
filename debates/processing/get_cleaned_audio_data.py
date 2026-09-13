
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





def get_filtered_audio_dataframe(labelled_audio_dataframe, relative_ambiguity_threshold):
    while True:
        filtered_audio_dataframe = remove_ambiguous_segments_from_dataframe(labelled_audio_dataframe)

        if len(filtered_audio_dataframe) == len(labelled_audio_dataframe):
            break

        labelled_audio_dataframe = get_labelled_audio_dataframe(
            audio_dataframe=filtered_audio_dataframe,
            relative_ambiguity_threshold=relative_ambiguity_threshold,
        )
        
    return filtered_audio_dataframe

def remove_ambiguous_segments_from_dataframe(labelled_audio_dataframe):
    filtered_audio_dataframe = labelled_audio_dataframe[~labelled_audio_dataframe["is_ambiguous"]].copy()
    return filtered_audio_dataframe



def get_labelled_audio_dataframe(audio_dataframe: pd.DataFrame, relative_ambiguity_threshold: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standardized_embedding_matrix = standardize_speak_embeddings(audio_dataframe)
    kmeans_model, cluster_labels = run_kmeans(standardized_embedding_matrix, cluster_count=3)

    closest_centroid_distance, second_closest_centroid_distance, relative_margin, is_ambiguous = compute_cluster_metrics(kmeans_model, standardized_embedding_matrix, relative_ambiguity_threshold)

    labelled_audio_dataframe = audio_dataframe.copy()
    labelled_audio_dataframe["speaker_cluster"] = cluster_labels + 1
    labelled_audio_dataframe["closest_centroid_distance"] = closest_centroid_distance
    labelled_audio_dataframe["second_closest_centroid_distance"] = second_closest_centroid_distance
    labelled_audio_dataframe["relative_margin"] = relative_margin
    labelled_audio_dataframe["is_ambiguous"] = is_ambiguous

    principal_component_matrix = get_principal_components_matrix(standardized_embedding_matrix)
    labelled_audio_dataframe["principal_component_1"] = principal_component_matrix[:, 0]
    labelled_audio_dataframe["principal_component_2"] = principal_component_matrix[:, 1]

    return labelled_audio_dataframe


def standardize_speak_embeddings(audio_dataframe: pd.DataFrame) -> np.ndarray:
    speak_embeddings_matrix = build_speak_embeddings_matrix(audio_dataframe)
    return StandardScaler().fit_transform(speak_embeddings_matrix)


# stacks the vectors of length 512 
def build_speak_embeddings_matrix(audio_dataframe: pd.DataFrame) -> np.ndarray:
    return np.vstack(audio_dataframe["speak_embeddings"].to_numpy())


def run_kmeans(standardized_embedding_matrix: np.ndarray, cluster_count: int) -> tuple[KMeans, np.ndarray]:
    kmeans_model = KMeans(n_clusters=cluster_count, n_init=20, random_state=100)
    cluster_labels = kmeans_model.fit_predict(standardized_embedding_matrix)
    return kmeans_model, cluster_labels


def compute_cluster_metrics(kmeans_model: KMeans, standardized_embedding_matrix: np.ndarray, relative_ambiguity_threshold: float) -> pd.DataFrame:
    centroid_distances = kmeans_model.transform(standardized_embedding_matrix)
    sorted_distances = np.sort(centroid_distances, axis=1)

    closest_centroid_distance = sorted_distances[:, 0]
    second_closest_centroid_distance = sorted_distances[:, 1]

    relative_margin = (second_closest_centroid_distance - closest_centroid_distance) / np.maximum(closest_centroid_distance, 1e-12)
    is_ambiguous = relative_margin < relative_ambiguity_threshold

    return closest_centroid_distance, second_closest_centroid_distance, relative_margin, is_ambiguous


def get_principal_components_matrix(standardized_embedding_matrix):
    pca_model = PCA(n_components=2, random_state=100)
    principal_component_matrix = pca_model.fit_transform(standardized_embedding_matrix)
    return principal_component_matrix

