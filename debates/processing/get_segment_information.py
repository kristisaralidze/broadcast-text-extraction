# Simpler version: build one final dataset with every filtered segment labelled by candidate.
# Main output: all_candidate_segments_dataframe wiht mapped segments to candidaatees 

import pandas as pd
import numpy as np
from pathlib import Path

from processing.get_audio_and_speech_data import (
    get_audio_dataframe,
    )


from processing.get_cleaned_audio_data import (
    get_labelled_audio_dataframe,
    get_filtered_audio_dataframe
    )


PROJECT_FEATURES_DIR = Path(__file__).parent / "Project_Features"





def candidate_is_in_debate(candidate_name: str, audio_file_name: str) -> bool:
    return candidate_name in get_candidates_in_debate(audio_file_name)

def get_candidates_in_debate(audio_file_name: str) -> tuple[str, str]:
    left_candidate_name, right_side = get_debate_name(audio_file_name).split("_vs_", 1)
    right_candidate_name = "_".join(right_side.split("_")[:-2])
    return left_candidate_name, right_candidate_name

def get_debate_name(audio_file_name: str) -> str:
    return audio_file_name.removesuffix("_audio")




def get_candidate_cluster_options(candidate_name: str, audio_file_names: list[str], relative_ambiguity_threshold: float) -> pd.DataFrame:
    cluster_options = []

    for audio_file_name in audio_file_names:
        if not candidate_is_in_debate(candidate_name, audio_file_name):
            continue

        labelled_audio_dataframe = get_labelled_audio_dataframe(get_audio_dataframe(audio_file_name),relative_ambiguity_threshold)
        filtered_audio_dataframe = get_filtered_audio_dataframe(labelled_audio_dataframe, relative_ambiguity_threshold)
        
        host_cluster = get_host_cluster(filtered_audio_dataframe)

        for speaker_cluster in sorted(filtered_audio_dataframe["speaker_cluster"].unique()):
            if speaker_cluster == host_cluster:
                continue

            cluster_dataframe = filtered_audio_dataframe.loc[filtered_audio_dataframe["speaker_cluster"] == speaker_cluster]

            cluster_options.append(
                {
                    "candidate_name": candidate_name,
                    "audio_file_name": audio_file_name,
                    "debate_name": get_debate_name(audio_file_name),
                    "speaker_cluster": int(speaker_cluster),
                    "host_cluster": host_cluster,
                    "segment_count": len(cluster_dataframe),
                    "total_duration_seconds": cluster_dataframe["duration"].sum(),
                    "centroid": get_cluster_centroid(filtered_audio_dataframe, speaker_cluster),
                }
            )

    return pd.DataFrame(cluster_options)


# We just assume the host cluster is the one with the smallest k means cluster 
def get_host_cluster(filtered_audio_dataframe: pd.DataFrame) -> int:
    cluster_duration = filtered_audio_dataframe.groupby("speaker_cluster")["duration"].sum()
    return int(cluster_duration.idxmin())


def get_cluster_centroid(audio_dataframe: pd.DataFrame, speaker_cluster: int) -> np.ndarray:
    cluster_embeddings = audio_dataframe.loc[
        audio_dataframe["speaker_cluster"] == speaker_cluster,
        "speak_embeddings",
    ]
    return np.vstack(cluster_embeddings.to_numpy()).mean(axis=0)





def add_candidate_similarity_scores(cluster_options: pd.DataFrame) -> pd.DataFrame:
    scored_options = cluster_options.copy()
    similarity_scores = []

    for _, cluster_option in scored_options.iterrows():
        best_similarity_by_debate = []

        other_debate_names = sorted(
            debate_name for debate_name in scored_options["debate_name"].unique()
            if debate_name != cluster_option["debate_name"]
        )

        for other_debate_name in other_debate_names:
            other_debate_options = scored_options.loc[scored_options["debate_name"] == other_debate_name]

            debate_similarities = [get_cosine_similarity(cluster_option["centroid"], other_option["centroid"])
                for _, other_option in other_debate_options.iterrows()]
            
            best_similarity_by_debate.append(max(debate_similarities))

        similarity_scores.append(np.mean(best_similarity_by_debate))

    scored_options["candidate_similarity_score"] = similarity_scores
    return scored_options


def get_cosine_similarity(first_vector: np.ndarray, second_vector: np.ndarray) -> float:
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator == 0:
        return 0.0
    return float(np.dot(first_vector, second_vector) / denominator)



def infer_candidate_clusters(candidate_name: str, audio_file_names: list[str], relative_ambiguity_threshold: float) -> pd.DataFrame:
    cluster_options = get_candidate_cluster_options(candidate_name, audio_file_names, relative_ambiguity_threshold)
    scored_options = add_candidate_similarity_scores(cluster_options)
    inferred_clusters = []

    for debate_name, debate_options in scored_options.groupby("debate_name"):
        debate_options = debate_options.sort_values(
            "candidate_similarity_score",
            ascending=False,
        ).reset_index(drop=True)

        best_option = debate_options.iloc[0]
        second_option = debate_options.iloc[1] if len(debate_options) > 1 else None
        second_score = second_option["candidate_similarity_score"] if second_option is not None else np.nan

        inferred_clusters.append(
            {
                "candidate_name": candidate_name,
                "audio_file_name": best_option["audio_file_name"],
                "debate_name": debate_name,
                "speaker_cluster": int(best_option["speaker_cluster"]),
                "host_cluster": int(best_option["host_cluster"]),
                "segment_count": int(best_option["segment_count"]),
                "total_duration_seconds": float(best_option["total_duration_seconds"]),
                "candidate_similarity_score": float(best_option["candidate_similarity_score"]),
                "second_speaker_cluster": int(second_option["speaker_cluster"]) if second_option is not None else -1,
                "second_similarity_score": float(second_score) if pd.notna(second_score) else np.nan,
                "similarity_score_margin": float(best_option["candidate_similarity_score"] - second_score) if pd.notna(second_score) else np.nan,
            }
        )

    return pd.DataFrame(inferred_clusters)



def get_candidate_segments(candidate_clusters: pd.DataFrame, relative_ambiguity_threshold) -> pd.DataFrame:
    candidate_segments = []

    for _, candidate_cluster in candidate_clusters.iterrows():
        audio_dataframe = get_audio_dataframe(candidate_cluster["audio_file_name"])
        labelled_audio_dataframe = get_labelled_audio_dataframe(audio_dataframe, relative_ambiguity_threshold)
        filtered_audio_dataframe = get_filtered_audio_dataframe(labelled_audio_dataframe, relative_ambiguity_threshold)

        labelled_segments = filtered_audio_dataframe.loc[filtered_audio_dataframe["speaker_cluster"] == candidate_cluster["speaker_cluster"]].copy()

        labelled_segments["candidate_name"] = candidate_cluster["candidate_name"]
        labelled_segments["debate_name"] = candidate_cluster["debate_name"]
        labelled_segments["audio_file_name"] = candidate_cluster["audio_file_name"]
        labelled_segments["candidate_similarity_score"] = candidate_cluster["candidate_similarity_score"]
        candidate_segments.append(labelled_segments)

    return pd.concat(candidate_segments, ignore_index=True)



def add_speech_data_to_candidate_segments(all_candidate_segments_dataframe):
    speech_dataframes = []

    for debate_name in all_candidate_segments_dataframe["debate_name"].unique():
        speech_path = PROJECT_FEATURES_DIR / f"{debate_name}_speech.pkl"

        if not speech_path.exists():
            continue

        speech_dataframe = pd.read_pickle(speech_path).copy()
        speech_dataframe["debate_name"] = debate_name

        speech_dataframes.append(speech_dataframe)

    all_speech_dataframe = pd.concat(speech_dataframes, ignore_index=True)

    candidate_segments_with_speech = all_candidate_segments_dataframe.merge(
        all_speech_dataframe[["debate_name", "timestamp", "duration", "transcript", "text_embedding"]],
        left_on=["debate_name", "time stamp", "duration"],
        right_on=["debate_name", "timestamp", "duration"],
        how="left"
    )

    candidate_segments_with_speech["has_transcript"] = (
        candidate_segments_with_speech["transcript"].notna()
    )

    return candidate_segments_with_speech




def get_host_segments_dataframe(audio_file_names, relative_ambiguity_threshold):
    host_segments = []

    for audio_file_name in audio_file_names:
        debate_name = audio_file_name.removesuffix("_audio")

        audio_dataframe = get_audio_dataframe(audio_file_name)
        labelled_audio_dataframe = get_labelled_audio_dataframe(audio_dataframe, relative_ambiguity_threshold)
        filtered_audio_dataframe = get_filtered_audio_dataframe(labelled_audio_dataframe, relative_ambiguity_threshold)
        host_cluster = get_host_cluster(filtered_audio_dataframe)

        debate_host_segments = filtered_audio_dataframe.loc[filtered_audio_dataframe["speaker_cluster"] == host_cluster].copy()

        debate_host_segments["debate_name"] = debate_name
        debate_host_segments["audio_file_name"] = audio_file_name

        speech_path = PROJECT_FEATURES_DIR / f"{debate_name}_speech.pkl"

        if speech_path.exists():
            speech_dataframe = pd.read_pickle(speech_path).copy()

            debate_host_segments = debate_host_segments.merge(
                speech_dataframe[["timestamp", "duration", "transcript", "text_embedding"]],
                left_on=["time stamp", "duration"],
                right_on=["timestamp", "duration"],
                how="left"
            )

        else:
            debate_host_segments["transcript"] = pd.NA
            debate_host_segments["text_embedding"] = pd.NA

        debate_host_segments["has_transcript"] = debate_host_segments["transcript"].notna()

        host_segments.append(debate_host_segments)

    return pd.concat(host_segments, ignore_index=True)
