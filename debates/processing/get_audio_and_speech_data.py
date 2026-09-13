from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_FEATURES_DIR = Path(__file__).parent / "Project_Features"



def load_dataframe(input_file_name) -> pd.DataFrame:
    file_path = PROJECT_FEATURES_DIR / Path(f"{input_file_name}.pkl")
    return pd.read_pickle(file_path)


def get_audio_dataframe(input_file_name: str) -> pd.DataFrame:
    audio_dataframe = load_dataframe(input_file_name)

    required_columns = ["time stamp","duration","meanF0Hz","stdevF0Hz","HNR","localJitter","localabsoluteJitter",
                        "rapJitter","ppq5Jitter","ddpJitter","localShimmer","localdbShimmer","apq3Shimmer","apq5Shimmer",
                        "apq11Shimmer","ddaShimmer","f1_mean","f2_mean","f3_mean","f4_mean","f1_median","f2_median","f3_median",
                        "f4_median","npause","speechrate","articulationrate","asd","speak_embeddings","pF","fdisp","avgFormant","mff"]
    
    # to check if all the pickle files are complete 
    missing_columns = [column_name for column_name in required_columns if column_name not in audio_dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # to check that everything is good with spaeker embeddings 
    embedding_lengths = audio_dataframe["speak_embeddings"].apply(len)
    if embedding_lengths.nunique() != 1:
        raise ValueError("Not all speaker embeddings have the same length.")

    return audio_dataframe


def get_audio_file_names():
    return [p.stem for p in PROJECT_FEATURES_DIR.glob("*audio.pkl")]


def get_candidate_names(audio_file_names):
    return list(dict.fromkeys(candidate for file_name in audio_file_names for candidate in (
        lambda left, right: [left, "_".join(right.split("_")[:-2])])(*file_name.removesuffix("_audio").split("_vs_", 1))))




