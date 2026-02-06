import pandas as pd
from src.inference import run_inference

if __name__ == "__main__":
    df = pd.read_csv("data/raw/SpotifyFeatures.csv")
    scored_df = run_inference(df)

    scored_df.to_csv(
        "data/processed/scored_tracks.csv",
        index=False
    )

    print(scored_df[[
        "track_id",
        "genre",
        "risk_probability",
    ]].head(10))
    print("Scored tracks saved to data/processed/scored_tracks1.csv")
