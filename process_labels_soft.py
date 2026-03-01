import pandas as pd
import numpy as np

# Process labels_detailed.csv into soft-label CSVs for soft-label learning experiments.
#
# Two output CSVs are produced:
#   processed_labels_soft_P.csv  -- P-type: fraction of annotators who chose each emotion
#                                    as their PRIMARY label (EmoClass_Major)
#   processed_labels_soft_S.csv  -- S-type: fraction of annotator mentions across ALL
#                                    emotions perceived (EmoClass_Major + EmoClass_Second)
#
# References:
#   Shamsi et al. (Odyssey 2024) - CONILIUM: agreement-aware loss functions for SER
#   Chou et al. (2024) - Stimulus Modality Matters (all-inclusive label concept)

DETAILED_LABEL_PATH = '/Users/othsueh/Development/MSP-PODCAST_Lables/Labels/labels_detailed.csv'
CONSENSUS_LABEL_PATH = '/Users/othsueh/Development/MSP-PODCAST_Lables/Labels/labels_consensus.csv'
OUT_SOFT_P = '/Users/othsueh/Development/MSP-PODCAST_Lables/Labels/processed_labels_soft_P.csv'
OUT_SOFT_S = '/Users/othsueh/Development/MSP-PODCAST_Lables/Labels/processed_labels_soft_S.csv'

# The 8 standard emotion categories used in the model
EMOTIONS = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]

# Map from full name used in detailed labels → canonical name in EMOTIONS
EMOTION_NAME_MAP = {
    "Angry": "Angry",
    "Sad": "Sad",
    "Happy": "Happy",
    "Surprise": "Surprise",
    "Fear": "Fear",
    "Disgust": "Disgust",
    "Contempt": "Contempt",
    "Neutral": "Neutral",
}


def parse_secondary(raw):
    """Parse a comma-separated secondary emotion string into a list of canonical names.
    Returns only emotions that map to the 8 standard classes; ignores Others, unknowns."""
    if not isinstance(raw, str) or raw.strip() == "":
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [EMOTION_NAME_MAP[p] for p in parts if p in EMOTION_NAME_MAP]


def build_split_map(consensus_path):
    """Build a FileName -> Split_Set lookup from the consensus label file."""
    df = pd.read_csv(consensus_path)
    return dict(zip(df["FileName"], df["Split_Set"]))


def process_soft_P(detailed_df, split_map):
    """
    P-type soft labels: for each sample, the fraction of annotators who selected
    each emotion as their primary label (EmoClass_Major).
    Annotators who chose an 'Other-*' or unmapped primary label contribute 0 to all classes.
    """
    records = []
    for filename, group in detailed_df.groupby("FileName"):
        split = split_map.get(filename)
        if split is None:
            continue

        counts = {e: 0 for e in EMOTIONS}
        total_workers = len(group)

        for _, row in group.iterrows():
            major = str(row["EmoClass_Major"]).strip()
            if major in EMOTION_NAME_MAP:
                counts[EMOTION_NAME_MAP[major]] += 1

        total_valid = sum(counts.values())
        if total_valid == 0:
            # All annotators chose Other; drop this sample
            continue

        # Normalize by total workers (not just valid ones), per paper definition
        row_out = {"FileName": filename}
        for e in EMOTIONS:
            row_out[e] = counts[e] / total_workers
        row_out["Split_Set"] = split
        records.append(row_out)

    return pd.DataFrame(records, columns=["FileName"] + EMOTIONS + ["Split_Set"])


def process_soft_S(detailed_df, split_map):
    """
    S-type soft labels: for each sample, the fraction of total emotion mentions
    (across all annotators, primary + secondary) that belong to each of the 8 classes.
    'Other-*' and unmapped emotion names are ignored.
    """
    records = []
    for filename, group in detailed_df.groupby("FileName"):
        split = split_map.get(filename)
        if split is None:
            continue

        counts = {e: 0 for e in EMOTIONS}

        for _, row in group.iterrows():
            # Primary label
            major = str(row["EmoClass_Major"]).strip()
            if major in EMOTION_NAME_MAP:
                counts[EMOTION_NAME_MAP[major]] += 1

            # Secondary labels (comma-separated)
            for emo in parse_secondary(row.get("EmoClass_Second", "")):
                counts[emo] += 1

        total_mentions = sum(counts.values())
        if total_mentions == 0:
            # No standard-class mentions at all; drop this sample
            continue

        row_out = {"FileName": filename}
        for e in EMOTIONS:
            row_out[e] = counts[e] / total_mentions
        row_out["Split_Set"] = split
        records.append(row_out)

    return pd.DataFrame(records, columns=["FileName"] + EMOTIONS + ["Split_Set"])


def main():
    print("Loading detailed labels...")
    detailed_df = pd.read_csv(DETAILED_LABEL_PATH)
    print(f"  Loaded {len(detailed_df)} annotator rows for {detailed_df['FileName'].nunique()} files")

    print("Loading consensus labels for Split_Set mapping...")
    split_map = build_split_map(CONSENSUS_LABEL_PATH)
    print(f"  {len(split_map)} files have a Split_Set partition")

    print("\nProcessing P-type soft labels...")
    df_P = process_soft_P(detailed_df, split_map)
    print(f"  {len(df_P)} samples retained")
    print(f"  Split counts:\n{df_P['Split_Set'].value_counts().to_string()}")
    print(f"  Label sum stats (should be ~1.0):\n{df_P[EMOTIONS].sum(axis=1).describe()}")
    df_P.to_csv(OUT_SOFT_P, index=False)
    print(f"  Saved to {OUT_SOFT_P}")

    print("\nProcessing S-type soft labels...")
    df_S = process_soft_S(detailed_df, split_map)
    print(f"  {len(df_S)} samples retained")
    print(f"  Split counts:\n{df_S['Split_Set'].value_counts().to_string()}")
    print(f"  Label sum stats (should be ~1.0):\n{df_S[EMOTIONS].sum(axis=1).describe()}")
    df_S.to_csv(OUT_SOFT_S, index=False)
    print(f"  Saved to {OUT_SOFT_S}")

    print("\nDone.")


if __name__ == "__main__":
    main()
