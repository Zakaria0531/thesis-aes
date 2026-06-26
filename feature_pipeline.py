"""Entry point for feature extraction; writes features.parquet and feature_delta.parquet."""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.features import RANDOM_SEED, get_features, load_essays, load_spacy_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR   = Path("data")
OUTPUT_DIR = Path(".")


def main(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    essays = load_essays(data_dir)
    if essays.empty:
        logger.error("No essays found in '%s'. Aborting.", data_dir)
        return

    nlp = load_spacy_model()

    logger.info("Featurizing %d essays ...", len(essays))
    records: list[dict] = []
    for _, row in tqdm(essays.iterrows(), total=len(essays), desc="Featurizing"):
        records.append({
            "essay_id":    row["essay_id"],
            "prompt_name": row["prompt_name"],
            "level":       row["level"],
            "score":       row["score"],
            **get_features(row["text"], nlp),
        })

    features_df   = pd.DataFrame(records)
    features_path = output_dir / "features.parquet"
    features_df.to_parquet(features_path, index=False)
    logger.info("Saved features → '%s' (%d rows)", features_path, len(features_df))

    meta_cols    = {"essay_id", "prompt_name", "level", "score"}
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    originals    = features_df[features_df["level"] == 0].set_index("essay_id")[feature_cols]

    delta_records: list[dict] = []
    for _, row in features_df[features_df["level"] > 0].iterrows():
        eid = row["essay_id"]
        if eid not in originals.index:
            logger.warning("Original not found for essay_id '%s' — skipping delta.", eid)
            continue
        delta_records.append({
            "essay_id":    eid,
            "prompt_name": row["prompt_name"],
            "level":       row["level"],
            "score":       row["score"],
            **{col: row[col] - originals.loc[eid, col] for col in feature_cols},
        })

    delta_df   = pd.DataFrame(delta_records)
    delta_path = output_dir / "feature_delta.parquet"
    delta_df.to_parquet(delta_path, index=False)
    logger.info("Saved feature deltas → '%s' (%d rows)", delta_path, len(delta_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract linguistic features from AI-intervened essays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",    default=str(DATA_DIR))
    parser.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir))