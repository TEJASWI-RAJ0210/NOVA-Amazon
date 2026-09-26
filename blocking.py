"""
blocking.py — Candidate generation (blocking) stage.

Run this once per team with --other-source source2 (Team A) or
--other-source source3 (Team B). It reads the cleaned files produced by
clean.py and outputs, for every Source 1 entity, a shortlist of plausible
Source 2 (or Source 3) candidates — this shortlist is what features.py and
train_and_predict.py will score.

Strategy (two blocking passes, unioned together for higher recall):
  1. Key blocking: group by (country, sorted-token-prefix-of-name). Cheap,
     catches word-order transpositions and country mismatches immediately.
  2. TF-IDF character n-gram blocking: vectorize name+address text with a
     char n-gram TF-IDF, then for each S1 row find its top-K nearest S2/S3
     rows by cosine similarity, restricted to the same country group where
     possible. Catches typos / abbreviations that key blocking misses.

We UNION both passes' candidates (better recall, features + model handle
precision later). Always measure recall ceiling on train before trusting
this on test.

Usage:
    python3 blocking.py --other-source source2
    python3 blocking.py --other-source source3

Outputs (next to dataset/):
    candidates_train_<other_source>.tsv   columns: source1_entity_id, candidate_entity_id
    candidates_test_<other_source>.tsv    same columns
"""

import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATASET_DIR = "dataset"
TOP_K = 15          # candidates per S1 entity from the TF-IDF pass
NAME_PREFIX_LEN = 3  # tokens used to build the key-blocking key


def _load_cleaned(split: str, source: str) -> pd.DataFrame:
    path = os.path.join(DATASET_DIR, split, f"cleaned_{split}_{source}.tsv")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])


def _blocking_key(row) -> str:
    """country + first N sorted name tokens — order-independent, cheap key."""
    tokens = str(row.get("business_name_tokens", "")).split()[:NAME_PREFIX_LEN]
    return f"{row.get('country_clean', '')}||{' '.join(tokens)}"


def key_blocking_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    s1_df = s1_df.copy()
    other_df = other_df.copy()
    s1_df["_key"] = s1_df.apply(_blocking_key, axis=1)
    other_df["_key"] = other_df.apply(_blocking_key, axis=1)

    merged = s1_df[["entity_id", "_key"]].merge(
        other_df[["entity_id", "_key"]], on="_key", suffixes=("_s1", "_other")
    )
    merged = merged.rename(
        columns={"entity_id_s1": "source1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    return merged[["source1_entity_id", "candidate_entity_id"]]


def tfidf_blocking_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """
    Fit one char n-gram TF-IDF over both dataframes' combined text, then for
    each S1 row pull its top_k nearest neighbors among `other_df` rows by
    cosine distance. Blocked further by country when both sides have a
    non-empty country (skips the filter for empty/unknown country to avoid
    losing recall on messy rows).
    """
    s1_df = s1_df.copy()
    other_df = other_df.copy()
    s1_df["_text"] = (s1_df["business_name_clean"].fillna("") + " " + s1_df["business_address_clean"].fillna(""))
    other_df["_text"] = (other_df["business_name_clean"].fillna("") + " " + other_df["business_address_clean"].fillna(""))

    results = []
    countries = set(s1_df["country_clean"]) | set(other_df["country_clean"])
    for country in countries:
        s1_sub = s1_df[s1_df["country_clean"] == country]
        other_sub = other_df[other_df["country_clean"] == country]
        if s1_sub.empty or other_sub.empty:
            continue
        _add_nn_candidates(s1_sub, other_sub, top_k, results)

    # Also run one unrestricted pass in case country is missing/unreliable
    # for some rows, so we don't lose true matches to a bad country field.
    unknown_s1 = s1_df[~s1_df["country_clean"].isin(countries) | (s1_df["country_clean"] == "")]
    if not unknown_s1.empty:
        _add_nn_candidates(unknown_s1, other_df, top_k, results)

    if not results:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    return pd.concat(results, ignore_index=True).drop_duplicates()


def _add_nn_candidates(s1_sub: pd.DataFrame, other_sub: pd.DataFrame, top_k: int, results: list):
    if len(other_sub) == 0 or len(s1_sub) == 0:
        return
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    combined_text = pd.concat([s1_sub["_text"], other_sub["_text"]], ignore_index=True)
    vectorizer.fit(combined_text)

    other_matrix = vectorizer.transform(other_sub["_text"])
    s1_matrix = vectorizer.transform(s1_sub["_text"])

    k = min(top_k, len(other_sub))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(other_matrix)
    _, indices = nn.kneighbors(s1_matrix)

    other_ids = other_sub["entity_id"].values
    s1_ids = s1_sub["entity_id"].values
    rows = []
    for i, s1_id in enumerate(s1_ids):
        for idx in indices[i]:
            rows.append((s1_id, other_ids[idx]))
    results.append(pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"]))


def measure_recall_ceiling(candidates: pd.DataFrame, ground_truth_path: str, other_source: str):
    """
    Prints what fraction of true matches survive blocking (train only).
    Only counts ground-truth ids belonging to `other_source` (S2- or S3-),
    since a source2-only candidate set can never contain an S3- match.
    """
    prefix = "S2-" if other_source == "source2" else "S3-"
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""]).fillna("")
    total_true, found_true = 0, 0
    cand_by_s1 = candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(set)
    for _, row in gt.iterrows():
        matched = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if not matched:
            continue
        true_ids = [m for m in matched.split(",") if m and m.startswith(prefix)]
        cand_set = cand_by_s1.get(row["source1_entity_id"], set())
        for tid in true_ids:
            total_true += 1
            if tid in cand_set:
                found_true += 1
    if total_true == 0:
        print(f"[recall check] no {prefix} matches found in ground truth — check filtering")
        return
    print(f"[recall ceiling] {found_true}/{total_true} = {found_true/total_true:.3%} of true {prefix} matches survive blocking")


def run(other_source: str):
    for split in ("train", "test"):
        s1_df = _load_cleaned(split, "source1")
        # only keep rows relevant to this other source's id prefix
        other_df = _load_cleaned(split, other_source)

        key_cands = key_blocking_candidates(s1_df, other_df)
        tfidf_cands = tfidf_blocking_candidates(s1_df, other_df)
        all_cands = pd.concat([key_cands, tfidf_cands], ignore_index=True).drop_duplicates()

        out_path = os.path.join(DATASET_DIR, split, f"candidates_{split}_{other_source}.tsv")
        all_cands.to_csv(out_path, sep="\t", index=False)
        print(f"[ok] {split}/{other_source}: {len(all_cands)} candidate pairs -> {out_path}")

        if split == "train":
            gt_path = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")
            if os.path.exists(gt_path):
                measure_recall_ceiling(all_cands, gt_path, other_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--other-source", required=True, choices=["source2", "source3"])
    args = parser.parse_args()
    run(args.other_source)
