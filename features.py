"""
features.py — Pairwise similarity feature engineering.

Reads the candidate pairs from blocking.py and the cleaned source files from
clean.py, and computes a feature row per (source1_entity_id, candidate_entity_id)
pair. These features feed train_and_predict.py.

Usage:
    python3 features.py --other-source source2
    python3 features.py --other-source source3

Outputs:
    features_train_<other_source>.tsv
    features_test_<other_source>.tsv
"""

import argparse
import os
from difflib import SequenceMatcher

import pandas as pd

DATASET_DIR = "dataset"


def _load_cleaned(split: str, source: str) -> pd.DataFrame:
    path = os.path.join(DATASET_DIR, split, f"cleaned_{split}_{source}.tsv")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""]).fillna("")


def _load_candidates(split: str, other_source: str) -> pd.DataFrame:
    path = os.path.join(DATASET_DIR, split, f"candidates_{split}_{other_source}.tsv")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""]).fillna("")


def jaccard(tokens_a: str, tokens_b: str) -> float:
    set_a, set_b = set(tokens_a.split()), set(tokens_b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def levenshtein_ratio(a: str, b: str) -> float:
    """difflib ratio as a dependency-free stand-in for normalized edit distance."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def token_sort_similarity(a: str, b: str) -> float:
    """Sort tokens alphabetically before comparing — neutralizes word-order swaps."""
    return levenshtein_ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def build_features(s1_df: pd.DataFrame, other_df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    s1_lookup = s1_df.set_index("entity_id")
    other_lookup = other_df.set_index("entity_id")

    rows = []
    for _, cand in candidates.iterrows():
        s1_id, other_id = cand["source1_entity_id"], cand["candidate_entity_id"]
        if s1_id not in s1_lookup.index or other_id not in other_lookup.index:
            continue
        s1_row = s1_lookup.loc[s1_id]
        other_row = other_lookup.loc[other_id]

        name_a, name_b = s1_row["business_name_clean"], other_row["business_name_clean"]
        core_a, core_b = s1_row.get("business_name_core", ""), other_row.get("business_name_core", "")
        addr_a, addr_b = s1_row["business_address_clean"], other_row["business_address_clean"]
        tok_name_a, tok_name_b = s1_row["business_name_tokens"], other_row["business_name_tokens"]
        tok_addr_a, tok_addr_b = s1_row["business_address_tokens"], other_row["business_address_tokens"]
        country_a, country_b = s1_row.get("country_clean", ""), other_row.get("country_clean", "")

        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": other_id,
            "name_jaccard": jaccard(tok_name_a, tok_name_b),
            "name_levenshtein": levenshtein_ratio(name_a, name_b),
            "name_core_levenshtein": levenshtein_ratio(core_a, core_b),
            "name_token_sort_ratio": token_sort_similarity(name_a, name_b),
            "name_common_prefix_len": common_prefix_len(name_a, name_b),
            "name_length_diff": abs(len(name_a) - len(name_b)),
            "address_jaccard": jaccard(tok_addr_a, tok_addr_b),
            "address_levenshtein": levenshtein_ratio(addr_a, addr_b),
            "country_match": int(country_a == country_b and country_a != ""),
            "country_known": int(country_a != "" and country_b != ""),
        })

    return pd.DataFrame(rows)


def run(other_source: str):
    for split in ("train", "test"):
        s1_df = _load_cleaned(split, "source1")
        other_df = _load_cleaned(split, other_source)
        candidates = _load_candidates(split, other_source)

        feat_df = build_features(s1_df, other_df, candidates)
        out_path = os.path.join(DATASET_DIR, split, f"features_{split}_{other_source}.tsv")
        feat_df.to_csv(out_path, sep="\t", index=False)
        print(f"[ok] {split}/{other_source}: {len(feat_df)} feature rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--other-source", required=True, choices=["source2", "source3"])
    args = parser.parse_args()
    run(args.other_source)
