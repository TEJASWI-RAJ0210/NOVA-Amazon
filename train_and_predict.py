"""
train_and_predict.py — Build labeled training pairs, train a classifier,
tune the decision threshold for macro F0.5, and generate this team's
matching_<other_source>.tsv and candidates_<other_source>.tsv.

Both the Logistic Regression and Gradient Boosting models used here are
classical ML (not LLMs) — trivially compliant with the "<=8B params,
MIT/Apache 2.0" rule, and no external data/APIs are used anywhere.

Usage:
    python3 train_and_predict.py --other-source source2
    python3 train_and_predict.py --other-source source3

Outputs (in output/):
    matching_<other_source>.tsv    # source1_entity_id, matched_entity_ids (this source only)
    candidates_<other_source>.tsv  # source1_entity_id, candidate_entity_ids (this source only)
"""

import argparse
import os
import random

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

DATASET_DIR = "dataset"
OUTPUT_DIR = "output"
FEATURE_COLS = [
    "name_jaccard", "name_levenshtein", "name_core_levenshtein",
    "name_token_sort_ratio", "name_common_prefix_len", "name_length_diff",
    "address_jaccard", "address_levenshtein", "country_match", "country_known",
]
RANDOM_SEED = 42


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])


def build_labels(features_df: pd.DataFrame, ground_truth_path: str) -> pd.DataFrame:
    """Attach a 0/1 `label` column: 1 if this candidate pair is a true match."""
    gt = _load(ground_truth_path).fillna("")
    true_pairs = set()
    for _, row in gt.iterrows():
        matched = row["matched_entity_ids"]
        if not matched:
            continue
        for mid in str(matched).split(","):
            if mid:
                true_pairs.add((row["source1_entity_id"], mid))

    features_df = features_df.copy()
    features_df["label"] = features_df.apply(
        lambda r: int((r["source1_entity_id"], r["candidate_entity_id"]) in true_pairs), axis=1
    )
    return features_df


def group_train_val_split(labeled_df: pd.DataFrame, val_frac: float = 0.2):
    """Split by source1_entity_id so no S1 entity's pairs leak across train/val."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=RANDOM_SEED)
    groups = labeled_df["source1_entity_id"]
    train_idx, val_idx = next(gss.split(labeled_df, groups=groups))
    return labeled_df.iloc[train_idx].reset_index(drop=True), labeled_df.iloc[val_idx].reset_index(drop=True)


def f_beta_macro(pred_matches: dict, true_matches: dict, beta: float = 0.5) -> float:
    """
    Macro-average F_beta across all S1 entity ids present in true_matches.
    pred_matches / true_matches: dict of source1_entity_id -> set(matched_ids)
    A singleton (empty true set) scores 1.0 if predicted empty, else 0.0.
    """
    scores = []
    for s1_id, true_set in true_matches.items():
        pred_set = pred_matches.get(s1_id, set())
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(true_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue
        f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        scores.append(f)
    return sum(scores) / len(scores) if scores else 0.0


def pairs_to_dict(df: pd.DataFrame, id_col: str, score_col: str = None, threshold: float = None) -> dict:
    """Group candidate/matched ids per source1_entity_id into a dict of sets."""
    result = {}
    for s1_id, group in df.groupby("source1_entity_id"):
        if score_col is not None and threshold is not None:
            ids = set(group.loc[group[score_col] >= threshold, id_col])
        else:
            ids = set(group[id_col])
        result[s1_id] = ids
    return result


def tune_threshold(val_df: pd.DataFrame, val_scores, true_matches: dict) -> float:
    """Sweep thresholds, return the one maximizing macro F0.5 on validation."""
    val_df = val_df.copy()
    val_df["score"] = val_scores
    best_thresh, best_f = 0.5, -1.0
    for thresh in [i / 100 for i in range(5, 100, 5)]:
        pred = pairs_to_dict(val_df, "candidate_entity_id", score_col="score", threshold=thresh)
        # include S1 ids with zero surviving candidates as empty predictions
        f = f_beta_macro(pred, true_matches, beta=0.5)
        if f > best_f:
            best_f, best_thresh = f, thresh
    print(f"[threshold tuning] best threshold={best_thresh:.2f}, val macro F0.5={best_f:.4f}")
    return best_thresh


def run(other_source: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_feat_path = os.path.join(DATASET_DIR, "train", f"features_train_{other_source}.tsv")
    test_feat_path = os.path.join(DATASET_DIR, "test", f"features_test_{other_source}.tsv")
    gt_path = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")

    train_feat = _load(train_feat_path)
    for col in FEATURE_COLS:
        train_feat[col] = pd.to_numeric(train_feat[col], errors="coerce").fillna(0)

    labeled = build_labels(train_feat, gt_path)

    # Also need every S1 id's true match set (including singletons with none
    # of their candidates surviving blocking) for correct macro F0.5 scoring.
    gt = _load(gt_path).fillna("")
    true_matches_all = {}
    for _, row in gt.iterrows():
        matched = str(row["matched_entity_ids"])
        ids = set(m for m in matched.split(",") if m) if matched else set()
        # keep only ids relevant to this source (S2 for team A, S3 for team B)
        prefix = "S2-" if other_source == "source2" else "S3-"
        true_matches_all[row["source1_entity_id"]] = {i for i in ids if i.startswith(prefix)}

    train_split, val_split = group_train_val_split(labeled)

    X_train, y_train = train_split[FEATURE_COLS], train_split["label"]
    X_val = val_split[FEATURE_COLS]

    val_true = {sid: ids for sid, ids in true_matches_all.items() if sid in set(val_split["source1_entity_id"])}

    candidates_models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "gboost": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }

    best_model_name, best_model, best_thresh, best_f = None, None, 0.5, -1.0
    if y_train.nunique() < 2:
        print("[warn] training fold has only one class (too few positives in this split) — "
              "this is expected on very small/toy data, but on the real dataset check your "
              "blocking recall and ground-truth join if you see this. Falling back to a "
              "rule-based similarity score for this run.")
    else:
        for name, model in candidates_models.items():
            try:
                model.fit(X_train, y_train)
            except ValueError as e:
                print(f"[warn] model={name} failed to fit ({e}); skipping")
                continue
            val_scores = model.predict_proba(X_val)[:, 1]
            thresh = tune_threshold(val_split, val_scores, val_true)
            pred = pairs_to_dict(
                val_split.assign(score=val_scores), "candidate_entity_id", score_col="score", threshold=thresh
            )
            f = f_beta_macro(pred, val_true, beta=0.5)
            print(f"[model={name}] val macro F0.5={f:.4f} @ threshold={thresh:.2f}")
            if f > best_f:
                best_f, best_model_name, best_model, best_thresh = f, name, model, thresh

    if best_model_name is not None:
        print(f"[selected] model={best_model_name}, threshold={best_thresh:.2f}, val macro F0.5={best_f:.4f}")
        # Refit best model on ALL labeled train data (train+val) before predicting test
        final_model = candidates_models[best_model_name]
        final_model.fit(labeled[FEATURE_COLS], labeled["label"])
    else:
        final_model = None
        best_thresh = 0.5
        print("[fallback] no model could be trained — using a rule-based average-similarity score instead")

    test_feat = _load(test_feat_path)
    for col in FEATURE_COLS:
        test_feat[col] = pd.to_numeric(test_feat[col], errors="coerce").fillna(0)

    if len(test_feat) > 0:
        if final_model is not None:
            test_feat["score"] = final_model.predict_proba(test_feat[FEATURE_COLS])[:, 1]
        else:
            # Rule-based fallback: simple average of the two strongest similarity signals.
            test_feat["score"] = (test_feat["name_jaccard"] + test_feat["address_jaccard"]) / 2
    else:
        test_feat["score"] = []

    # Write this team's slice of candidate_pairs.tsv (pre-threshold, full candidate set)
    test_source1 = _load(os.path.join(DATASET_DIR, "test", "test_source1.tsv"))
    all_s1_ids = set(test_source1["entity_id"])

    cand_dict = pairs_to_dict(test_feat, "candidate_entity_id") if len(test_feat) else {}
    match_dict = (
        pairs_to_dict(test_feat, "candidate_entity_id", score_col="score", threshold=best_thresh)
        if len(test_feat) else {}
    )

    cand_rows = [{"source1_entity_id": sid, "candidate_entity_ids": ",".join(sorted(cand_dict.get(sid, set())))}
                 for sid in all_s1_ids]
    match_rows = [{"source1_entity_id": sid, "matched_entity_ids": ",".join(sorted(match_dict.get(sid, set())))}
                  for sid in all_s1_ids]

    pd.DataFrame(cand_rows).to_csv(os.path.join(OUTPUT_DIR, f"candidates_{other_source}.tsv"), sep="\t", index=False)
    pd.DataFrame(match_rows).to_csv(os.path.join(OUTPUT_DIR, f"matching_{other_source}.tsv"), sep="\t", index=False)
    print(f"[ok] wrote output/candidates_{other_source}.tsv and output/matching_{other_source}.tsv")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--other-source", required=True, choices=["source2", "source3"])
    args = parser.parse_args()
    run(args.other_source)
