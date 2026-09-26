"""
merge_and_validate.py — Combine Team A's (S1<->S2) and Team B's (S1<->S3)
outputs into the single final matching_results.tsv and candidate_pairs.tsv,
then run the official utils/validate_submission.py against them.

Usage:
    python3 merge_and_validate.py \
        --matching-a output/matching_source2.tsv --matching-b output/matching_source3.tsv \
        --candidates-a output/candidates_source2.tsv --candidates-b output/candidates_source3.tsv \
        --test-dir dataset/test \
        --out-dir output/
"""

import argparse
import os
import subprocess
import sys

import pandas as pd


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""]).fillna("")


def _merge_id_lists(a_path: str, b_path: str, id_col: str) -> pd.DataFrame:
    df_a = _load(a_path)
    df_b = _load(b_path)

    merged = pd.merge(df_a, df_b, on="source1_entity_id", how="outer", suffixes=("_a", "_b")).fillna("")

    def combine(row):
        ids_a = row[f"{id_col}_a"].split(",") if row[f"{id_col}_a"] else []
        ids_b = row[f"{id_col}_b"].split(",") if row[f"{id_col}_b"] else []
        combined = [i for i in (ids_a + ids_b) if i]
        # de-dup while preserving order
        seen, out = set(), []
        for i in combined:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return ",".join(out)

    merged[id_col] = merged.apply(combine, axis=1)
    return merged[["source1_entity_id", id_col]]


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    matching = _merge_id_lists(args.matching_a, args.matching_b, "matched_entity_ids")
    candidates = _merge_id_lists(args.candidates_a, args.candidates_b, "candidate_entity_ids")

    # Ensure every S1 test entity has exactly one row, even if somehow absent
    # from both team outputs (defensive — shouldn't happen if train_and_predict
    # ran correctly, but a missing entity means automatic rejection).
    test_s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    all_s1_ids = set(_load(test_s1_path)["entity_id"])

    matching = matching.set_index("source1_entity_id").reindex(sorted(all_s1_ids)).fillna("")
    matching.index.name = "source1_entity_id"
    matching = matching.reset_index()

    candidates = candidates.set_index("source1_entity_id").reindex(sorted(all_s1_ids)).fillna("")
    candidates.index.name = "source1_entity_id"
    candidates = candidates.reset_index()

    matching_out = os.path.join(args.out_dir, "matching_results.tsv")
    candidates_out = os.path.join(args.out_dir, "candidate_pairs.tsv")
    matching.to_csv(matching_out, sep="\t", index=False)
    candidates.to_csv(candidates_out, sep="\t", index=False)
    print(f"[ok] wrote {matching_out} ({len(matching)} rows)")
    print(f"[ok] wrote {candidates_out} ({len(candidates)} rows)")

    validator = os.path.join("utils", "validate_submission.py")
    if os.path.exists(validator):
        print("\n[running official validator]")
        result = subprocess.run(
            [sys.executable, validator,
             "--matching", matching_out,
             "--candidate", candidates_out,
             "--test-dir", args.test_dir],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            print("[FAILED] fix the issues above before submitting")
            sys.exit(1)
        print("[PASS] safe to submit")
    else:
        print(f"[warn] validator not found at {validator} — run it manually before submitting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching-a", required=True)
    parser.add_argument("--matching-b", required=True)
    parser.add_argument("--candidates-a", required=True)
    parser.add_argument("--candidates-b", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out-dir", default="output")
    run(parser.parse_args())
