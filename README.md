# Business Entity Resolution — Reproduction Guide

## Setup
```bash
pip install -r requirements.txt
```

Expected data layout (place under the directory you run these scripts from):
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Step 1 — Clean (run once, shared by both teams)
```bash
python3 clean.py
```
Produces `cleaned_*.tsv` next to each source file.

## Step 2 — Team A: Source1 <-> Source2
```bash
python3 blocking.py --other-source source2
python3 features.py --other-source source2
python3 train_and_predict.py --other-source source2
```

## Step 2 — Team B: Source1 <-> Source3
```bash
python3 blocking.py --other-source source3
python3 features.py --other-source source3
python3 train_and_predict.py --other-source source3
```

## Step 3 — Merge into final submission files
```bash
python3 merge_and_validate.py \
  --matching-a output/matching_source2.tsv --matching-b output/matching_source3.tsv \
  --candidates-a output/candidates_source2.tsv --candidates-b output/candidates_source3.tsv \
  --test-dir dataset/test \
  --out-dir output/
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the
official `utils/validate_submission.py` automatically if it's present in `./utils/`.

## Notes
- All normalization happens in `clean.py` — no external data, APIs, or lookups are used
  anywhere in this pipeline.
- Models used (`LogisticRegression`, `GradientBoostingClassifier` from scikit-learn) are
  classical ML, open-source (BSD-licensed scikit-learn), and trivially within the
  "<=8B parameters, MIT/Apache 2.0" constraint.
- `country` is treated as an open string label throughout — no hardcoded country list,
  so France (test-only) flows through the same code path as US/India.
