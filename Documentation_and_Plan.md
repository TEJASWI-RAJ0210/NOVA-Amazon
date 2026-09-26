# Amazon ML Challenge 2026 — Business Entity Resolution
## Team Plan, Strategy & Run Guide

Challenge window: 25 Sept 12:00 AM IST → 27 Sept 11:59 PM IST. **We are starting on Day 2 (26 Sept) with ~1.5 days left.** This plan covers today (26th) and tomorrow (27th, submission deadline 11:59 PM).

---

## 1. Strategy — why we split this way

The task is: for every Source 1 (S1) entity, find its matching records in Source 2 (S2) and Source 3 (S3).

Instead of splitting by pipeline *stage* (which forces Team B to wait for Team A's output), we split by **source pair**, so both teams run the full pipeline (clean → block → feature-engineer → model → predict) independently and in parallel:

| Team | Owns | Output |
|---|---|---|
| **Team A** (2 members) | S1 ↔ S2 matching, full pipeline | `output/matching_source2.tsv`, `output/candidates_source2.tsv` |
| **Team B** (2 members) | S1 ↔ S3 matching, full pipeline | `output/matching_source3.tsv`, `output/candidates_source3.tsv` |

Both teams use the **same shared cleaning module** (`clean.py`) so name/address normalization is consistent — build this together in the first hour before splitting.

At the end, a merge script (`merge_and_validate.py`) combines both teams' outputs per `source1_entity_id` into the final `matching_results.tsv` and `candidate_pairs.tsv`, then runs the official `validate_submission.py`.

**Within each team of 2:** don't split cleaning vs. modeling — split by *doing it together on a shared repo*, one person driving blocking+features, the other driving model training+threshold tuning, reviewing each other's code. Both should understand the whole pipeline since you'll need to write one combined methodology doc.

---

## 2. Do we need to clean the data? — Yes, and it matters a lot

Both `business_name` and `business_address` will have:
- Abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited, Rd/Road, St/Street)
- Legal suffix inconsistency (Pvt Ltd vs Private Limited vs LLC vs Inc)
- Punctuation differences (`&` vs `and`), word-order swaps, typos
- Missing address components, landmark references ("Near SBI ATM")
- Transliteration variants

If you don't normalize these, your string-similarity features (Jaccard, Levenshtein, TF-IDF cosine) will be noisy and both blocking recall and model precision will suffer. `clean.py` (below) handles this — **build and agree on this file together before splitting into Team A/B**, since consistent cleaning is what makes your two teams' outputs mergeable and comparable.

Do **not** clean by hardcoding to `{US, India}` only — the test set adds France, and the whole pipeline (cleaning, blocking, features) must degrade gracefully to an unseen country label rather than break or silently drop rows.

---

## 3. Two-Day Timeline

### Day 1 — Today (26 Sept)

| Time | Task | Who |
|---|---|---|
| Hr 0–1 | Environment setup, install deps, load all 4 train files + test files, sanity-check row counts, nulls, duplicate IDs | All 4 |
| Hr 1–2 | **Build `clean.py` together** — agree on normalization rules, test on 20 sample rows from each source, eyeball before/after | All 4 (pair program) |
| Hr 2–3 | EDA: distribution of match counts per S1 entity in ground truth (how many are singletons? how many have 1 match? >1?), country distribution in train vs test, name/address length distributions | All 4, then split |
| Hr 3–6 | **Team A**: build `blocking.py` for S1↔S2 (key-based + TF-IDF nearest-neighbor blocking), measure recall ceiling on train (what fraction of true S2 matches survive blocking?) | Team A |
| Hr 3–6 | **Team B**: build `blocking.py` for S1↔S3 (same approach) | Team B |
| Hr 6–8 | **Team A**: `features.py` — compute similarity features on candidate pairs | Team A |
| Hr 6–8 | **Team B**: `features.py` — same for S1↔S3 | Team B |
| Hr 8–10 | **Team A**: `train_and_predict.py` — build labeled training pairs from ground truth + candidates, train baseline Logistic Regression, check validation F0.5 | Team A |
| Hr 8–10 | **Team B**: same for S1↔S3 | Team B |
| Hr 10 | **Checkpoint**: both teams merge outputs (`merge_and_validate.py`), run `validate_submission.py`, make your **first leaderboard submission** even if score is mediocre — confirms the whole pipeline and format work end-to-end | All 4 |
| Evening | Note down validation F0.5 per team, list top error patterns (false merges vs missed matches) | All 4 |

**Goal by end of Day 1: one valid, submitted `matching_results.tsv` on the leaderboard, however rough.** This de-risks format/pipeline bugs early — never leave your first submission to the last hours.

### Day 2 — Tomorrow (27 Sept, deadline 11:59 PM)

| Time | Task | Who |
|---|---|---|
| Hr 0–2 | Review Day 1 errors. Improve blocking recall if it's the bottleneck (loosen key blocking, increase top-k in nearest neighbor). Improve features if model is the bottleneck (add token-sort ratio, address component overlap, common-prefix length) | Team A / Team B respectively |
| Hr 2–4 | Swap models: try Gradient Boosting / Random Forest instead of Logistic Regression (still tiny, well within the 8B-param / open-license rule — that constraint targets pretrained LLMs, not classical ML). Compare validation F0.5 | Team A / Team B |
| Hr 4–5 | **Threshold tuning**: since F0.5 is precision-heavy (weights precision 2× recall) and macro-averaged including singletons, sweep the decision threshold on your validation split and pick the value that maximizes macro F0.5, not accuracy or F1. When unsure about a borderline pair, lean toward predicting "no match" | Team A / Team B |
| Hr 5–6 | Merge, validate, submit updated leaderboard version. Compare against Day 1 baseline — keep whichever version scores higher | All 4 |
| Hr 6–7 | Sanity pass: check the France rows in test — are they going through the same code path with no crashes / no empty predictions caused by bugs (not by genuine no-match)? | All 4 |
| Hr 7–8.5 | 1–2 more leaderboard iterations if time allows (remember: max 5 submissions/day) | All 4 |
| Hr 8.5–10 | **Freeze the model.** Stop tuning. Write `Documentation_template.md` methodology (see Section 5) | All 4 |
| Hr 10–11 | Assemble final submission zip (see Section 6), run `validate_submission.py` one final time on the exact files going into the zip | All 4 |
| Buffer | Leave at least 1 hour of slack before 11:59 PM for upload issues | — |

---

## 4. Code layout

```
code/
├── clean.py                 # shared normalization — build together first
├── blocking.py               # candidate generation (Team A runs for S2, Team B for S3)
├── features.py                # pairwise similarity features
├── train_and_predict.py       # training pair construction, model, threshold tuning, prediction
└── merge_and_validate.py      # combine Team A + Team B outputs, run official validator
```

Each script is runnable standalone with `python3 script.py` and has a `if __name__ == "__main__"` block — check the top of each file for the exact paths it expects (adjust to your actual `dataset/train/...` and `dataset/test/...` layout).

### Suggested run order (per team, e.g. Team A for S2):
```bash
python3 clean.py                                  # produces cleaned_*.tsv for all sources (run once, shared)
python3 blocking.py --other-source source2         # produces candidates_train_s2.tsv, candidates_test_s2.tsv
python3 features.py --other-source source2         # produces features_train_s2.tsv, features_test_s2.tsv
python3 train_and_predict.py --other-source source2  # trains model, tunes threshold, writes output/matching_source2.tsv, output/candidates_source2.tsv
```
Team B runs the same commands with `--other-source source3`.

### Final merge:
```bash
python3 merge_and_validate.py \
  --matching-a output/matching_source2.tsv --matching-b output/matching_source3.tsv \
  --candidates-a output/candidates_source2.tsv --candidates-b output/candidates_source3.tsv \
  --test-dir dataset/test \
  --out-dir output/
```
This writes the final `output/matching_results.tsv` and `output/candidate_pairs.tsv`, and calls the official `utils/validate_submission.py` automatically — fix anything it flags before uploading.

---

## 5. Methodology document (`Documentation_template.md`) — what to fill in

Since there's no page limit, prioritize clarity over brevity, but keep it tight. Structure to follow (matches what both PDFs ask for):

1. **Methodology overview** — the source-pair split (S1↔S2 by Team A, S1↔S3 by Team B), why (parallelization under time pressure), and how outputs are merged.
2. **Candidate generation / blocking strategy** — describe the key-based blocking (country + sorted-token-prefix) combined with TF-IDF nearest-neighbor blocking; report your measured recall ceiling (what % of true matches survive blocking) — this number matters a lot to reviewers.
3. **Model architecture & feature engineering** — list every feature (name/address Jaccard, Levenshtein ratio, TF-IDF cosine, country match, etc.), the model type (Logistic Regression / Gradient Boosting — note it's classical ML, not an LLM, and confirm license/size compliance is trivially satisfied), and how the threshold was chosen (validation F0.5 maximization).
4. **Precision/recall trade-off discussion** — explain how you tuned specifically for F0.5 (precision-heavy), and how singleton handling was validated.
5. **Limitations & what you'd do with more time** — be honest here; reviewers value this (e.g., "we didn't have time to try a supervised blocking model" or "France generalization was validated only qualitatively").

---

## 6. Final submission zip checklist

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/            # clean.py, blocking.py, features.py, train_and_predict.py, merge_and_validate.py
│       ├── README.md       # exact run commands, in order, from raw data to output
│       └── requirements.txt
└── Documentation_template.md
```

Before zipping:
- [ ] Run `validate_submission.py` on the exact `output/` files going into the zip — must print `PASS`
- [ ] Confirm every test Source 1 entity has exactly one row in `matching_results.tsv`
- [ ] Confirm every matched ID also appears in `candidate_pairs.tsv` for the same S1 entity
- [ ] Confirm no external data / API calls anywhere in the code (grep for `requests.get`, `urllib`, any geocoding/lookup library)
- [ ] Confirm final model is open-license (sklearn/xgboost models are fine) and effectively has no LLM involved — if anyone used an LLM for text cleaning or embeddings, double check it's ≤8B params and MIT/Apache licensed, and document it
- [ ] `requirements.txt` has pinned versions (`pip freeze > requirements.txt` inside your env)
- [ ] `README.md` lets a stranger reproduce both output files from scratch

---

## 7. Compliance reminders (read once, don't violate)

- **No external lookups** — no business registry APIs, no geocoding APIs, no internet-sourced data augmentation. Everything must come from the provided training data. This includes not looking up real company names to "verify" matches.
- **Model constraint**: final model ≤ 8B parameters, MIT/Apache 2.0 licensed. Classical ML (sklearn, XGBoost, LightGBM) trivially satisfies this — no need to reach for an LLM at all for this task; string-similarity + gradient boosting is a strong, fully compliant approach.
- **Country field**: treat as an open string set. Don't filter, hardcode, or one-hot-encode assuming only `{US, India}` — France appears only in test and every French entity must still get a prediction row.
