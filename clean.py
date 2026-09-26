"""
clean.py — Shared cleaning & normalization module.

BUILD THIS TOGETHER (both teams) BEFORE splitting into Team A (S1<->S2) and
Team B (S1<->S3). Consistent normalization is what makes your two teams'
outputs comparable and mergeable at the end.

What this does:
  - Normalizes business_name: transliterates Indic-script text to Latin,
    lowercases, strips punctuation, expands common legal/abbreviation forms,
    separates out legal-suffix tokens, tokenizes.
  - Normalizes business_address: transliterates Indic-script text to Latin,
    lowercases, strips punctuation, expands common address abbreviations,
    tokenizes.
  - Leaves `country` as a plain uppercased string (open set — do NOT map to a
    fixed list of countries; France must pass through untouched).
  - Handles missing fields (NaN -> empty string) instead of crashing.

Optional dependency (for Indic-script transliteration):
    pip install indic_transliteration
If not installed, cleaning still runs (Indic tokens are left as-is, i.e.
degrade gracefully instead of crashing) — install the package for correct
Latin-script output.

Run standalone to produce cleaned copies of every source file:
    python3 clean.py

Expected input layout (adjust INPUT_DIR below if yours differs):
    dataset/train/train_source1.tsv
    dataset/train/train_source2.tsv
    dataset/train/train_source3.tsv
    dataset/train/train_ground_truth.tsv
    dataset/test/test_source1.tsv
    dataset/test/test_source2.tsv
    dataset/test/test_source3.tsv

Output: cleaned_<original_filename>.tsv next to each input file, with two
extra columns per text field: `<field>_clean` (normalized string) and
`<field>_tokens` (space-joined sorted unique tokens, useful for Jaccard).
"""

import os
import re
import unicodedata
import functools
import pandas as pd

INPUT_DIR = "dataset"

# --- Optional Indic transliteration -----------------------------------------
try:
    from indic_transliteration import sanscript
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    sanscript = None
    TRANSLITERATION_AVAILABLE = False
    print("[warn] 'indic_transliteration' not installed — Indic-script text "
          "will NOT be transliterated. Run: pip install indic_transliteration")

# Unicode block -> sanscript source scheme
_INDIC_SCRIPT_RANGES = []
if TRANSLITERATION_AVAILABLE:
    _INDIC_SCRIPT_RANGES = [
        (0x0900, 0x097F, sanscript.DEVANAGARI),
        (0x0980, 0x09FF, sanscript.BENGALI),
        (0x0A00, 0x0A7F, sanscript.GURMUKHI),   # Punjabi
        (0x0A80, 0x0AFF, sanscript.GUJARATI),
        (0x0B00, 0x0B7F, sanscript.ORIYA),      # Odia
        (0x0B80, 0x0BFF, sanscript.TAMIL),
        (0x0C00, 0x0C7F, sanscript.TELUGU),
        (0x0C80, 0x0CFF, sanscript.KANNADA),
        (0x0D00, 0x0D7F, sanscript.MALAYALAM),
    ]


def contains_indic_script(text):
    if not text:
        return False

    for ch in str(text):
        if (
            "\u0900" <= ch <= "\u097F" or   # Devanagari
            "\u0D00" <= ch <= "\u0D7F" or   # Malayalam
            "\u0B80" <= ch <= "\u0BFF" or   # Tamil
            "\u0C00" <= ch <= "\u0C7F" or   # Telugu
            "\u0C80" <= ch <= "\u0CFF" or   # Kannada
            "\u0980" <= ch <= "\u09FF" or   # Bengali
            "\u0A80" <= ch <= "\u0AFF" or   # Gujarati
            "\u0A00" <= ch <= "\u0A7F" or   # Punjabi
            "\u0B00" <= ch <= "\u0B7F"      # Odia
        ):
            return True

    return False


def _detect_scheme(ch):
    cp = ord(ch)
    for lo, hi, scheme in _INDIC_SCRIPT_RANGES:
        if lo <= cp <= hi:
            return scheme
    return None


@functools.lru_cache(maxsize=200000)
def _transliterate_token_cached(token, scheme):
    try:
        return sanscript.transliterate(token, scheme, sanscript.ITRANS)
    except Exception:
        return token


def transliterate_indic_text(text: str) -> str:
    """Transliterate any Indic-script tokens in `text` to Latin script
    (phonetic, via indic_transliteration/ITRANS), leaving Latin/numeric
    tokens untouched. No-op if the package isn't installed or text has
    no Indic characters."""
    if not text or not TRANSLITERATION_AVAILABLE or not contains_indic_script(text):
        return text

    out_tokens = []
    for tok in text.split():
        scheme = None
        for ch in tok:
            scheme = _detect_scheme(ch)
            if scheme:
                break
        out_tokens.append(_transliterate_token_cached(tok, scheme) if scheme else tok)
    return " ".join(out_tokens)


# --- Abbreviation expansion dictionaries -----------------------------------
# Keys/values are matched as whole tokens after basic tokenization, so order
# doesn't matter and we don't accidentally rewrite substrings inside other
# words (e.g. "st" only expands when it's its own token, not inside "east").

NAME_SUFFIX_EXPANSIONS = {
    "pvt": "private",
    "ltd": "limited",
    "co": "company",
    "corp": "corporation",
    "inc": "incorporated",
    "llp": "llp",          # keep as-is, distinct legal form
    "llc": "llc",
    "plc": "plc",
    "assoc": "associates",
    "intl": "international",
    "grp": "group",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "service",
    "&": "and",
}

# LEGAL_FORM_NORMALIZATION = {
#     "l l p": "llp",
#     "l l c": "llc",
#     "p l c": "plc",
#     "p l l c": "pllc",
# }

# Suffix tokens that, once expanded, we may want to strip out entirely to
# get a "core name" for stricter matching. Kept as a separate helper so you
# can experiment with using core_name vs full cleaned name as a feature.
LEGAL_SUFFIX_TOKENS = {
    "private", "limited", "company", "corporation", "incorporated",
    "llp", "llc", "plc", "group", "co",
    #"associates", "pllc"
}

ADDRESS_ABBREV_EXPANSIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "apt": "apartment",
    "flr": "floor",
    "fl": "floor",
    "bldg": "building",
    "no": "number",
    "nr": "near",
    "opp": "opposite",
    "sec": "sector",
    "colony": "colony",
    "twp": "township",
    "hwy": "highway",
    "pkwy": "parkway",
    "sq": "square",
}

# Pre-punctuation-strip regex fixes for addresses (must run while separators
# like '/', '.' are still present).
ADDRESS_PRE_SUBS = [
    (re.compile(r"\bc\s*/\s*o\b"), "care of"),
    (re.compile(r"\bh\s*\.?\s*no\b"), "house number"),
]

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Normalize transliteration variants (e.g. accented chars) to ASCII."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _basic_normalize(text, pre_punct_subs=None) -> str:
    """Transliterate Indic script, lowercase, strip accents, remove
    punctuation, collapse whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = transliterate_indic_text(text)
    text = _strip_accents(text)
    text = text.lower()
    if pre_punct_subs:
        for pattern, repl in pre_punct_subs:
            text = pattern.sub(repl, text)
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def _expand_tokens(text: str, expansion_map: dict) -> str:
    tokens = text.split()
    expanded = [expansion_map.get(tok, tok) for tok in tokens]
    return " ".join(expanded)


def clean_business_name(raw_name) -> str:
    """Full cleaning pipeline for a business name string."""
    text = _basic_normalize(raw_name)

    # Normalize spaced legal abbreviations created by punctuation removal
    text = re.sub(r"\bl\s+l\s+p\b", "llp", text)
    text = re.sub(r"\bl\s+l\s+c\b", "llc", text)
    text = re.sub(r"\bp\s+l\s+c\b", "plc", text)
    text = re.sub(r"\bp\s+l\s+l\s+c\b", "pllc", text)

    text = _expand_tokens(text, NAME_SUFFIX_EXPANSIONS)
    return text


def clean_business_name_core(cleaned_name: str) -> str:
    """Drop legal-suffix tokens to get a stricter 'core' name for matching."""
    tokens = [t for t in cleaned_name.split() if t not in LEGAL_SUFFIX_TOKENS]
    return " ".join(tokens)


def clean_address(raw_address) -> str:
    """Full cleaning pipeline for an address string."""
    text = _basic_normalize(raw_address, pre_punct_subs=ADDRESS_PRE_SUBS)
    text = _expand_tokens(text, ADDRESS_ABBREV_EXPANSIONS)
    return text


def clean_country(raw_country) -> str:
    """
    Normalize country as an OPEN string label. Do not map to a fixed set —
    the test set contains France, which never appears in training.
    """
    if raw_country is None or (isinstance(raw_country, float) and pd.isna(raw_country)):
        return ""
    return str(raw_country).strip().upper()


def tokens_sorted_unique(text: str) -> str:
    """Space-joined sorted unique tokens — handy for Jaccard / blocking keys."""
    return " ".join(sorted(set(text.split())))


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning to a source dataframe with business_name, business_address,
    country columns. Returns a copy with added *_clean, *_core, *_tokens cols.
    Safe against missing columns/values.
    """
    df = df.copy()

    if "business_name" in df.columns:
        df["business_name_clean"] = df["business_name"].apply(clean_business_name)
        df["business_name_core"] = df["business_name_clean"].apply(clean_business_name_core)
        df["business_name_tokens"] = df["business_name_clean"].apply(tokens_sorted_unique)

    if "business_address" in df.columns:
        df["business_address_clean"] = df["business_address"].apply(clean_address)
        df["business_address_tokens"] = df["business_address_clean"].apply(tokens_sorted_unique)

    if "country" in df.columns:
        df["country_clean"] = df["country"].apply(clean_country)

    return df


def _load_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])


def main():
    files = [
        os.path.join(INPUT_DIR, "train", "train_source1.tsv"),
        os.path.join(INPUT_DIR, "train", "train_source2.tsv"),
        os.path.join(INPUT_DIR, "train", "train_source3.tsv"),
        os.path.join(INPUT_DIR, "test", "test_source1.tsv"),
        os.path.join(INPUT_DIR, "test", "test_source2.tsv"),
        os.path.join(INPUT_DIR, "test", "test_source3.tsv"),
    ]

    for path in files:
        if not os.path.exists(path):
            print(f"[skip] {path} not found — adjust INPUT_DIR if your layout differs")
            continue
        df = _load_tsv(path)
        cleaned = clean_dataframe(df)
        out_path = os.path.join(os.path.dirname(path), "cleaned_" + os.path.basename(path))
        cleaned.to_csv(out_path, sep="\t", index=False)
        print(f"[ok] {path} -> {out_path} ({len(cleaned)} rows)")


if __name__ == "__main__":
    main()