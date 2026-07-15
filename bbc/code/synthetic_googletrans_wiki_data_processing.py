from pathlib import Path
import re
import unicodedata
import pandas as pd
from datasets import load_dataset

from sacremoses import MosesPunctNormalizer


# ============================================================
# Input / output
# ============================================================

INPUT_PATH = Path(
    "sentence_pairs_data/synthetic_wikipedia_bbc_to_id_en_googletrans.jsonl"
)

OUTPUT_PATH = Path(
    "sentence_pairs_data/synthetic_wikipedia_bbc_to_id_en_googletrans_clean.jsonl"
)

REJECTED_PATH = Path(
    "sentence_pairs_data/synthetic_wikipedia_bbc_to_id_en_googletrans_rejected.jsonl"
)


EXPECTED_COLUMNS = [
    "src_lang",
    "tgt_lang",
    "source",
    "target",
    "source_dataset",
    "generation_method",
    "wiki_id",
    "title",
    "url",
    "sentence_index",
    "original_row",
]


# ============================================================
# Filtering thresholds
# ============================================================

MIN_SOURCE_CHARS = 3
MIN_TARGET_CHARS = 3

MAX_SOURCE_CHARS = 500
MAX_TARGET_CHARS = 500

MIN_SOURCE_TOKENS = 1
MIN_TARGET_TOKENS = 1

MAX_SOURCE_TOKENS = 150
MAX_TARGET_TOKENS = 150

MIN_CHAR_RATIO = 0.25
MAX_CHAR_RATIO = 3.00

MIN_TOKEN_RATIO = 0.25
MAX_TOKEN_RATIO = 3.00

MAX_PUNCT_RATIO = 0.40
MAX_DIGIT_RATIO = 0.50

MAX_COPY_RATIO = 0.90

# ============================================================
# Script / character ratio filtering
# ============================================================

# For Batak Toba written in Latin script, we expect mostly Latin characters.
# If the source contains too much Batak script or other non-Latin script,
# Google Translate often produces noisy hallucinated translations.
MAX_SOURCE_NON_LATIN_LETTER_RATIO = 0.20

# For Indonesian / English targets, almost all letters should be Latin.
MAX_TARGET_NON_LATIN_LETTER_RATIO = 0.20

# ============================================================
# Regex helpers
# ============================================================

MULTISPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
DIGIT_RE = re.compile(r"\d")

WIKI_REF_RE = re.compile(r"\[\d+\]")
WIKI_FILE_RE = re.compile(
    r"\b(?:File|Image|Berkas|Gambar):",
    flags=re.IGNORECASE,
)
WIKI_CATEGORY_RE = re.compile(
    r"\b(?:Category|Kategori):",
    flags=re.IGNORECASE,
)


# ============================================================
# Sacremoses punctuation normalizers
# ============================================================

if MosesPunctNormalizer is not None:
    MOSES_NORMALIZERS = {
        "en": MosesPunctNormalizer(lang="en"),
        "id": MosesPunctNormalizer(lang="id"),
        "bbc": MosesPunctNormalizer(lang="id"),  # fallback: Batak Toba is not supported
        "default": MosesPunctNormalizer(lang="en"),
    }
else:
    MOSES_NORMALIZERS = None


def moses_normalize_punctuation(text, lang=None):
    """
    Normalize punctuation with sacremoses.

    Notes:
    - Sacremoses does not support Batak Toba directly.
    - For bbc, Indonesian normalization is a reasonable fallback because
      the script and punctuation conventions are similar.
    """
    if text is None or pd.isna(text):
        return None

    text = str(text)

    if MOSES_NORMALIZERS is None:
        return text

    normalizer = MOSES_NORMALIZERS.get(lang, MOSES_NORMALIZERS["default"])
    return normalizer.normalize(text)


# ============================================================
# Punctuation / bracket balancing repair
# ============================================================

BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "“": "”",
    "‘": "’",
    '"': '"',
    "'": "'",
}

OPENING_BRACKETS = set(BRACKET_PAIRS.keys())
CLOSING_BRACKETS = set(BRACKET_PAIRS.values())

MIRRORED_CLOSING_TO_OPENING = {
    ")": "(",
    "]": "[",
    "}": "{",
    "”": "“",
    "’": "‘",
}


def strip_unmatched_edge_punctuation(text):
    """
    Repair common unbalanced punctuation caused by sentence extraction.

    Examples:
    - "(This is a sentence."  -> "This is a sentence."
    - "This is a sentence)."  -> "This is a sentence."
    - '"This is a sentence.'  -> "This is a sentence."
    - "This is a sentence\""  -> "This is a sentence"

    This function is conservative:
    - It only removes unmatched punctuation at the sentence edges.
    - It does not try to rewrite punctuation inside the sentence.
    """
    if not text:
        return text

    text = text.strip()

    changed = True
    while changed and text:
        changed = False

        # Remove unmatched leading opening brackets/quotes
        first = text[0]

        if first in ["(", "[", "{", "“", "‘"]:
            expected_close = BRACKET_PAIRS[first]
            if expected_close not in text[1:]:
                text = text[1:].strip()
                changed = True
                continue

        if first in ['"', "'"]:
            if text.count(first) == 1:
                text = text[1:].strip()
                changed = True
                continue

        # Remove unmatched trailing closing brackets/quotes
        last = text[-1]

        if last in [")", "]", "}", "”", "’"]:
            expected_open = MIRRORED_CLOSING_TO_OPENING[last]
            if expected_open not in text[:-1]:
                text = text[:-1].strip()
                changed = True
                continue

        if last in ['"', "'"]:
            if text.count(last) == 1:
                text = text[:-1].strip()
                changed = True
                continue

    return text


def has_unbalanced_pairs(text):
    """
    Detect remaining unbalanced brackets/quotes after repair.

    This is used as a final safety filter.
    """
    if not text:
        return False

    pairs = {
        "(": ")",
        "[": "]",
        "{": "}",
        "“": "”",
        "‘": "’",
    }

    for open_p, close_p in pairs.items():
        if text.count(open_p) != text.count(close_p):
            return True

    # Straight quotes are balanced if they appear an even number of times.
    if text.count('"') % 2 != 0:
        return True

    # Apostrophe is tricky because English contractions use it.
    # So we do NOT reject based on single quote count.
    return False


def repair_punctuation_balance(text):
    """
    Try to fix easy edge-level punctuation imbalance.
    """
    if text is None or pd.isna(text):
        return None

    text = str(text).strip()
    text = strip_unmatched_edge_punctuation(text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    return text

def is_latin_letter(char):
    """
    Return True if a character is a Latin-script letter.

    This includes normal Latin letters and accented Latin letters.
    Examples:
    - a, b, c -> Latin
    - é, ü, ñ -> Latin
    - ᯑ, ᯂ, ᯔ -> non-Latin
    """
    if not char.isalpha():
        return False

    try:
        name = unicodedata.name(char)
    except ValueError:
        return False

    return "LATIN" in name


def is_non_latin_letter(char):
    """
    Return True if a character is alphabetic but not Latin-script.
    """
    return char.isalpha() and not is_latin_letter(char)


def non_latin_letter_ratio(text):
    """
    Ratio of non-Latin letters among all letters.

    We calculate only over alphabetic characters, not punctuation,
    spaces, or numbers.

    Example:
    - "Horas ma hita" -> 0.0
    - "ᯑᯰᯮ ᯤ Horas" -> high ratio
    """
    if not text:
        return 0.0

    letters = [ch for ch in text if ch.isalpha()]

    if not letters:
        return 0.0

    non_latin_letters = [ch for ch in letters if is_non_latin_letter(ch)]

    return len(non_latin_letters) / len(letters)


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text, lang=None):
    """
    Normalize Unicode, remove control characters, remove HTML tags,
    remove common Wikipedia citation markers, normalize punctuation,
    repair edge punctuation imbalance, and collapse whitespace.
    """
    if pd.isna(text):
        return None

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove control chars and HTML tags
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)

    # Remove citation markers like [1], [23]
    text = WIKI_REF_RE.sub(" ", text)

    # Sacremoses punctuation normalization
    text = moses_normalize_punctuation(text, lang=lang)

    # Collapse whitespace before repair
    text = MULTISPACE_RE.sub(" ", text).strip()

    # Repair unbalanced edge punctuation/brackets
    text = repair_punctuation_balance(text)

    # Final whitespace cleanup
    text = MULTISPACE_RE.sub(" ", text).strip()

    return text


# ============================================================
# Utility functions
# ============================================================

def token_count(text):
    if not text:
        return 0
    return len(text.split())


def char_len(text):
    if not text:
        return 0
    return len(text)


def safe_ratio(a, b):
    if b == 0:
        return 999999
    return a / b


def punct_ratio(text):
    if not text:
        return 0.0
    return len(PUNCT_RE.findall(text)) / max(len(text), 1)


def digit_ratio(text):
    if not text:
        return 0.0
    return len(DIGIT_RE.findall(text)) / max(len(text), 1)


def copy_ratio(source, target):
    """
    Simple token-overlap ratio between source and target.
    Helps detect untranslated or copied output.
    """
    if not source or not target:
        return 0.0

    src_tokens = set(source.lower().split())
    tgt_tokens = set(target.lower().split())

    if not src_tokens or not tgt_tokens:
        return 0.0

    overlap = src_tokens.intersection(tgt_tokens)
    return len(overlap) / min(len(src_tokens), len(tgt_tokens))


# ============================================================
# Load dataset
# ============================================================

def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading: {path}")

    dataset = load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )

    df = dataset.to_pandas()

    print(f"Rows loaded: {len(df):,}")
    return df


# ============================================================
# Rejection logic
# ============================================================

def add_rejection_reason(row):
    source = row["source"]
    target = row["target"]

    if pd.isna(source) or pd.isna(target):
        return "missing_source_or_target"

    if source == "" or target == "":
        return "empty_source_or_target"

    source_chars = char_len(source)
    target_chars = char_len(target)

    source_tokens = token_count(source)
    target_tokens = token_count(target)

    if source_chars < MIN_SOURCE_CHARS:
        return "source_too_short"

    if target_chars < MIN_TARGET_CHARS:
        return "target_too_short"

    if source_chars > MAX_SOURCE_CHARS:
        return "source_too_long"

    if target_chars > MAX_TARGET_CHARS:
        return "target_too_long"

    if source_tokens < MIN_SOURCE_TOKENS:
        return "source_too_few_tokens"

    if target_tokens < MIN_TARGET_TOKENS:
        return "target_too_few_tokens"

    if source_tokens > MAX_SOURCE_TOKENS:
        return "source_too_many_tokens"

    if target_tokens > MAX_TARGET_TOKENS:
        return "target_too_many_tokens"

    char_ratio = safe_ratio(target_chars, source_chars)
    token_ratio = safe_ratio(target_tokens, source_tokens)

    if char_ratio < MIN_CHAR_RATIO:
        return "target_source_char_ratio_too_low"

    if char_ratio > MAX_CHAR_RATIO:
        return "target_source_char_ratio_too_high"

    if token_ratio < MIN_TOKEN_RATIO:
        return "target_source_token_ratio_too_low"

    if token_ratio > MAX_TOKEN_RATIO:
        return "target_source_token_ratio_too_high"

    if source == target:
        return "source_same_as_target"

    if copy_ratio(source, target) > MAX_COPY_RATIO:
        return "source_target_high_token_overlap"

    if punct_ratio(source) > MAX_PUNCT_RATIO:
        return "source_too_much_punctuation"

    if punct_ratio(target) > MAX_PUNCT_RATIO:
        return "target_too_much_punctuation"

    if digit_ratio(source) > MAX_DIGIT_RATIO:
        return "source_too_many_digits"

    if digit_ratio(target) > MAX_DIGIT_RATIO:
        return "target_too_many_digits"
    
    if non_latin_letter_ratio(source) > MAX_SOURCE_NON_LATIN_LETTER_RATIO:
        return "source_too_many_non_latin_letters"

    if non_latin_letter_ratio(target) > MAX_TARGET_NON_LATIN_LETTER_RATIO:
        return "target_too_many_non_latin_letters"

    if URL_RE.search(source) or URL_RE.search(target):
        return "contains_url"

    if EMAIL_RE.search(source) or EMAIL_RE.search(target):
        return "contains_email"

    if WIKI_FILE_RE.search(source) or WIKI_FILE_RE.search(target):
        return "contains_wiki_file_marker"

    if WIKI_CATEGORY_RE.search(source) or WIKI_CATEGORY_RE.search(target):
        return "contains_wiki_category_marker"

    # Final check after repair.
    # We only reject if imbalance remains after normalization/repair.
    if has_unbalanced_pairs(source):
        return "source_unbalanced_punctuation"

    if has_unbalanced_pairs(target):
        return "target_unbalanced_punctuation"

    return "keep"


# ============================================================
# Main cleaning function
# ============================================================

def clean_parallel_corpus(df):
    # Ensure all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[EXPECTED_COLUMNS].copy()

    # Normalize metadata fields
    metadata_text_cols = [
        "src_lang",
        "tgt_lang",
        "source_dataset",
        "generation_method",
        "wiki_id",
        "title",
        "url",
    ]

    for col in metadata_text_cols:
        df[col] = df[col].astype("string").str.strip()

    # Normalize main text fields with language-aware sacremoses
    df["source"] = df.apply(
        lambda r: normalize_text(r["source"], lang=r["src_lang"]),
        axis=1,
    )

    df["target"] = df.apply(
        lambda r: normalize_text(r["target"], lang=r["tgt_lang"]),
        axis=1,
    )

    # Convert numeric/index fields safely
    df["sentence_index"] = pd.to_numeric(
        df["sentence_index"],
        errors="coerce",
    )

    df["original_row"] = pd.to_numeric(
        df["original_row"],
        errors="coerce",
    )

    # Add inspection statistics
    df["source_chars"] = df["source"].apply(char_len)
    df["target_chars"] = df["target"].apply(char_len)
    df["source_tokens"] = df["source"].apply(token_count)
    df["target_tokens"] = df["target"].apply(token_count)

    df["char_ratio_tgt_src"] = df.apply(
        lambda r: safe_ratio(r["target_chars"], r["source_chars"]),
        axis=1,
    )

    df["token_ratio_tgt_src"] = df.apply(
        lambda r: safe_ratio(r["target_tokens"], r["source_tokens"]),
        axis=1,
    )

    df["copy_ratio"] = df.apply(
        lambda r: copy_ratio(r["source"], r["target"]),
        axis=1,
    )

    df["source_punct_ratio"] = df["source"].apply(punct_ratio)
    df["target_punct_ratio"] = df["target"].apply(punct_ratio)

    df["source_digit_ratio"] = df["source"].apply(digit_ratio)
    df["target_digit_ratio"] = df["target"].apply(digit_ratio)

    df["source_non_latin_letter_ratio"] = df["source"].apply(
        non_latin_letter_ratio
    )

    df["target_non_latin_letter_ratio"] = df["target"].apply(
        non_latin_letter_ratio
    )

    df["rejection_reason"] = df.apply(add_rejection_reason, axis=1)

    kept = df[df["rejection_reason"] == "keep"].copy()
    rejected = df[df["rejection_reason"] != "keep"].copy()

    print("\nRejected by heuristic filters:")
    print(rejected["rejection_reason"].value_counts())

    # Deduplication 1: exact source-target pair
    before = len(kept)
    kept = kept.drop_duplicates(
        subset=["src_lang", "tgt_lang", "source", "target"],
        keep="first",
    )
    print(f"\nRemoved exact duplicate pairs: {before - len(kept):,}")

    # Deduplication 2: repeated source for same target language
    before = len(kept)
    kept = kept.drop_duplicates(
        subset=["src_lang", "tgt_lang", "source"],
        keep="first",
    )
    print(f"Removed duplicate sources: {before - len(kept):,}")

    kept_final = kept[EXPECTED_COLUMNS].copy()

    return kept_final, rejected


def main():
    if MosesPunctNormalizer is None:
        print(
            "\nWARNING: sacremoses is not installed. "
            "Run: pip install sacremoses\n"
            "The script will continue without Moses punctuation normalization.\n"
        )

    df = load_jsonl(INPUT_PATH)

    print("\nInitial language distribution:")
    print(df[["src_lang", "tgt_lang"]].value_counts())

    clean_df, rejected_df = clean_parallel_corpus(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    rejected_df.to_json(
        REJECTED_PATH,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print("\nFinal clean dataset:")
    print(f"Rows: {len(clean_df):,}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nRejected dataset:")
    print(f"Rows: {len(rejected_df):,}")
    print(f"Saved to: {REJECTED_PATH}")

    print("\nFinal language distribution:")
    print(clean_df[["src_lang", "tgt_lang"]].value_counts())

    print("\nTarget language distribution:")
    print(clean_df["tgt_lang"].value_counts())

    print("\nSample clean rows:")
    print(clean_df.sample(min(5, len(clean_df)), random_state=42))


if __name__ == "__main__":
    main()