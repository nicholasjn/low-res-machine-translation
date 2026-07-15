from pathlib import Path
import re
import unicodedata
import pandas as pd
from datasets import load_dataset

from sacremoses import MosesPunctNormalizer

INPUT_PATHS = [
    # "sentence_pairs_data/synthetic_madlad_bbc_to_id_en_googletrans.jsonl",
    "sentence_pairs_data/synthetic_madlad_bbc_to_en_googletrans.jsonl"
]

OUTPUT_PATH = Path(
    # "sentence_pairs_data/synthetic_madlad_bbc_to_id_en_googletrans_clean.jsonl",
    "sentence_pairs_data/synthetic_madlad_bbc_to_en_googletrans_clean.jsonl"
)

REJECTED_PATH = Path(
    # "sentence_pairs_data/synthetic_madlad_bbc_to_id_en_googletrans_rejected.jsonl",
    "sentence_pairs_data/synthetic_madlad_bbc_to_en_googletrans_rejected.jsonl"
)


EXPECTED_COLUMNS = [
    "src_lang",
    "tgt_lang",
    "source",
    "target",
    "source_dataset",
    "generation_method",
    "madlad400_id",
    "original_row",
]


# ============================================================
# Formal filtering thresholds
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

# Sacremoses normalizer
MOSES_NORMALIZER = MosesPunctNormalizer(lang="en")


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

PLACEHOLDER_RE = re.compile(r"_{3,}")
EXCESSIVE_DOTS_RE = re.compile(r"\.{6,}")
EXCESSIVE_EXCL_QUEST_RE = re.compile(r"[!?]{4,}")
ONLY_SYMBOL_RE = re.compile(r"^\W+$", flags=re.UNICODE)


# ============================================================
# Missing / text helpers
# ============================================================

def is_missing_text(value):
    """
    Robust missing-value check for pandas / pyarrow / JSONL values.

    Handles:
    - None
    - float NaN
    - pd.NA
    - empty or whitespace-only strings
    """
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(value, str) and value.strip() == "":
        return True

    return False


def as_text_or_none(value):
    """
    Convert value to text or None.

    After normalization, text columns should contain only:
    - Python str
    - None
    """
    if is_missing_text(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def as_text(value):
    """
    Convert value to string for statistics.
    Missing values become empty string.
    """
    text = as_text_or_none(value)
    return text if text is not None else ""


# ============================================================
# Text normalization
# ============================================================

def normalize_extra_punctuation(text):
    """
    Extra normalization beyond sacremoses.

    Handles:
    - Japanese-style quotes
    - curly quotes
    - excessive dots
    - bad quote spacing
    - duplicated edge quotes
    """
    if text is None:
        return None

    text = str(text)

    # Normalize quote variants
    text = text.replace("「", '"').replace("」", '"')
    text = text.replace("『", '"').replace("』", '"')
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("«", '"').replace("»", '"')

    # Normalize excessive dots: ...., ...... -> ...
    text = re.sub(r"\.{4,}", "...", text)

    # Normalize spaced ellipsis
    text = re.sub(r"\s*\.\.\.\s*", " ... ", text)

    # Remove duplicated quotes at sentence edges
    # Example: ""Hello." -> "Hello."
    text = re.sub(r'^"+(?=[A-Za-zÀ-ÖØ-öø-ÿ0-9])', '"', text)
    text = re.sub(r'"+$', '"', text)

    # Fix spaces before punctuation
    text = re.sub(r"\s+([,.!?;:%])", r"\1", text)

    # Fix spaces after opening quote/bracket
    text = re.sub(r'(["(\[])\s+', r"\1", text)

    # Fix spaces before closing quote/bracket
    text = re.sub(r'\s+(["\)\]])', r"\1", text)

    # Collapse repeated spaces
    text = MULTISPACE_RE.sub(" ", text).strip()

    return text


def normalize_text(text):
    """
    Main normalization pipeline:
    1. Unicode normalization
    2. Remove control chars and HTML
    3. Sacremoses punctuation normalization
    4. Extra custom punctuation cleanup
    5. Repair simple unbalanced quotes/brackets
    6. Collapse whitespace
    """
    text = as_text_or_none(text)

    if text is None:
        return None

    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    if not text:
        return None

    text = MOSES_NORMALIZER.normalize(text)
    text = normalize_extra_punctuation(text)
    text = repair_unbalanced_pairs(text)

    if not text:
        return None

    return text


# ============================================================
# Statistics helpers
# ============================================================

def token_count(text):
    text = as_text(text)
    if not text:
        return 0
    return len(text.split())


def char_len(text):
    text = as_text(text)
    return len(text)


def safe_ratio(a, b):
    if b == 0:
        return 999999
    return a / b


def punct_ratio(text):
    text = as_text(text)
    if not text:
        return 0.0
    return len(PUNCT_RE.findall(text)) / max(len(text), 1)


def digit_ratio(text):
    text = as_text(text)
    if not text:
        return 0.0
    return len(DIGIT_RE.findall(text)) / max(len(text), 1)


def copy_ratio(source, target):
    """
    Simple overlap ratio between source and target tokens.
    This helps detect untranslated or copied output.
    """
    source = as_text(source)
    target = as_text(target)

    if not source or not target:
        return 0.0

    src_tokens = set(source.lower().split())
    tgt_tokens = set(target.lower().split())

    if not src_tokens or not tgt_tokens:
        return 0.0

    overlap = src_tokens.intersection(tgt_tokens)
    return len(overlap) / min(len(src_tokens), len(tgt_tokens))


# ============================================================
# Extra noisy/incomplete sentence checks
# ============================================================

def repair_unbalanced_pairs(text):
    """
    Try to repair simple unbalanced quotes/brackets instead of rejecting.

    Strategy:
    - If there is an unmatched opening bracket at the beginning, remove it.
    - If there is an unmatched closing bracket at the end, remove it.
    - If quotes are unbalanced, remove extra edge quotes first.
    - This is intentionally conservative: it repairs common noisy web-text cases
      without trying to rewrite sentence structure.
    """
    if not text:
        return text

    text = text.strip()

    # --------------------------------------------------------
    # Repair unbalanced brackets at sentence edges
    # --------------------------------------------------------

    edge_pairs = [
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
    ]

    for open_char, close_char in edge_pairs:
        open_count = text.count(open_char)
        close_count = text.count(close_char)

        # Example: "(This is a sentence." -> "This is a sentence."
        if open_count > close_count and text.startswith(open_char):
            text = text[1:].strip()

        # Example: "This is a sentence.)" -> "This is a sentence."
        open_count = text.count(open_char)
        close_count = text.count(close_char)

        if close_count > open_count and text.endswith(close_char):
            text = text[:-1].strip()

    # --------------------------------------------------------
    # Repair unbalanced double quotes
    # --------------------------------------------------------

    quote_count = text.count('"')

    if quote_count % 2 != 0:
        # Example: '"Hello world.' -> 'Hello world.'
        if text.startswith('"') and not text.endswith('"'):
            text = text[1:].strip()

        # Example: 'Hello world."' where there is only one quote at end
        elif text.endswith('"') and not text.startswith('"'):
            text = text[:-1].strip()

        # If still odd and there are duplicated edge quotes, simplify them
        quote_count = text.count('"')

        if quote_count % 2 != 0:
            text = re.sub(r'^"+', '', text).strip()
            text = re.sub(r'"+$', '', text).strip()

    # --------------------------------------------------------
    # Final whitespace cleanup
    # --------------------------------------------------------

    text = MULTISPACE_RE.sub(" ", text).strip()

    return text


def has_unbalanced_pairs(text):
    """
    Detect likely broken sentence fragments.

    For quotes, we check if the number of double quotes is odd.
    """
    text = as_text(text)

    if not text:
        return True

    bracket_pairs = [
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
    ]

    for open_char, close_char in bracket_pairs:
        if text.count(open_char) != text.count(close_char):
            return True

    # Odd number of double quotes usually means unbalanced quote
    if text.count('"') % 2 != 0:
        return True

    return False


def has_placeholder_or_blanks(text):
    text = as_text(text)

    if not text:
        return False

    if PLACEHOLDER_RE.search(text):
        return True

    return False


def has_extreme_repeated_punctuation(text):
    text = as_text(text)

    if not text:
        return False

    # Very noisy dots before/after normalization
    if EXCESSIVE_DOTS_RE.search(text):
        return True

    # Too many !!! or ????
    if EXCESSIVE_EXCL_QUEST_RE.search(text):
        return True

    return False


def is_symbol_only(text):
    text = as_text(text)

    if not text:
        return True

    return bool(ONLY_SYMBOL_RE.search(text))


def has_enough_letters(text):
    """
    Avoid keeping strings that are mostly numbers/symbols.
    """
    text = as_text(text)

    if not text:
        return False

    letters = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text)

    return len(letters) >= 3


# ============================================================
# Load files
# ============================================================

def load_one_jsonl(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading: {path}")

    ds = load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )

    df = ds.to_pandas()
    df["input_file"] = str(path)

    print(f"  Rows: {len(df):,}")
    return df


def load_all_jsonl(paths):
    dfs = []

    for path in paths:
        dfs.append(load_one_jsonl(path))

    merged = pd.concat(dfs, ignore_index=True)

    print("\nMerged rows before cleaning:")
    print(f"  {len(merged):,}")

    return merged


# ============================================================
# Cleaning with rejection reasons
# ============================================================

def add_rejection_reason(row):
    source = row["source"]
    target = row["target"]

    if is_missing_text(source) or is_missing_text(target):
        return "missing_source_or_target"

    source = as_text(source)
    target = as_text(target)

    if source == "" or target == "":
        return "empty_source_or_target"

    # --------------------------------------------------------
    # Extra custom noise filters
    # --------------------------------------------------------

    if has_placeholder_or_blanks(source):
        return "source_contains_placeholder_blank"

    if has_placeholder_or_blanks(target):
        return "target_contains_placeholder_blank"

    if has_extreme_repeated_punctuation(source):
        return "source_extreme_repeated_punctuation"

    if has_extreme_repeated_punctuation(target):
        return "target_extreme_repeated_punctuation"

    if is_symbol_only(source):
        return "source_symbol_only"

    if is_symbol_only(target):
        return "target_symbol_only"

    if not has_enough_letters(source):
        return "source_too_few_letters"

    if not has_enough_letters(target):
        return "target_too_few_letters"

    if has_unbalanced_pairs(source):
        return "source_unbalanced_after_repair"

    if has_unbalanced_pairs(target):
        return "target_unbalanced_after_repair"

    # --------------------------------------------------------
    # Existing formal filters
    # --------------------------------------------------------

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

    if URL_RE.search(source) or URL_RE.search(target):
        return "contains_url"

    if EMAIL_RE.search(source) or EMAIL_RE.search(target):
        return "contains_email"

    return "keep"


def clean_parallel_corpus(df):
    # Ensure all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Keep expected columns plus input_file
    df = df[EXPECTED_COLUMNS + ["input_file"]].copy()

    # Normalize metadata
    for col in ["src_lang", "tgt_lang", "source_dataset", "generation_method"]:
        df[col] = df[col].astype("string").str.strip()

    # Keep raw text for inspection
    df["raw_source"] = df["source"]
    df["raw_target"] = df["target"]

    # Normalize text columns
    for col in ["source", "target"]:
        df[col] = df[col].map(normalize_text).astype("object")

    # Add length/statistics columns for inspection
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

    df["rejection_reason"] = df.apply(add_rejection_reason, axis=1)

    kept = df[df["rejection_reason"] == "keep"].copy()
    rejected = df[df["rejection_reason"] != "keep"].copy()

    print("\nRejected by heuristic filters:")
    print(rejected["rejection_reason"].value_counts())

    # Deduplication step 1: exact pair deduplication
    before = len(kept)
    kept = kept.drop_duplicates(
        subset=["src_lang", "tgt_lang", "source", "target"],
        keep="first",
    )
    exact_pair_duplicates = before - len(kept)
    print(f"\nRemoved exact duplicate pairs: {exact_pair_duplicates:,}")

    # Deduplication step 2: duplicate source deduplication
    before = len(kept)
    kept = kept.drop_duplicates(
        subset=["src_lang", "tgt_lang", "source"],
        keep="first",
    )
    duplicate_sources = before - len(kept)
    print(f"Removed duplicate sources: {duplicate_sources:,}")

    # Keep only original dataset fields in final clean file
    clean_columns = EXPECTED_COLUMNS
    kept_final = kept[clean_columns].copy()

    return kept_final, rejected


def main():
    merged = load_all_jsonl(INPUT_PATHS)

    print("\nRows by input file:")
    print(merged["input_file"].value_counts())

    clean_df, rejected_df = clean_parallel_corpus(merged)

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
    print(f"  Rows: {len(clean_df):,}")
    print(f"  Saved to: {OUTPUT_PATH}")

    print("\nRejected dataset:")
    print(f"  Rows: {len(rejected_df):,}")
    print(f"  Saved to: {REJECTED_PATH}")

    print("\nLanguage distribution:")
    print(clean_df[["src_lang", "tgt_lang"]].value_counts())

    print("\nGeneration method distribution:")
    print(clean_df["generation_method"].value_counts())

    print("\nSample clean rows:")
    print(clean_df.sample(min(5, len(clean_df)), random_state=42))


if __name__ == "__main__":
    main()