"""
src/preprocess.py
Reusable text cleaning, tokenization, feature engineering and data-loading helpers.
The SAME functions are used for training and for the Streamlit app, which
prevents training-vs-deployment preprocessing mismatch.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "twitter_disaster_dataset.csv"

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")
# Own stop-word list. NOTE: scikit-learn's built-in English list contains words such
# as "fire" and "bill", which are meaningful for disaster tweets, so it is NOT used.
# Negations ("no", "not") are deliberately kept.
STOP_WORDS = set("""a about above after again all am an and any are as at be because been
before being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers herself him himself his how i if in into
is it its itself just me more most my myself of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves im ive dont
amp rt via u ur don re ll ve doesn didn isn wasn""".split())

_vader = SentimentIntensityAnalyzer()


def clean_text(text: str) -> str:
    """Lowercase; remove URLs, @mentions, special characters, digits and punctuation.
    The hashtag symbol is dropped but the word is kept (#flood -> flood)."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("&amp;", " ")
    text = _NON_LETTER.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list:
    """Split a cleaned tweet into word tokens; drop stop words and 1-letter tokens."""
    return [t for t in clean_text(text).split() if t not in STOP_WORDS and len(t) > 1]


def analyzer(text: str) -> list:
    """Analyzer for TfidfVectorizer: word tokens + bigrams of those tokens."""
    toks = tokenize(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


EXTRA_FEATURE_NAMES = ["char_len", "word_count", "n_hashtags", "n_mentions",
                       "n_urls", "has_exclaim", "vader_compound"]


def extra_features(texts) -> np.ndarray:
    """Hand-crafted numeric features computed from RAW tweet text."""
    rows = []
    for t in pd.Series(texts).astype(str):
        rows.append([
            len(t),
            len(t.split()),
            t.count("#"),
            len(_MENTION.findall(t)),
            len(_URL.findall(t)),
            int("!" in t),
            _vader.polarity_scores(t)["compound"],
        ])
    return np.array(rows, dtype=float)


def load_and_clean_data(path=DATA_PATH, verbose=True) -> pd.DataFrame:
    """Load the CSV and remove label noise / duplicate tweets.
    - tweets that appear with CONFLICTING labels are dropped (cannot be trusted)
    - remaining exact duplicate texts are reduced to one copy (prevents the same
      tweet landing in both the train and the test set)."""
    df = pd.read_csv(path)
    n0 = len(df)
    nun = df.groupby("text")["target"].transform("nunique")
    df = df[nun == 1]
    n1 = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    if verbose:
        print(f"Rows loaded: {n0} | after dropping conflicting-label tweets: {n1} "
              f"| after removing duplicate texts: {len(df)}")
    return df


def get_train_test(df: pd.DataFrame, test_size=0.2):
    """Stratified split on the RAW text (all fitting happens later, on train only)."""
    return train_test_split(df["text"], df["target"], test_size=test_size,
                            stratify=df["target"], random_state=RANDOM_STATE)
