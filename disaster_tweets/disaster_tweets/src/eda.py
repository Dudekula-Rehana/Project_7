"""
src/eda.py
Exploratory data analysis. Prints findings and saves figures to reports/figures/.
Run from the project root:   python -m src.eda
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocess import DATA_PATH, ROOT, load_and_clean_data, tokenize

FIG_DIR = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid")
LABELS = {0: "Not disaster", 1: "Disaster"}


def top_words(texts, n=20):
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return c.most_common(n)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    print("=== Structure ===")
    print("Shape:", raw.shape)
    print(raw.dtypes.to_string())
    print("\nMissing values:\n", raw.isna().sum().to_string())
    print("\nDuplicate rows:", raw.duplicated().sum(),
          "| duplicate texts:", raw.duplicated("text").sum())

    # 1) class distribution
    counts = raw["target"].map(LABELS).value_counts()
    print("\nClass distribution:\n", counts.to_string())
    plt.figure(figsize=(5, 4))
    lab = raw["target"].map(LABELS)
    ax = sns.countplot(x=lab, hue=lab, order=["Not disaster", "Disaster"], legend=False,
                       palette=["#2980b9", "#c0392b"])
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.title("Class distribution"); plt.xlabel(""); plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150); plt.close()

    df = load_and_clean_data(verbose=True)
    df["char_len"] = df["text"].str.len()
    df["n_hashtags"] = df["text"].str.count("#")
    df["n_mentions"] = df["text"].str.count("@")
    df["has_url"] = df["text"].str.contains("http").astype(int)
    df["label"] = df["target"].map(LABELS)

    # 2) tweet length by class
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="char_len", hue="label", bins=30, kde=True, element="step")
    plt.title("Tweet length by class"); plt.xlabel("Characters"); plt.tight_layout()
    plt.savefig(FIG_DIR / "length_by_class.png", dpi=150); plt.close()

    # 3) hashtags / mentions / URLs by class
    summary = df.groupby("label")[["char_len", "n_hashtags", "n_mentions", "has_url"]].mean().round(3)
    print("\nMean of simple text features by class:\n", summary.to_string())
    summary.T.plot.bar(figsize=(7, 4), color=["#c0392b", "#2980b9"])
    plt.title("Average of simple text features by class"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(FIG_DIR / "text_features_by_class.png", dpi=150); plt.close()

    # 4) most frequent words per class
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (lab, colour) in zip(axes, [(1, "#c0392b"), (0, "#2980b9")]):
        tw = top_words(df.loc[df["target"] == lab, "text"], 20)
        words, freq = zip(*tw)
        ax.barh(words[::-1], freq[::-1], color=colour)
        ax.set_title(f"Top 20 words - {LABELS[lab]}")
        print(f"\nTop words - {LABELS[lab]}: {[w for w, _ in tw]}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "top_words_by_class.png", dpi=150); plt.close()

    # 5) top bigrams in disaster tweets
    bg = Counter()
    for t in df.loc[df["target"] == 1, "text"]:
        tk = tokenize(t)
        bg.update(f"{a} {b}" for a, b in zip(tk, tk[1:]))
    big = bg.most_common(15)
    plt.figure(figsize=(7, 5))
    plt.barh([b for b, _ in big][::-1], [c for _, c in big][::-1], color="#c0392b")
    plt.title("Top bigrams in disaster tweets"); plt.tight_layout()
    plt.savefig(FIG_DIR / "top_bigrams_disaster.png", dpi=150); plt.close()
    print("\nTop bigrams (disaster):", [b for b, _ in big])

    # 6) keyword column vs target (EDA only - NOT used by the model)
    kw = raw.dropna(subset=["keyword"]).groupby("keyword")["target"].agg(["mean", "count"])
    kw = kw[kw["count"] >= 20].sort_values("mean")
    sel = pd.concat([kw.head(10), kw.tail(10)])
    plt.figure(figsize=(7, 6))
    plt.barh(sel.index, sel["mean"], color=["#2980b9"] * 10 + ["#c0392b"] * 10)
    plt.xlabel("Share of tweets labelled disaster"); plt.title("Keyword column vs target (EDA only)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "keyword_vs_target.png", dpi=150); plt.close()
    print("\nShare labelled disaster when keyword is missing vs present:")
    print(raw.groupby(raw["keyword"].isna())["target"].mean().round(3).to_string())
    print("-> Missing keyword is strongly linked to the target (collection artefact), "
          "so keyword/location are NOT used as model inputs.")
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
