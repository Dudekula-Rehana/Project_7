"""
app.py - Streamlit interface for the Disaster Tweet Classifier.
Run from the project root:   python -m streamlit run app.py
The app only LOADS the saved pipeline (models/disaster_pipeline.joblib); no retraining.
"""
import io

import pandas as pd
import streamlit as st

from src.predict import (MAX_CHARS, ModelNotFoundError, load_metrics, load_pipeline,
                         predict_batch, predict_text)
from src.preprocess import ROOT

FIG_DIR = ROOT / "reports" / "figures"

st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="wide")


@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    return load_pipeline()


# ---- Header -----------------------------------------------------------------------
st.title("🚨 Disaster Tweet Classifier")
st.write(
    "Social media is often the first place news of an emergency appears - but words like "
    "*fire* or *crash* are also used as slang and metaphors. This app uses a machine learning "
    "model to estimate whether a tweet is about a **real disaster**."
)

try:
    model = get_model()
except ModelNotFoundError as e:
    st.error(str(e))
    st.info("Open a terminal in the project folder and run:  `python -m src.train`")
    st.stop()
except Exception as e:  # corrupted file, version mismatch, ...
    st.error(f"The model could not be loaded: {e}")
    st.info("Re-create it with `python -m src.train` using the same Python environment you use "
            "to run the app.")
    st.stop()

metrics = load_metrics()

# ---- Sidebar ----------------------------------------------------------------------
with st.sidebar:
    st.header("About the model")
    st.write(
        "**Pipeline:** raw tweet -> text cleaning (URLs, @mentions, punctuation removed) -> "
        "TF-IDF word features (single words + word pairs) -> classifier."
    )
    if metrics:
        st.write(f"**Model:** {metrics['final_model']}")
        tm = metrics["test_metrics"]
        st.write("**Held-out test-set results** (tweets the model never saw in training):")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{tm['accuracy']:.1%}")
        c2.metric("Precision", f"{tm['precision']:.1%}")
        c1.metric("Recall", f"{tm['recall']:.1%}")
        c2.metric("F1-score", f"{tm['f1']:.1%}")
    st.caption(
        "The model only reads the words of the tweet. It can be wrong - especially for sarcasm, "
        "very short tweets or events it has not seen. Do not use it as the only source for "
        "emergency decisions."
    )

tab_single, tab_batch, tab_perf = st.tabs(["🔍 Classify a tweet", "📄 Classify a CSV file", "📊 Model performance"])

# ---- Tab 1: single tweet -------------------------------------------------------------
EXAMPLES = {
    "Wildfire report": "Forest fire near La Ronge Sask. Canada, residents asked to evacuate",
    "Slang use of 'fire'": "That new song is fire, absolutely killing it tonight",
    "Everyday tweet": "Just had a great pizza for lunch with friends",
    "Flood warning": "Heavy rain causes flash flooding of streets, rescue teams deployed",
}


def _set_example(txt):
    st.session_state["tweet"] = txt


with tab_single:
    st.subheader("Paste or type a tweet")
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (label, txt) in zip(cols, EXAMPLES.items()):
        col.button(label, on_click=_set_example, args=(txt,))

    tweet = st.text_area("Tweet text", key="tweet", height=120, max_chars=MAX_CHARS,
                         placeholder="e.g. Earthquake of magnitude 6.1 hits near the coast ...")

    if st.button("Classify tweet", type="primary"):
        try:
            res = predict_text(model, tweet)
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            if res["label"] == 1:
                st.error(f"### 🚨 Likely a DISASTER tweet")
            else:
                st.success(f"### ✅ Likely NOT a disaster tweet")
            if res["probability"] is not None:
                p = res["probability"]
                st.write(f"**Model's estimated probability of 'disaster': {p:.0%}**")
                st.progress(p)
                st.caption(
                    "This is the model's score, not a guarantee. Scores near 50% mean the model is "
                    "unsure."
                )
            if res["pos_words"] or res["neg_words"]:
                st.markdown("**Which words influenced this result?**")
                a, b = st.columns(2)
                with a:
                    st.write("Pushing toward *disaster*:")
                    st.write(", ".join(f"`{w}`" for w, _ in res["pos_words"]) or "-")
                with b:
                    st.write("Pushing toward *not disaster*:")
                    st.write(", ".join(f"`{w}`" for w, _ in res["neg_words"]) or "-")
                st.caption("Words shown are those the model recognises from training; "
                           "'word_word' means a pair of consecutive words.")

# ---- Tab 2: batch ---------------------------------------------------------------------
with tab_batch:
    st.subheader("Classify many tweets at once")
    st.write("Upload a CSV file that has a column named **text**.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read this file as CSV: {e}")
        else:
            if "text" not in df_in.columns:
                st.error(f"No 'text' column found. Columns in your file: {list(df_in.columns)}")
            elif df_in.empty:
                st.warning("The file has no rows.")
            else:
                if len(df_in) > 20000:
                    st.warning("Only the first 20,000 rows are processed.")
                    df_in = df_in.head(20000)
                labels, probs = predict_batch(model, df_in["text"].fillna(""))
                out = df_in.copy()
                out["prediction"] = ["Disaster" if l == 1 else "Not a disaster" for l in labels]
                if probs is not None:
                    out["disaster_probability"] = probs.round(3)
                st.write(f"Classified {len(out)} tweets - "
                         f"{(labels == 1).sum()} flagged as disaster.")
                st.dataframe(out)
                buf = io.StringIO(); out.to_csv(buf, index=False)
                st.download_button("Download results (CSV)", buf.getvalue(),
                                   "predictions.csv", "text/csv")

# ---- Tab 3: performance ---------------------------------------------------------------
with tab_perf:
    st.subheader("How well does the model work?")
    if metrics:
        tm, trm = metrics["test_metrics"], metrics["train_metrics"]
        st.write(
            f"The final model (**{metrics['final_model']}**) was chosen using 5-fold "
            "cross-validation on the training tweets only. It was then evaluated **once** on "
            f"{metrics['n_test']} held-out tweets."
        )
        perf = pd.DataFrame({"Training data": trm, "Held-out test data": tm}).T
        st.dataframe(perf.style.format("{:.3f}"))
        if trm["f1"] - tm["f1"] > 0.1:
            st.info(
                "The training scores are much higher than the test scores: the model partly "
                "memorises training tweets (overfitting), which is common for word-based models "
                "on small datasets. The test-set numbers are the honest estimate."
            )
    for fname, cap in [("class_distribution.png", "Class distribution in the data"),
                       ("confusion_matrix.png", "Confusion matrix (test set)"),
                       ("roc_curve.png", "ROC curve (test set)"),
                       ("precision_recall_curve.png", "Precision-recall curve (test set)"),
                       ("top_features.png", "Words with the strongest influence on the model")]:
        f = FIG_DIR / fname
        if f.exists():
            st.image(str(f), caption=cap, width=700 if "top_features" in fname else 420)
