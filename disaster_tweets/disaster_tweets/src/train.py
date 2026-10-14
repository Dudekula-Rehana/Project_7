"""
src/train.py
End-to-end training: cross-validated model comparison -> hyper-parameter tuning ->
final selection -> ONE evaluation on the held-out test set -> save pipeline.

Run from the project root:   python -m src.train
"""
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             average_precision_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from src.preprocess import (EXTRA_FEATURE_NAMES, RANDOM_STATE, ROOT, analyzer,
                            extra_features, get_train_test, load_and_clean_data)

warnings.filterwarnings("ignore")
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ----------------------------------------------------------------------------------
# Pipeline builders (raw tweet text in -> prediction out)
# ----------------------------------------------------------------------------------
def build_pipeline(clf, use_extra=False):
    """TF-IDF (word uni+bigrams) [+ scaled hand-crafted features] -> classifier."""
    tfidf = TfidfVectorizer(analyzer=analyzer, min_df=2, sublinear_tf=True)
    if not use_extra:
        return Pipeline([("tfidf", tfidf), ("clf", clf)])
    extra = Pipeline([("calc", FunctionTransformer(extra_features)),
                      ("scale", StandardScaler())])
    union = FeatureUnion([("tfidf", tfidf), ("extra", extra)])
    return Pipeline([("features", union), ("clf", clf)])


def candidate_models():
    """Candidate algorithms, all standard choices for sparse TF-IDF text features."""
    return {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "Multinomial Naive Bayes": build_pipeline(MultinomialNB()),
        "Linear SVM": build_pipeline(LinearSVC(random_state=RANDOM_STATE)),
        "Random Forest": build_pipeline(RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)),
        "Logistic Regression + extra features": build_pipeline(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), use_extra=True),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"tfidf__min_df": [1, 2, 3], "clf__C": [0.5, 1, 2, 5, 10]},
    "Multinomial Naive Bayes": {"tfidf__min_df": [1, 2, 3], "clf__alpha": [0.1, 0.3, 0.5, 1.0]},
    "Linear SVM": {"tfidf__min_df": [1, 2, 3], "clf__C": [0.05, 0.1, 0.3, 0.5, 1, 2]},
    "Random Forest": {"clf__max_depth": [None, 50], "clf__min_samples_leaf": [1, 2]},
}


def cv_scores(model, X, y):
    res = cross_validate(model, X, y, cv=CV, scoring=SCORING,
                         return_train_score=True, n_jobs=-1)
    row = {m: res[f"test_{m}"].mean() for m in SCORING}
    row.update({f"{m}_std": res[f"test_{m}"].std() for m in ["f1"]})
    row["train_f1"] = res["train_f1"].mean()
    return row


def get_scores(model, X):
    """Continuous score for ROC / PR curves."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y):
    pred = model.predict(X)
    score = get_scores(model, X)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score)}


# ----------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------
def plot_cv_comparison(df, title, path):
    d = df.sort_values("f1")
    ax = d[["accuracy", "precision", "recall", "f1"]].plot.barh(figsize=(9, 5), width=0.8)
    ax.set_xlim(0.5, 1.0)
    ax.set_title(title)
    ax.set_xlabel("5-fold CV score (training data only)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_test_results(model, X_test, y_test, name):
    pred = model.predict(X_test)
    score = get_scores(model, X_test)
    # confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not disaster", "Disaster"], yticklabels=["Not disaster", "Disaster"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - test set\n{name}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150); plt.close()
    # ROC
    fpr, tpr, _ = roc_curve(y_test, score)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, score):.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve - test set"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150); plt.close()
    # Precision-Recall
    p, r, _ = precision_recall_curve(y_test, score)
    plt.figure(figsize=(5, 4.5))
    plt.plot(r, p, label=f"AP = {average_precision_score(y_test, score):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-recall curve - test set"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "precision_recall_curve.png", dpi=150); plt.close()
    return cm


def plot_top_features(pipe, n=20):
    """Coefficient plot for linear models (interpretation)."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    if "tfidf" in pipe.named_steps:
        names = list(pipe.named_steps["tfidf"].get_feature_names_out())
    else:
        names = list(pipe.named_steps["features"].transformer_list[0][1].get_feature_names_out())
        names += EXTRA_FEATURE_NAMES
    coef = clf.coef_.ravel()
    order = np.argsort(coef)
    top = pd.DataFrame({"feature": [names[i] for i in order[-n:]], "coef": coef[order[-n:]]})
    bot = pd.DataFrame({"feature": [names[i] for i in order[:n]], "coef": coef[order[:n]]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].barh(top["feature"], top["coef"], color="#c0392b"); axes[0].set_title("Words pushing toward DISASTER")
    axes[1].barh(bot["feature"], bot["coef"], color="#2980b9"); axes[1].set_title("Words pushing toward NOT disaster")
    plt.tight_layout(); plt.savefig(FIG_DIR / "top_features.png", dpi=150); plt.close()
    return top.iloc[::-1]["feature"].tolist(), bot["feature"].tolist()


# ----------------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------------
def main():
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test = get_train_test(df)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  "
          f"(disaster share train {y_train.mean():.3f} / test {y_test.mean():.3f})")

    # 1) Baseline cross-validation of all candidates (training data only)
    print("\n[1/4] 5-fold cross-validation of candidate models ...")
    base_rows = {}
    for name, model in candidate_models().items():
        base_rows[name] = cv_scores(model, X_train, y_train)
        print(f"   {name:40s} F1={base_rows[name]['f1']:.4f}")
    base = pd.DataFrame(base_rows).T
    base.to_csv(REPORT_DIR / "cv_baseline_comparison.csv")
    plot_cv_comparison(base, "Baseline models - 5-fold cross-validation", FIG_DIR / "cv_baseline.png")

    # 2) Tune the two best text-only model families by CV F1
    text_only = base.drop(index="Logistic Regression + extra features")
    top2 = text_only.sort_values("f1", ascending=False).index[:2].tolist()
    print(f"\n[2/4] Hyper-parameter tuning (GridSearchCV, scoring=F1) for: {top2}")
    tuned, tuned_rows, best_params, cv_f1 = {}, {}, {}, {}
    for name in top2:
        gs = GridSearchCV(candidate_models()[name], PARAM_GRIDS[name], cv=CV,
                          scoring="f1", n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        tuned[name] = gs.best_estimator_
        best_params[name] = gs.best_params_
        tuned_rows[f"{name} (tuned)"] = cv_scores(gs.best_estimator_, X_train, y_train)
        cv_f1[name] = tuned_rows[f"{name} (tuned)"]["f1"]
        print(f"   {name}: best params {gs.best_params_}  CV F1 {base.loc[name, 'f1']:.4f} -> "
              f"{tuned_rows[f'{name} (tuned)']['f1']:.4f}")

    # 3) Experiment: do hand-crafted features (length, hashtags, mentions, VADER) help?
    final_candidates = dict(tuned)
    rows = dict(tuned_rows)
    lr_name = "Logistic Regression"
    if lr_name in tuned:
        p = best_params[lr_name]
        lr_extra = build_pipeline(LogisticRegression(C=p["clf__C"], max_iter=2000,
                                                     random_state=RANDOM_STATE), use_extra=True)
        lr_extra.set_params(features__tfidf__min_df=p["tfidf__min_df"])
        lr_extra.fit(X_train, y_train)
        final_candidates["Logistic Regression + extra features"] = lr_extra
        rows["Logistic Regression + extra features (tuned params)"] = cv_scores(lr_extra, X_train, y_train)
        cv_f1["Logistic Regression + extra features"] = rows["Logistic Regression + extra features (tuned params)"]["f1"]
        print(f"   LR + extra features CV F1: "
              f"{rows['Logistic Regression + extra features (tuned params)']['f1']:.4f}")
    tuned_df = pd.DataFrame(rows).T
    tuned_df.to_csv(REPORT_DIR / "cv_tuned_comparison.csv")

    # 4) Final model choice (rule fixed BEFORE looking at the test set)
    best_name = max(cv_f1, key=cv_f1.get)
    lr_keys = [k for k in cv_f1 if k.startswith("Logistic Regression")]
    if lr_keys:
        best_lr = max(lr_keys, key=cv_f1.get)
        if cv_f1[best_lr] >= cv_f1[best_name] - 0.005:   # interpretable + gives probabilities
            best_name = best_lr
    final = final_candidates[best_name]
    print(f"\n[3/4] Selected final model: {best_name}  (CV F1 {cv_f1[best_name]:.4f})")
    print("      Rule: highest CV F1; Logistic Regression preferred if within 0.005 "
          "(interpretable, gives probabilities).")

    # 5) ONE evaluation on the held-out test set
    print("\n[4/4] Evaluating on the held-out test set ...")
    test_m = evaluate(final, X_test, y_test)
    train_m = evaluate(final, X_train, y_train)
    cm = plot_test_results(final, X_test, y_test, best_name)
    top_words = plot_top_features(final)
    report = classification_report(y_test, final.predict(X_test),
                                   target_names=["Not disaster", "Disaster"])
    print(report)
    print("Train vs CV vs Test F1: "
          f"{train_m['f1']:.4f} / {cv_f1[best_name]:.4f} / {test_m['f1']:.4f}")

    # Qualitative robustness check (not a statistic; just a sanity check)
    probes = ["Forest fire near La Ronge Sask. Canada",
              "Heavy rain causes flash flooding of streets, residents evacuated",
              "That new song is fire, absolutely killing it tonight",
              "I'm gonna crash and burn on this exam tomorrow",
              "Just had a great pizza for lunch",
              "Earthquake of magnitude 6.1 hits near the coast, tsunami warning issued"]
    print("\nSanity check on hand-written tweets:")
    for t in probes:
        pr = get_scores(final, [t])[0]
        print(f"   {'DISASTER' if final.predict([t])[0] else 'not disaster':13s} ({pr:.2f})  {t}")

    # Save artefacts
    joblib.dump(final, MODEL_DIR / "disaster_pipeline.joblib")
    metrics = {
        "final_model": best_name,
        "selection_rule": "Highest 5-fold CV F1 on training data; Logistic Regression preferred if within 0.005.",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "cv_f1_selected": float(cv_f1[best_name]),
        "train_metrics": {k: float(v) for k, v in train_m.items()},
        "test_metrics": {k: float(v) for k, v in test_m.items()},
        "confusion_matrix_test": cm.tolist(),
        "best_params": {k: {a: (None if b is None else b) for a, b in v.items()}
                        for k, v in best_params.items()},
        "has_predict_proba": bool(hasattr(final, "predict_proba")),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORT_DIR / "classification_report.txt").write_text(report)
    summary = pd.concat([base.assign(stage="baseline"), tuned_df.assign(stage="tuned")])
    summary.to_csv(REPORT_DIR / "model_comparison_all.csv")
    plot_cv_comparison(tuned_df, "Tuned models - 5-fold cross-validation", FIG_DIR / "cv_tuned.png")
    print(f"\nSaved: models/disaster_pipeline.joblib, models/metrics.json  "
          f"(total time {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
