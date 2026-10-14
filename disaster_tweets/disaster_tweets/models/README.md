# Disaster Tweet Classification (NLP) - with Streamlit app

Classifies a tweet as a **real disaster (1)** or **not (0)**.
Built for the "NLP Project for Disaster Tweet Classification" task (Project_7).
Deployment uses **Streamlit only** (no Flask / Django / cloud server).

## 1. Project structure
```
disaster_tweets/
├── app.py                      # Streamlit web app (UI only)
├── requirements.txt
├── README.md
├── data/twitter_disaster_dataset.csv
├── models/
│   ├── disaster_pipeline.joblib   # saved preprocessing + model pipeline
│   └── metrics.json               # real results written by training
├── notebooks/01_eda_and_modeling.ipynb   # full EDA + modelling notebook (with outputs)
├── reports/                       # comparison tables, classification report, logs
│   └── figures/                   # all charts (PNG)
└── src/
    ├── preprocess.py              # cleaning, tokenisation, features, data loading
    ├── eda.py                     # exploratory analysis + figures
    ├── train.py                   # CV comparison, tuning, final test, saves model
    └── predict.py                 # prediction logic used by the app
```

## 2. Setup (Windows, VS Code)
Open the `disaster_tweets` folder in VS Code, open a terminal (PowerShell) and run:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name disaster_tweets
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again.
Always run commands **from the project root** (the folder containing `app.py`).

## 3. Run the project
| Step | Command | What it does |
|---|---|---|
| EDA | `python -m src.eda` | prints findings, saves charts to `reports/figures/` |
| Train | `python -m src.train` | CV comparison, tuning, test evaluation, saves the model (takes ~5 minutes) |
| App | `python -m streamlit run app.py` | opens the web app at http://localhost:8501 |

A trained model is already included, so you can start the app straight away.
**If the app says the model cannot be loaded** (this can happen if your scikit-learn version differs
from the one used to save it), simply run `python -m src.train` once in your environment.

Notebook: open `notebooks/01_eda_and_modeling.ipynb`, choose the kernel `disaster_tweets`, and *Run All*.

## 4. Method
1. **Data:** 7,613 tweets (the task text mentions ~10,000; the supplied file has 7,613). Columns: `id, keyword, location, text, target`. Target: 57% not-disaster / 43% disaster.
2. **Data quality:** 18 duplicate tweets had conflicting labels (dropped); other duplicate texts reduced to one copy -> 7,485 tweets.
3. **Leakage decision:** a *missing* `keyword` is strongly linked to the target (69% disaster vs 43%), a data-collection artefact unavailable in real use -> `keyword` and `location` are **not** model inputs. The model uses the tweet text only.
4. **Cleaning/tokenisation:** lower-case; remove URLs, @mentions, special characters, digits, punctuation; keep the hashtag word; split into tokens; remove stop words (own list - scikit-learn's list wrongly removes "fire").
5. **Split:** stratified 80/20 (5,988 train / 1,497 test, `random_state=42`).
6. **Features:** TF-IDF on word unigrams + bigrams. Experiment with extra features (length, hashtags, mentions, URLs, exclamation, VADER sentiment) - no clear gain, not used. Word2Vec/GloVe not used (optional in the task).
7. **Models (5-fold stratified CV on training data):** Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest, Logistic Regression + extra features. The vectoriser is inside the Pipeline, so it is fitted on training folds only.
8. **Tuning:** GridSearchCV (scoring = F1) on the two best families.
9. **Final model:** tuned Logistic Regression (rule fixed before testing: highest CV F1, Logistic Regression preferred if within 0.005 - interpretable and gives probabilities). Evaluated **once** on the test set.

## 5. Results (from the actual run; `reports/` holds the full tables)
Cross-validation F1 on training data:

| Model | CV F1 untuned | CV F1 tuned |
|---|---|---|
| Logistic Regression | 0.744 | **0.761** |
| Multinomial Naive Bayes | 0.739 | - |
| Linear SVM | 0.741 | - |
| Random Forest | 0.745 | 0.747 |
| Logistic Regression + extra features | 0.745 | 0.760 |

**Held-out test set (1,497 tweets), final model:**

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.795 | 0.801 | 0.690 | 0.741 | 0.846 |

Confusion matrix (rows = actual, columns = predicted): not-disaster 750 / 109, disaster 198 / 440.

**Overfitting check:** training F1 is 0.98 versus 0.76 (CV) and 0.74 (test): the model partly memorises training tweets. The CV/test numbers are the honest estimates.
Re-running training may change numbers slightly in a different environment.

## 6. Limitations
- Words only: sarcasm, metaphor ("crash and burn") and brand-new events are hard.
- Recall ~69%: about 3 in 10 real disaster tweets are missed - use as a triage aid, not as the only alert source.
- Labels contain some noise; near-duplicate retweets may remain across train/test.
- Feature importance shows associations, not causes.

## 7. Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | run commands from the project root; use `python -m ...` as shown |
| `streamlit` not recognised | use `python -m streamlit run app.py` with the venv active |
| Model load error / version warning | `python -m src.train` |
| Port busy | `python -m streamlit run app.py --server.port 8502` |
| PowerShell blocks activate | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |

## 8. Optional: public link
Push the folder to GitHub and deploy `app.py` on Streamlit Community Cloud (free). Not required for local use.
