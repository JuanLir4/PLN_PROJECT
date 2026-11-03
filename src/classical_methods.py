from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from data import get_folds, load_train
from evaluate import evaluate

Path("results").mkdir(exist_ok=True)


tr = load_train()
y = tr["label"].to_numpy()
folds = list(get_folds(tr))
LABELS = [0, 1]

print("Train size:", len(tr), "| folds:", len(folds))


def run_cv(model, X, y, folds, meta, hp, rows, preds):
    for fold, (tr_idx, va_idx) in enumerate(folds):
        m = clone(model)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        y_pred = m.predict(X.iloc[va_idx])
        s = evaluate(y[va_idx], y_pred, labels=LABELS)

        rows.append({**meta, "fold": fold,
                     "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                     "hiperparametros": hp})
        for i, yt, yp in zip(va_idx, y[va_idx], y_pred):
            preds.append({**meta, "fold": fold, "idx": int(i),
                          "y_true": int(yt), "y_pred": int(yp),
                          "hiperparametros": hp})



p = tr["pergunta"].astype(str)
r = tr["resposta"].astype(str)
t = tr["tema"].astype(str)

VARIANTS = {
    "pergunta": p,
    "tema_pergunta": t + " " + p,
    "resposta": r,
    "pergunta_resposta": p + " " + r,
    "tema_pergunta_resposta": t + " " + p + " " + r, 
}

NGRAMS = [(1, 1), (1, 2), (1, 3)]
MODELS = {
    "Naive Bayes": (MultinomialNB(), {"clf__alpha": [0.1, 0.5, 1.0]}),
    "Logistic Regression": (LogisticRegression(max_iter=5000), {"clf__C": [0.1, 1, 10]}),
    "Linear SVM": (LinearSVC(), {"clf__C": [0.1, 1, 10]}),
}

rows, preds = [], []
for variant, X in VARIANTS.items():
    for name, (clf, clf_grid) in MODELS.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", clf)])
        grid = GridSearchCV(
            pipe,
            {"tfidf__ngram_range": NGRAMS, **clf_grid},
            cv=folds,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid.fit(X, y)
        print(f"[{variant:24s}] {name:20s} best CV macro-F1 = {grid.best_score_:.4f}  {grid.best_params_}")

        meta = {"modelo": name, "representacao": "tfidf",
                "classificador": name, "input_variant": variant}
        run_cv(grid.best_estimator_, X, y, folds, meta,
               str(grid.best_params_), rows, preds)

classical = pd.DataFrame(rows)
classical.to_csv("results/classical_cv.csv", index=False)
pd.DataFrame(preds).to_csv("results/classical_cv_predictions.csv", index=False)

summary = (classical.groupby(["input_variant", "modelo"])[["accuracy", "macro_f1"]]
                    .agg(["mean", "std"]).round(3))
print("\nCLASSICAL SUMMARY")
print(summary)
summary.to_csv("results/classical_summary.csv")


QUESTION_WORDS = ["quem", "qual", "quais", "como", "onde", "quando", "porque", "o que"]


def extract_features(row):
    text = f"{row['tema']} {row['pergunta']} {row['resposta']}".lower()
    words = text.split()
    return pd.Series({
        "length_words": len(words),
        "starts_question_word": int(any(row["pergunta"].lower().startswith(w)
                                        for w in QUESTION_WORDS)),
        "question_marks": text.count("?"),
        "commas": text.count(","),
        "colons": text.count(":"),
        "vocab_richness": len(set(words)) / len(words) if words else 0,
    })


X_feat = tr.apply(extract_features, axis=1)

HC_MODELS = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}

hc_rows, hc_preds = [], []
for name, model in HC_MODELS.items():
    meta = {"modelo": name, "representacao": "handcrafted",
            "classificador": name, "input_variant": "features"}
    run_cv(model, X_feat, y, folds, meta, "default", hc_rows, hc_preds)

hc = pd.DataFrame(hc_rows)
hc.to_csv("results/handcrafted_cv.csv", index=False)
pd.DataFrame(hc_preds).to_csv("results/handcrafted_cv_predictions.csv", index=False)

print("\nHANDCRAFTED SUMMARY")
print(hc.groupby("modelo")[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(3))