from data import load_train, load_test, get_folds
 
from pandas import concat, Series
 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
 
import pandas as pd
 
from pathlib import Path
 
from sklearn.metrics import (
    accuracy_score,
    f1_score
)
tr = load_train()
te = load_test()
 
df = concat([tr, te], ignore_index=True)
 
print("Dataset size:", len(df))
 
X = (
    df["tema"]
    + " "
    + df["pergunta"]
    + " "
    + df["resposta"]
)
y = df["label"]
 
folds = get_folds(df)
 
pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=5000))
])
 
param_grid_lr = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2),
        (1, 3)
    ],
    "clf__C": [
        0.1,
        1,
        10
    ]
}
 
grid_lr = GridSearchCV(
    estimator=pipe_lr,
    param_grid=param_grid_lr,
    cv=folds,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)
 
grid_lr.fit(X, y)
 
pipe_nb = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])
 
param_grid_nb = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2),
        (1, 3)
    ],
    "clf__alpha": [
        0.1,
        0.5,
        1.0
    ]
}
 
grid_nb = GridSearchCV(
    pipe_nb,
    param_grid_nb,
    cv=folds,
    scoring="f1_macro",
    n_jobs=-1
)
 
grid_nb.fit(X, y)
 
 
pipe_svm = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC())
])
 
param_grid_svm = {
    "tfidf__ngram_range": [
        (1,1),
        (1,2),
        (1,3)
    ],
    "clf__C": [
        0.1,
        1,
        10
    ]
}
 
grid_svm = GridSearchCV(
    pipe_svm,
    param_grid_svm,
    cv=folds,
    scoring="f1_macro",
    n_jobs=-1
)
 
grid_svm.fit(X, y)
 
Path("results").mkdir(exist_ok=True)
 
rows = []
preds = []
 
models = [
    ("Naive Bayes", grid_nb),
    ("Logistic Regression", grid_lr),
    ("Linear SVM", grid_svm)
]
 
for model_name, grid in models:
 
    best_model = grid.best_estimator_
 
    for fold, (tr_idx, va_idx) in enumerate(folds):
 
        X_train = X.iloc[tr_idx]
        X_val = X.iloc[va_idx]
 
        y_train = y.iloc[tr_idx]
        y_val = y.iloc[va_idx]
 
        best_model.fit(X_train, y_train)
 
        y_pred = best_model.predict(X_val)
 
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(
            y_val,
            y_pred,
            average="macro"
        )
 
        rows.append({
            "modelo": model_name,
            "representacao": "tfidf",
            "fold": fold,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "hiperparametros": str(grid.best_params_)
        })
 
        for idx, yt, yp in zip(va_idx, y_val, y_pred):
 
            preds.append({
                "modelo": model_name,
                "representacao": "tfidf",
                "fold": fold,
                "idx": int(idx),
                "y_true": int(yt),
                "y_pred": int(yp),
                "hiperparametros": str(grid.best_params_)
            })
 
classical_results = pd.DataFrame(rows)
classical_preds = pd.DataFrame(preds)
 
classical_results.to_csv(
    "results/classical_cv.csv",
    index=False
)
 
classical_preds.to_csv(
    "results/classical_cv_predictions.csv",
    index=False
)
 
classical_summary = (
    classical_results
    .groupby(["modelo"])
    [["accuracy", "macro_f1"]]
    .agg(["mean", "std"])
    .round(3)
)
 
print("\nCLASSICAL SUMMARY")
print(classical_summary)
 
classical_summary.to_csv(
    "results/classical_summary.csv"
)
 
print(f"Naive Bayes         : {grid_nb.best_score_:.4f}")
print(f"Logistic Regression : {grid_lr.best_score_:.4f}")
print(f"Linear SVM          : {grid_svm.best_score_:.4f}")
 
 
 
QUESTION_WORDS = [
    "quem",
    "qual",
    "quais",
    "como",
    "onde",
    "quando",
    "porque",
    "o que"
]
 
def extract_features(row):
 
    text = f"{row['tema']} {row['pergunta']} {row['resposta']}".lower()
 
    words = text.split()
 
    return Series({
        "length_words": len(words),
 
        "starts_question_word":
            int(
                any(
                    row["pergunta"].lower().startswith(w)
                    for w in QUESTION_WORDS
                )
            ),
 
        "question_marks":
            text.count("?"),
 
        "commas":
            text.count(","),
 
        "colons":
            text.count(":"),
 
        "vocab_richness":
            len(set(words)) / len(words)
            if len(words) > 0 else 0
    })
 
X_features = df.apply(extract_features, axis=1)
 
#print(X_features.head())
 
lr_score = cross_val_score(
LogisticRegression(max_iter=5000),
X_features,
y,
cv=folds,
scoring="f1_macro"
).mean()
 
rf_score = cross_val_score(
    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    X_features,
    y,
    cv=folds,
    scoring="f1_macro"
).mean()
 
svm_score = cross_val_score(
    SVC(),
    X_features,
    y,
    cv=folds,
    scoring="f1_macro"
).mean()
 
handcrafted_results = pd.DataFrame([
    {
        "modelo": "Logistic Regression",
        "representacao": "handcrafted",
        "macro_f1": lr_score
    },
    {
        "modelo": "Random Forest",
        "representacao": "handcrafted",
        "macro_f1": rf_score
    },
    {
        "modelo": "SVM",
        "representacao": "handcrafted",
        "macro_f1": svm_score
    }
])
 
handcrafted_results.to_csv(
    "results/handcrafted_summary.csv",
    index=False
)
 
print("Handcrafted Features")
print(f"Logistic Regression: {lr_score:.4f}")
print(f"Random Forest      : {rf_score:.4f}")
print(f"SVM                : {svm_score:.4f}")
 
handcrafted_models = [
    ("Logistic Regression", LogisticRegression(max_iter=5000)),
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("SVM", SVC())
]
 
hc_rows = []
hc_preds = []
 
for model_name, model in handcrafted_models:
 
    for fold, (tr_idx, va_idx) in enumerate(folds):
 
        y_val = y.iloc[va_idx]
 
        model.fit(X_features.iloc[tr_idx], y.iloc[tr_idx])
        y_pred = model.predict(X_features.iloc[va_idx])
 
        hc_rows.append({
            "modelo": model_name,
            "representacao": "handcrafted",
            "fold": fold,
            "accuracy": accuracy_score(y_val, y_pred),
            "macro_f1": f1_score(y_val, y_pred, average="macro"),
            "hiperparametros": "default"
        })
 
        for idx, yt, yp in zip(va_idx, y_val, y_pred):
            hc_preds.append({
                "modelo": model_name,
                "representacao": "handcrafted",
                "fold": fold,
                "idx": int(idx),
                "y_true": int(yt),
                "y_pred": int(yp),
                "hiperparametros": "default"
            })
 
pd.DataFrame(hc_rows).to_csv("results/handcrafted_cv.csv", index=False)
pd.DataFrame(hc_preds).to_csv("results/handcrafted_cv_predictions.csv", index=False)