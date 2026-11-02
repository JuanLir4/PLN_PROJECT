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

print("Handcrafted Features")
print(f"Logistic Regression: {lr_score:.4f}")
print(f"Random Forest      : {rf_score:.4f}")
print(f"SVM                : {svm_score:.4f}")