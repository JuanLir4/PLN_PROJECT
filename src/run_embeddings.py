from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_train, load_test, get_folds
from evaluate import evaluate  

MODELS = {
    "e5": ("intfloat/multilingual-e5-base", "query: "),
    "bertimbau": ("neuralmind/bert-base-portuguese-cased", ""),
}
VARIANTS = ["pergunta", "tema_pergunta", "resposta", "pergunta_resposta", "tema_pergunta_resposta"]
C_VALUES = [0.001, 0.01, 0.1]

tr, te = load_train(), load_test()
df_all = pd.concat([tr, te], ignore_index=True)
y = tr["label"].to_numpy()
folds = get_folds(tr)

rows, preds = [], []
for model_name, (hf_id, prefix) in MODELS.items():
    encoder = None
    for variant in VARIANTS:
        # texto de entrada
        p = df_all["pergunta"].astype(str)
        r = df_all["resposta"].astype(str)
        t = df_all["tema"].astype(str)
        
        texts = {
            "pergunta": p,
            "tema_pergunta": t + " " + p,
            "resposta": r,
            "pergunta_resposta": p + " " + r,
            "tema_pergunta_resposta": t + " " + p + " " + r,
        }[variant]

        
        path = Path("embeddings") / f"{model_name}_{variant}.npy"
        if path.exists():
            emb = np.load(path)
        else:
            if encoder is None:
                encoder = SentenceTransformer(hf_id)
            emb = encoder.encode([prefix + t for t in texts], batch_size=32, show_progress_bar=True)
            np.save(path, emb)
        X = emb[: len(tr)]

        for C in C_VALUES:
            for fold, (tr_idx, va_idx) in enumerate(folds):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
                clf.fit(X[tr_idx], y[tr_idx])
                y_pred = clf.predict(X[va_idx])
                m = evaluate(y[va_idx], y_pred, labels=[0, 1])

                rows.append({
                    "modelo": model_name,
                    "representacao": "embedding",
                    "classificador": "logreg",
                    "input_variant": variant,
                    "fold": fold,
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "hiperparametros": f"C={C}",
                })
                for i, yt, yp in zip(va_idx, y[va_idx], y_pred):
                    preds.append({"modelo": model_name, "input_variant": variant,
                                  "hiperparametros": f"C={C}", "fold": fold,
                                  "idx": int(i), "y_true": int(yt), "y_pred": int(yp)})

res = pd.DataFrame(rows)
res.to_csv("results/embeddings_cv.csv", index=False)
pd.DataFrame(preds).to_csv("results/embeddings_cv_predictions.csv", index=False)

summary = (res.groupby(["modelo", "input_variant", "hiperparametros"])[["accuracy", "macro_f1"]]
              .agg(["mean", "std"]).round(3))
print(summary)