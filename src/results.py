from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("results")
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {"TF-IDF": "#4C72B0", "Handcrafted": "#DD8452", "Embeddings": "#55A868"}


files = {
    "TF-IDF": "classical_cv.csv",
    "Handcrafted": "handcrafted_cv.csv",
    "Embeddings": "embeddings_cv.csv",
}
frames = []
for familia, fname in files.items():
    d = pd.read_csv(RES / fname)
    d["familia"] = familia
    frames.append(d[["familia", "modelo", "input_variant", "hiperparametros",
                     "fold", "accuracy", "macro_f1"]])
cv = pd.concat(frames, ignore_index=True)


agg = (cv.groupby(["familia", "modelo", "input_variant", "hiperparametros"])
         [["accuracy", "macro_f1"]].agg(["mean", "std"]))
agg.columns = [f"{m}_{s}" for m, s in agg.columns]
agg = agg.reset_index()


best = agg.loc[agg.groupby(["familia", "modelo"])["macro_f1_mean"].idxmax()].copy()
best["label"] = best.apply(
    lambda r: f"{r.modelo} [{r.input_variant}]" if r.familia != "Handcrafted" else r.modelo,
    axis=1,
)
best = best.sort_values("macro_f1_mean", ascending=True).reset_index(drop=True)

print(best[["familia", "modelo", "input_variant", "hiperparametros",
            "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]]
      .round(3).to_string(index=False))
best.round(3).to_csv(RES / "best_per_model.csv", index=False)



def plot_metric(metric, title, filename):
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(best) + 1.5))
    y = np.arange(len(best))
    ax.barh(y, best[f"{metric}_mean"], xerr=best[f"{metric}_std"],
            color=[COLORS[f] for f in best["familia"]], capsize=3)
    for yi, m in zip(y, best[f"{metric}_mean"]):
        ax.text(m + 0.01, yi, f"{m:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(best["label"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"{title} (média ± desvio entre folds)")
    ax.set_title(title)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()],
              COLORS.keys(), loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / f"{filename}.png", dpi=300)
    plt.close(fig)
    print(f"salvo: {FIG / filename}.png")


plot_metric("macro_f1", "Macro-F1", "macro_f1")
plot_metric("accuracy", "Accuracy", "accuracy")