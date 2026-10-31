from data import load_train, load_test, get_folds
import pandas as pd

tr, te = load_train(), load_test()
df = pd.concat([tr, te], ignore_index=True)

df["n_words"] = df["pergunta"].str.split().apply(len)

print(df.groupby("metodo")["n_words"].describe())

df["ends_qmark"] = df["pergunta"].str.endswith("?")

print(
    pd.crosstab(
        df["metodo"],
        df["ends_qmark"],
        normalize="index"
    )
)

df["starts_upper"] = (
    df["pergunta"].str[0].str.isupper()
)

print(
    pd.crosstab(
        df["metodo"],
        df["starts_upper"],
        normalize="index"
    )
)

print(
    pd.crosstab(
        df["tema"],
        df["metodo"]
    )
)