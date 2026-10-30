import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
DATA_DIR = Path('data')

def _load(filename):
    df = pd.read_csv(
        DATA_DIR / filename,
        sep="\t",
        header=0,
        encoding="utf-8-sig",   
        quoting=3,              
    )
    df.columns = ["autor", "tema", "pergunta", "metodo", "resposta"]
    df["label"] = (df["metodo"].str.strip() == "LLM").astype(int) #label to number
    return df

def load_train():
    return _load("dataset_PLN_train.tsv")

def load_test():
    return _load("dataset_PLN_test.tsv")

#creating folds to cross validation:
#divided by autor to prevent overfitting
def get_folds(df, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(sgkf.split(df, df["label"], groups=df["autor"]))