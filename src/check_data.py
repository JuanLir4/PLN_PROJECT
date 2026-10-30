from data import load_train, load_test, get_folds

tr, te = load_train(), load_test()
print("Treino:", tr.shape, "| Teste:", te.shape)
print("\nMétodo no treino:\n", tr["metodo"].value_counts())
print("\nMétodo no teste:\n", te["metodo"].value_counts())
print("\nTemas treino/teste:", tr["tema"].nunique(), "/", te["tema"].nunique())
print("Valores únicos de método:", tr["metodo"].unique())

print("\nAutores no treino:", tr["autor"].nunique())
print("Autores no teste:", te["autor"].nunique())
print("Autores comuns treino/teste:", len(set(tr["autor"]) & set(te["autor"])))
print("\nPerguntas por autor (treino):")
print(tr["autor"].value_counts().describe())
print("\nAmostra de ids de autor:", tr["autor"].unique()[:8])

print("\nFolds:")
for i, (a, b) in enumerate(get_folds(tr)):
    comuns = set(tr.iloc[a]["autor"]) & set(tr.iloc[b]["autor"])
    print(f"fold {i}: treino={len(a)}, val={len(b)}, "
          f"LLM na val={tr.iloc[b]['label'].mean():.2f}, "
          f"autores partilhados treino/val={len(comuns)}")

print("\n", tr[["tema", "pergunta", "metodo"]].head())