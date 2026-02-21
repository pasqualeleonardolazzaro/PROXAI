import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# 1. Caricamento dati
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# 2. Modello base
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
y_proba = model.predict_proba(X)
loss_full = log_loss(y, y_proba)

print("2")

# 3. Calcolo approssimato delle influence scores (leave-one-out)
influences = []

for i in range(len(X)):
    print(i)
    X_loo = X.drop(index=i)
    y_loo = y.drop(index=i)

    model_loo = LogisticRegression(max_iter=10000)
    model_loo.fit(X_loo, y_loo)
    y_proba_loo = model_loo.predict_proba(X)
    loss_loo = log_loss(y, y_proba_loo)

    influences.append(loss_loo - loss_full)

# 4. Analisi
df = X.copy()
df['target'] = y
df['influence'] = influences
df_sorted = df.sort_values('influence', ascending=False)

print(df_sorted[['influence', 'target']].head())

# 5. Visualizzazione
plt.figure(figsize=(10, 5))
plt.hist(df['influence'], bins=30)
plt.title('Distribuzione delle Influence Scores')
plt.xlabel('Influenza stimata (delta log loss)')
plt.ylabel('Numero di punti')
plt.grid(True)
plt.show()
