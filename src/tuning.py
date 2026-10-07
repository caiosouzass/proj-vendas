"""Ajuste de hiperparâmetros com validação temporal aninhada e curva de convergência.

O ajuste usa apenas dados anteriores à primeira janela de teste do walk-forward (janelas internas),
de modo que nem o walk-forward externo nem o holdout participam da escolha.
"""
import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmarks import modelo_xgboost
from .modeling import DIAS_HOLDOUT, metricas, prever_modelo

PASSO, N_FOLDS, N_INTERNAS = 14, 4, 3


def fabrica_ridge(alpha):
    return lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def fabrica_lgbm(**p):
    base = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)
    return lambda: LGBMRegressor(**base, **p)


def fabrica_xgb(**p):
    from xgboost import XGBRegressor
    base = dict(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=1, verbosity=0)
    return lambda: XGBRegressor(**base, **p)


GRADES = {
    "Ridge": ({"alpha": [0.1, 1, 10, 30, 100, 300, 1000]}, lambda **p: fabrica_ridge(**p)),
    "LightGBM": ({"num_leaves": [3, 5, 7, 15], "learning_rate": [0.02, 0.05], "n_estimators": [100, 300, 600, 1000],
                  "min_child_samples": [3, 5, 10, 20]}, lambda **p: fabrica_lgbm(**p)),
    "XGBoost": ({"max_depth": [1, 2, 3, 4], "learning_rate": [0.03, 0.1, 0.2], "n_estimators": [100, 300],
                 "min_child_weight": [1, 3, 5, 8]}, lambda **p: fabrica_xgb(**p)),
}


def janelas_externas(n):
    return [(n - PASSO * (N_FOLDS - i), n - PASSO * (N_FOLDS - i - 1)) for i in range(N_FOLDS)]


def buscar(nome, X, yy, n):
    """Grade com janelas internas anteriores à 1ª janela externa. Retorna (melhores parâmetros, tabela)."""
    grade, fab = GRADES[nome]
    ate = janelas_externas(n)[0][0]
    linhas = []
    for valores in itertools.product(*grade.values()):
        p = dict(zip(grade, valores)); make = fab(**p); erros = []
        for i in range(N_INTERNAS):
            ini = ate - PASSO * (N_INTERNAS - i)
            pred, _ = prever_modelo(make, X.iloc[:ini], yy.iloc[:ini], X.iloc[ini:ini + PASSO])
            erros.append(metricas(yy.iloc[ini:ini + PASSO], pred)["WAPE %"])
        linhas.append({**p, "WAPE interno (%)": np.mean(erros)})
    tab = pd.DataFrame(linhas).sort_values("WAPE interno (%)").reset_index(drop=True)
    melhor = {k: (tab.loc[0, k].item() if hasattr(tab.loc[0, k], "item") else tab.loc[0, k]) for k in grade}
    return melhor, tab


def avaliar(make, X, yy, n):
    """WAPE médio do walk-forward externo e métricas do holdout para uma configuração."""
    wf = []
    for a, b in janelas_externas(n):
        pred, _ = prever_modelo(make, X.iloc[:a], yy.iloc[:a], X.iloc[a:b])
        wf.append(metricas(yy.iloc[a:b], pred)["WAPE %"])
    pred, _ = prever_modelo(make, X.iloc[:n], yy.iloc[:n], X.iloc[n:])
    return np.mean(wf), metricas(yy.iloc[n:], pred), pred


def curva_convergencia(X, yy, n, params, n_max=1000):
    """Erro (MAE em log) de treino e validação em função do nº de árvores, na última janela externa."""
    a, b = janelas_externas(n)[-1]
    m = fabrica_lgbm(**{**params, "n_estimators": n_max})()
    m.fit(X.iloc[:a], np.log1p(yy.iloc[:a]), eval_set=[(X.iloc[:a], np.log1p(yy.iloc[:a])), (X.iloc[a:b], np.log1p(yy.iloc[a:b]))],
          eval_names=["treino", "validacao"], eval_metric="l1")
    r = m.evals_result_
    return np.array(r["treino"]["l1"]), np.array(r["validacao"]["l1"])
