"""Treino, validação temporal e métricas dos modelos de previsão diária."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import H, montar

DIAS_HOLDOUT = 30


def metricas(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "WAPE %": 100 * np.abs(err).sum() / np.abs(y_true).sum(),
        "MAPE %": 100 * np.mean(np.abs(err) / np.abs(y_true)),
        "Viés %": 100 * err.sum() / y_true.sum(),
    }


def modelos() -> dict:
    return {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=5,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbose=-1,
        ),
    }


def prever_modelo(make, X_tr, y_tr, X_te, log=True):
    yt = np.log1p(y_tr) if log else y_tr
    m = make().fit(X_tr, yt)
    p = m.predict(X_te)
    return (np.expm1(p) if log else p), m


def baseline_sazonal(y: pd.Series, idx, h: int = H) -> np.ndarray:
    """Seasonal naive: mesmo dia da semana, ``ceil(h/7)`` semanas atrás."""
    lag = 7 * int(np.ceil(h / 7))
    return y.shift(lag).loc[idx].values


def walk_forward(y: pd.Series, nome: str, make, h: int = H, n_folds: int = 4,
                 passo: int = 14, fim_treino_min: int = 120, **kw) -> pd.DataFrame:
    """CV de janela expansiva: treina até t, prevê as próximas ``passo`` dias."""
    X = montar(y, h, **kw).dropna()
    yy = y.loc[X.index]
    teste_total = len(yy) - DIAS_HOLDOUT
    inicios = [teste_total - passo * (n_folds - i) for i in range(n_folds)]
    linhas = []
    for k, ini in enumerate(inicios):
        tr, te = slice(0, ini), slice(ini, ini + passo)
        p, _ = prever_modelo(make, X.iloc[tr], yy.iloc[tr], X.iloc[te])
        m = metricas(yy.iloc[te], p)
        linhas.append({"modelo": nome, "fold": k + 1, **m})
    return pd.DataFrame(linhas)
