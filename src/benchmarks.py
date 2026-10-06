"""Modelos clássicos de séries temporais (ETS, SARIMAX, Prophet) para comparação com Ridge/LightGBM.

Todos usam o mesmo protocolo dos demais modelos: para prever o dia D, só conhecem dados até D-H.
"""
import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import BLACK_FRIDAY, CYBER_MONDAY, DIA_DAS_MAES, DIA_NAMORADOS, FERIADOS, H, NATAL, calendario
from .modeling import DIAS_HOLDOUT, metricas, modelos, prever_modelo
from .features import montar

EVENTOS = ["feriado", "bf_dia", "bf_janela", "cyber_monday", "pre_natal", "natal_ano_novo",
           "pre_maes", "pre_namorados", "carnaval"]
N_FOLDS, PASSO = 4, 14


def _silenciar():
    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)


def feriados_prophet() -> pd.DataFrame:
    linhas = [("black_friday", BLACK_FRIDAY, -4, 2), ("cyber_monday", CYBER_MONDAY, 0, 0),
              ("natal_ano_novo", NATAL, -7, 8), ("dia_das_maes", DIA_DAS_MAES, -7, 0),
              ("dia_dos_namorados", DIA_NAMORADOS, -7, 0), ("carnaval", pd.Timestamp("2026-02-16"), -3, 2)]
    linhas += [("feriado", d, 0, 0) for d in FERIADOS]
    return pd.DataFrame(linhas, columns=["holiday", "ds", "lower_window", "upper_window"])


def _ets(ylog, origem, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _silenciar()
    r = ExponentialSmoothing(ylog.iloc[: origem + 1], trend="add", damped_trend=True,
                             seasonal="add", seasonal_periods=7).fit()
    return float(r.forecast(h).iloc[-1])


def _ets_ajustado(ylog, origem, h, alfa, gama):
    """ETS sazonal aditivo com suavização fixa (escolhida para o horizonte de h dias)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _silenciar()
    r = ExponentialSmoothing(ylog.iloc[: origem + 1], trend=None, seasonal="add", seasonal_periods=7,
                             initialization_method="heuristic").fit(smoothing_level=alfa, smoothing_seasonal=gama, optimized=False)
    return float(r.forecast(h).iloc[-1])


def escolher_suavizacao(ylog, ate, h, n_origens=28):
    """Escolhe (alfa, gama) minimizando o erro de h passos nas últimas semanas do treino (sem olhar o teste)."""
    melhor = (np.inf, (0.2, 0.1))
    for alfa, gama in itertools.product([0.05, 0.1, 0.2, 0.3, 0.5], [0.05, 0.1, 0.3, 0.5]):
        erro = np.mean([abs(_ets_ajustado(ylog, o, h, alfa, gama) - ylog.iloc[o + h])
                        for o in range(ate - h - n_origens, ate - h)])
        if erro < melhor[0]: melhor = (erro, (alfa, gama))
    return melhor[1]


def _prophet(ylog, origem, h, feriados):
    from prophet import Prophet
    _silenciar()
    df = pd.DataFrame({"ds": ylog.index[: origem + 1], "y": ylog.values[: origem + 1]})
    m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=True, holidays=feriados)
    m.fit(df)
    return float(m.predict(pd.DataFrame({"ds": [ylog.index[origem + h]]})).yhat.iloc[0])


def _sarimax_fit(ylog, ate, exog, order, sorder):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _silenciar()
    ex = exog.iloc[:ate] if exog is not None else None
    r = SARIMAX(ylog.iloc[:ate], exog=ex, order=order, seasonal_order=sorder, trend="c").fit(disp=False, maxiter=200)
    return r


def escolher_ordem(ylog, ate, exog):
    """Escolhe (p,q,P,Q) por AIC usando apenas dados anteriores à primeira janela de teste."""
    melhor = (np.inf, None)
    for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
        try:
            r = _sarimax_fit(ylog, ate, exog, (p, 0, q), (P, 0, Q, 7))
            if r.aic < melhor[0]: melhor = (r.aic, ((p, 0, q), (P, 0, Q, 7)))
        except Exception:
            continue
    return melhor[1]


def _sarimax(ylog, exog, origem, h, params, order, sorder):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _silenciar()
    ex = exog.iloc[: origem + 1] if exog is not None else None
    r = SARIMAX(ylog.iloc[: origem + 1], exog=ex, order=order, seasonal_order=sorder, trend="c").smooth(params)
    fut = exog.iloc[origem + 1: origem + 1 + h] if exog is not None else None
    return float(r.forecast(h, exog=fut).iloc[-1])


def rodar(y: pd.Series, n_jobs: int = 10, h: int = H):
    """Gera previsões de todos os modelos nos mesmos dias de teste (walk-forward + holdout)."""
    X = montar(y, h).dropna(); yy = y.loc[X.index]
    n = len(yy) - DIAS_HOLDOUT
    janelas = [(n - PASSO * (N_FOLDS - i), n - PASSO * (N_FOLDS - i - 1)) for i in range(N_FOLDS)] + [(n, len(yy))]
    ylog = np.log1p(y)
    ev = calendario(y.index)[EVENTOS].astype(float)
    feriados = feriados_prophet()
    pos = lambda data: y.index.get_loc(data)
    # ordem do SARIMAX escolhida só com dados anteriores à 1ª janela de teste
    ate0 = pos(yy.index[janelas[0][0]])
    ev0 = ev.loc[:, ev.iloc[:ate0].std() > 0]
    ordem = escolher_ordem(ylog, ate0, ev0)
    tarefas, params, colunas, suav = [], {}, {}, {}
    for w, (a, b) in enumerate(janelas):
        ate = pos(yy.index[a])
        cols = ev.columns[ev.iloc[:ate].std() > 0]; colunas[w] = cols
        params[(w, "eventos")] = _sarimax_fit(ylog, ate, ev[cols], *ordem).params.values
        params[(w, "puro")] = _sarimax_fit(ylog, ate, None, *ordem).params.values
        suav[w] = escolher_suavizacao(ylog, ate, h)
        for k in range(a, b):
            tarefas.append((w, k, pos(yy.index[k])))
    def um(w, k, i, nome):
        o = i - h
        if nome == "ETS (auto)": return _ets(ylog, o, h)
        if nome == "ETS (ajustado)": return _ets_ajustado(ylog, o, h, *suav[w])
        if nome == "Prophet": return _prophet(ylog, o, h, feriados)
        if nome == "SARIMAX (eventos)": return _sarimax(ylog, ev[colunas[w]], o, h, params[(w, "eventos")], *ordem)
        return _sarimax(ylog, None, o, h, params[(w, "puro")], *ordem)
    nomes = ["ETS (auto)", "ETS (ajustado)", "SARIMAX", "SARIMAX (eventos)", "Prophet"]
    out = {nm: Parallel(n_jobs=n_jobs)(delayed(um)(w, k, i, nm) for w, k, i in tarefas) for nm in nomes}
    preds = pd.DataFrame({nm: np.expm1(v) for nm, v in out.items()}, index=[yy.index[k] for _, k, _ in tarefas])
    # modelos de ML (cenário A) e baseline nos mesmos dias
    ml = {nm: [] for nm in ["Ridge", "LightGBM"]}
    for a, b in janelas:
        for nm, mk in modelos().items():
            ml[nm] += list(prever_modelo(mk, X.iloc[:a], yy.iloc[:a], X.iloc[a:b])[0])
    preds["Ridge"], preds["LightGBM"] = ml["Ridge"], ml["LightGBM"]
    preds["Média Ridge+LGBM"] = (preds.Ridge + preds.LightGBM) / 2
    preds["Naive sazonal"] = y.shift(7 * int(np.ceil(h / 7))).loc[preds.index].values
    return preds, yy, janelas, n, ordem


def resumir(preds, yy, janelas, n):
    """WAPE médio das janelas do walk-forward e métricas do holdout, por modelo."""
    a0 = janelas[0][0]
    real = yy.loc[preds.index]  # só os dias de teste
    linhas = []
    for nm in preds.columns:
        wf = [metricas(real.iloc[a - a0: b - a0], preds[nm].iloc[a - a0: b - a0])["WAPE %"] for a, b in janelas[:-1]]
        ho = metricas(real.iloc[n - a0:], preds[nm].iloc[n - a0:])
        linhas.append({"modelo": nm, "WAPE walk-forward (%)": np.mean(wf), "WAPE holdout (%)": ho["WAPE %"],
                       "MAPE holdout (%)": ho["MAPE %"], "MAE holdout": ho["MAE"], "Viés holdout (%)": ho["Viés %"]})
    return pd.DataFrame(linhas).set_index("modelo")
