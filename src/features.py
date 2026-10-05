"""Engenharia de features para a previsão diária de vendas."""
import numpy as np
import pandas as pd

# Feriados nacionais e datas comerciais relevantes no período da base.
FERIADOS = pd.to_datetime(
    [
        "2025-11-02", "2025-11-15", "2025-11-20", "2025-12-25", "2026-01-01",
        "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21", "2026-05-01",
        "2026-06-04",
    ]
)
DIA_DAS_MAES = pd.Timestamp("2026-05-10")
DIA_NAMORADOS = pd.Timestamp("2026-06-12")
BLACK_FRIDAY = pd.Timestamp("2025-11-28")
CYBER_MONDAY = pd.Timestamp("2025-12-01")
NATAL = pd.Timestamp("2025-12-25")

# Horizonte mínimo de previsão: lags e janelas só usam dados com >= H dias.
H = 7


def calendario(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Features que dependem apenas da data (conhecidas no futuro)."""
    f = pd.DataFrame(index=idx)
    f["dow"] = idx.dayofweek
    f["fim_semana"] = (idx.dayofweek >= 5).astype(int)
    f["dia_mes"] = idx.day
    f["fim_mes"] = (idx.day >= 28).astype(int)
    f["pos_pagamento"] = idx.day.isin([5, 6, 7, 20, 21]).astype(int)
    f["feriado"] = idx.isin(FERIADOS).astype(int)
    f["bf_dia"] = (idx == BLACK_FRIDAY).astype(int)
    f["bf_janela"] = ((idx >= BLACK_FRIDAY - pd.Timedelta(days=4)) & (idx <= BLACK_FRIDAY + pd.Timedelta(days=2))).astype(int)
    f["cyber_monday"] = (idx == CYBER_MONDAY).astype(int)
    f["pre_natal"] = ((idx >= NATAL - pd.Timedelta(days=7)) & (idx < NATAL)).astype(int)
    f["natal_ano_novo"] = ((idx >= NATAL) & (idx <= pd.Timestamp("2026-01-02"))).astype(int)
    f["pre_maes"] = ((idx >= DIA_DAS_MAES - pd.Timedelta(days=7)) & (idx <= DIA_DAS_MAES)).astype(int)
    f["pre_namorados"] = ((idx >= DIA_NAMORADOS - pd.Timedelta(days=7)) & (idx <= DIA_NAMORADOS)).astype(int)
    f["carnaval"] = ((idx >= "2026-02-13") & (idx <= "2026-02-18")).astype(int)
    f["t"] = np.arange(len(idx))
    return f


def com_defasagens(y: pd.Series, h: int = H) -> pd.DataFrame:
    """Lags e médias móveis do alvo, deslocados em ``h`` dias (sem vazamento)."""
    f = pd.DataFrame(index=y.index)
    f[f"lag_{h}"] = y.shift(h)
    f[f"lag_{h + 7}"] = y.shift(h + 7)
    f["media_7d"] = y.shift(h).rolling(7).mean()
    f["media_14d"] = y.shift(h).rolling(14).mean()
    f["mesmo_dow_media"] = (y.shift(h) + y.shift(h + 7) + y.shift(h + 14)) / 3
    return f


def montar(y: pd.Series, h: int = H, com_lags: bool = True) -> pd.DataFrame:
    X = calendario(y.index)
    if com_lags:
        X = X.join(com_defasagens(y, h))
    return X
