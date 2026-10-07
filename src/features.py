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
    # dia da semana em variáveis binárias (segunda é a referência): um modelo linear não pode tratar 0..6 como número
    for k, nome in enumerate(["seg", "ter", "qua", "qui", "sex", "sab", "dom"]):
        if k > 0:
            f[f"dow_{nome}"] = (idx.dayofweek == k).astype(int)
    f["fim_semana"] = (idx.dayofweek >= 5).astype(int)
    f["dia_mes"] = idx.day
    f["fim_mes"] = (idx.day >= 28).astype(int)
    f["pos_pagamento"] = idx.day.isin([5, 6, 7]).astype(int)  # dias 20-21 não tiveram suporte nos dados
    f["feriado"] = idx.isin(FERIADOS).astype(int)
    f["beauty_week"] = (idx.month == 11).astype(int)  # campanha que dura o mês de novembro inteiro (pico na Black Friday)
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
    """Lags e médias móveis do alvo, deslocados em ``h`` dias (sem vazamento).

    No início da série, onde falta histórico, médias usam os valores disponíveis (mín. 3) e ``lag_14`` cai para ``lag_7``;
    assim o treino aproveita as primeiras semanas de novembro, em vez de perder 21 dias da campanha.
    """
    f = pd.DataFrame(index=y.index)
    f[f"lag_{h}"] = y.shift(h)
    f[f"lag_{h + 7}"] = y.shift(h + 7).fillna(y.shift(h))
    f["media_7d"] = y.shift(h).rolling(7, min_periods=3).mean()
    f["media_14d"] = y.shift(h).rolling(14, min_periods=3).mean()
    f["mesmo_dow_media"] = pd.concat([y.shift(h), y.shift(h + 7), y.shift(h + 14)], axis=1).mean(axis=1)
    return f


def montar(y: pd.Series, h: int = H, com_lags: bool = True) -> pd.DataFrame:
    X = calendario(y.index)
    if com_lags:
        X = X.join(com_defasagens(y, h))
    return X
