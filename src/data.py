"""Carga e tratamento da base de vendas do e-commerce."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "vendas.csv"

CATEGORIAS_VALIDAS = [
    "CORPO E BANHO",
    "PERFUMARIA MASCULINA",
    "PERF. DE ENTRADA E DEOS",
    "PERFUMARIA FEMININA",
    "GIFTS",
    "MAQUIAGEM",
    "CABELOS",
    "FACIAL",
]
CAT_NAO_IDENT = "NÃO IDENTIFICADA"


def carregar(path: Path = RAW_PATH) -> pd.DataFrame:
    """Lê o CSV, padroniza nomes e marca categorias corrompidas.

    Linhas com categoria numérica (lixo de ETL) são mantidas, pois a receita é
    válida para o total diário; apenas o rótulo vira ``NÃO IDENTIFICADA``.
    """
    df = pd.read_csv(path, parse_dates=["dt_hr_venda"])
    df = df.rename(
        columns={
            "dt_hr_venda": "dt_hora",
            "DES_CANAL_VENDA_FINAL_AGRUP": "canal",
            "DES_CATEGORIA_MATERIAL": "categoria",
        }
    )
    df["categoria_corrompida"] = ~df["categoria"].isin(CATEGORIAS_VALIDAS)
    df.loc[df["categoria_corrompida"], "categoria"] = CAT_NAO_IDENT
    df["data"] = df["dt_hora"].dt.normalize()
    df["hora"] = df["dt_hora"].dt.hour
    df["dia_semana"] = df["dt_hora"].dt.dayofweek
    return df


def serie_diaria(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega a base no nível diário (total, todos canais e categorias)."""
    d = (
        df.groupby("data")[
            ["receita_aprovada", "nr_pedidos", "qt_material", "vlr_venda_desconto"]
        ]
        .sum()
        .asfreq("D")
    )
    return d
