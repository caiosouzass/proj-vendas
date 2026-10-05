"""Estilo padrão dos gráficos (paleta categórica fixa, grade discreta)."""
from pathlib import Path

import matplotlib.pyplot as plt

AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CINZA, TEXTO = "#8a8984", "#2b2b2a"
FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def aplicar_estilo():
    plt.rcParams.update({
        "figure.figsize": (10, 4.2), "figure.dpi": 110, "savefig.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": CINZA, "axes.labelcolor": TEXTO, "text.color": TEXTO,
        "xtick.color": TEXTO, "ytick.color": TEXTO, "axes.grid": True,
        "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=[AZUL, LARANJA, AQUA]),
        "axes.titleweight": "bold", "axes.titlesize": 12, "legend.frameon": False,
        "lines.linewidth": 2,
    })


def salvar(fig, nome):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{nome}.png", bbox_inches="tight", facecolor="white")


def milhoes(x, _=None):
    return f"{x / 1e6:.1f} mi"
