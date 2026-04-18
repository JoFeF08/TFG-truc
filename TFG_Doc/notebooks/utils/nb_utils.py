import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Arrel del projecte, relativa a qualsevol notebook
PROJECT_ROOT = Path('../../../')

def setup_pyplot(dpi: int = 120, grid_alpha: float = 0.3) -> None:
    plt.rcParams.update({
        'figure.dpi': dpi,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': grid_alpha,
    })


def suavitzar(series: pd.Series, window: int = 5) -> pd.Series:
    """Mitjana mòbil centrada. Els extrems es calculen amb min_periods=1."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def step_first_above(df: pd.DataFrame, threshold: float, col: str = 'eval_metric') -> str:
    """Retorna el primer step (format '3.5M') on `col` supera `threshold`."""
    over = df[df[col] >= threshold]
    if over.empty:
        return '—'
    return f'{over["step"].iloc[0] / 1e6:.1f}M'


def trobar_ultima_carpeta(patro: str, base: 
    Path = PROJECT_ROOT) -> Path | None:
    """Localitza la carpeta més recent que 
    coincideix amb `patro` (glob) relativa a `base`."""
    carpetes = sorted(glob.glob(str(base / patro)), key=os.path.getmtime)
    return Path(carpetes[-1]) if carpetes else None


def carregar_logs(carpeta_base: Path, agents: list,
                  log_name: str = 'training_log.csv') -> dict:
    """Carrega els CSVs de log per a cada agent dins `carpeta_base`."""
    dades = {}
    for agent in agents:
        path = carpeta_base / agent / log_name
        if path.exists():
            dades[agent] = pd.read_csv(path)
    return dades


def llegir_resum_txt(path: Path) -> str:
    """Llegeix i retorna el contingut d'un fitxer resum_*.txt."""
    if path.exists():
        return path.read_text(encoding='utf-8')
    return f'(no trobat: {path})'

# Fase 1 & 2 — 4 agents
COLORS_AGENTS = {
    'dqn_rlcard':  '#e74c3c',
    'nfsp_rlcard': '#3498db',
    'dqn_sb3':     '#2ecc71',
    'ppo_sb3':     '#9b59b6',
}

LABELS_AGENTS = {
    'dqn_rlcard':  'DQN RLCard',
    'nfsp_rlcard': 'NFSP RLCard',
    'dqn_sb3':     'DQN SB3',
    'ppo_sb3':     'PPO SB3',
}

# Fase 3 — variants per a DQN/PPO amb feature extractor preentrenat
COLORS_VARIANTS = {
    'scratch':  '#95a5a6',
    'frozen':   '#3498db',
    'finetune': '#e74c3c',
}

LABELS_VARIANTS = {
    'scratch':  'Scratch (init aleatori)',
    'frozen':   'Frozen (pesos preentrenats)',
    'finetune': 'Finetune (pesos preentrenats)',
}

LSTYLE_VARIANTS = {
    'scratch':  '--',
    'frozen':   '-',
    'finetune': ':',
}

LWIDTH_VARIANTS = {
    'scratch':  1.8,
    'frozen':   2.2,
    'finetune': 2.0,
}
