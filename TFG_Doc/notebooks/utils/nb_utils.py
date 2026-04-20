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


def carregar_curriculum(carpeta: Path, subfolder: str) -> pd.DataFrame | None:
    """Concatena training_log_mans + training_log_partides amb steps acumulats.
    Retorna None si no existeix training_log_partides.csv."""
    p_mans     = carpeta / subfolder / 'training_log_mans.csv'
    p_partides = carpeta / subfolder / 'training_log_partides.csv'
    if not p_partides.exists():
        return None
    df_p = pd.read_csv(p_partides)
    if p_mans.exists():
        df_m = pd.read_csv(p_mans)
        offset = int(df_m['step'].max()) if len(df_m) > 0 else 0
        df_p = df_p.copy()
        df_p['step'] = df_p['step'] + offset
    return df_p


def carregar_dades_tag(carpeta: Path, agents: list,
                       protocols: list, tag: str) -> dict:
    """Retorna dades[agent][protocol] = DataFrame per a una variant (tag).
    Nom de subcarpeta: {agent}_{protocol}_{tag}.
    Protocol 'curriculum': usa carregar_curriculum (steps acumulats)."""
    dades = {a: {} for a in agents}
    for agent in agents:
        for protocol in protocols:
            sub = f'{agent}_{protocol}_{tag}'
            if protocol == 'curriculum':
                df = carregar_curriculum(carpeta, sub)
            else:
                path = carpeta / sub / 'training_log.csv'
                df = pd.read_csv(path) if path.exists() else None
            if df is not None:
                dades[agent][protocol] = df
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

# Fase 30 — COS vs MLP
COLORS_TAG = {'cos': '#3498db', 'mlp': '#95a5a6'}
LABELS_TAG = {'cos': 'Amb COS', 'mlp': 'Sense COS (MLP)'}
LSTYLE_TAG = {'cos': '-',       'mlp': '--'}
LWIDTH_TAG = {'cos': 2.2,       'mlp': 1.8}
