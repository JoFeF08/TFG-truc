"""Mètrica de calibració estratègica del Truc (obrir envit / obrir truc).

Detecta el col·lapse d'envit (degeneració NFSP) que cap mètrica de win-rate veu:
un model degenerat deixa d'OBRIR envit encara que tingui bona mà d'envit, tot i
seguir guanyant l'`envit_bot` per resposta.

Definicions (mesurades sobre la PRIMERA jugada del model quan és mà, ronda 0):
  - calib_envit = P(obrir envit | envit_score ALT) − P(obrir envit | envit_score BAIX)
  - calib_truc  = P(obrir truc  | mà de truc forta)

Validació empírica (referències conegudes):
  best_nash_f6 (sa):      calib_envit ≈ 56 pp
  best_f6      (sa):      calib_envit ≈ 35 pp
  best_rerun   (degen):   calib_envit ≈  4 pp
  best_nash_rerun (degen):calib_envit ≈  2 pp
→ Llindar de no-degeneració: calib_envit ≥ 15 pp (CALIB_ENVIT_MIN).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from joc.entorn.env import TrucEnv
from joc.entorn_ma.game_ma import TrucGameMa
from joc.entorn.rols.player import TrucPlayer
from joc.entorn.rols.judger import TrucJudger
from joc.entorn.cartes_accions import ACTION_SPACE

_AP_E = ACTION_SPACE["apostar_envit"]
_AP_T = ACTION_SPACE["apostar_truc"]

# Llindars (validats sobre models sa vs degenerats)
CALIB_ENVIT_MIN = 15.0   # pp; gate de no-degeneració
CALIB_TRUC_MIN = 40.0    # pp; floor de seguretat (diagnòstic, no és el mode de fallada)

# Buckets (empirical quartiles de 20k samples)
ENVIT_BAIX = 7           # envit_score <= 7   (Q1, 25% més baix)
ENVIT_ALT = 29           # envit_score >= 29  (Q3, 25% més alt)
TRUC_FORTA = 216         # força >= 216 (Q3, 25% més alt)
TRUC_FEBLE = 160         # força < 160 (Q1, 25% més baix)


def _forca_ma(hand) -> int:
    return sum(TrucJudger.get_forca_carta(c) for c in hand)


def mesurar_calibracio(eval_agent: Any, env_config: dict,
                       n_inits: int = 4000, seed: int = 12345) -> dict[str, float]:
    """Mesura calib_envit i calib_truc d'un agent amb interfície `eval_step(state)`.

    Només cal una decisió per mà (la primera jugada quan el model és mà), així que
    és barat: `n_inits` inicialitzacions de partida + ~n_inits/2 inferències.
    """
    env = TrucEnv(config={
        "num_jugadors": env_config.get("num_jugadors", 2),
        "cartes_jugador": env_config.get("cartes_jugador", 3),
        "senyes": env_config.get("senyes", False),
    })
    learner = 0
    pcs = {0: TrucPlayer, 1: TrucPlayer}

    env_obs = []   # P(obrir envit) per bucket
    n_env_baix = n_env_alt = 0
    env_baix = env_alt = 0
    truc_forta_si = truc_forta_n = 0
    truc_feble_si = truc_feble_n = 0

    for i in range(n_inits):
        g = TrucGameMa(2, env_config.get("cartes_jugador", 3),
                       env_config.get("senyes", False), 999, player_class=pcs)
        g.np_random = np.random.RandomState(seed + i)
        g.init_game()
        if g.current_player != learner or g.round_counter != 0 or g.response_state.value != 0:
            continue
        rl_state = env._extract_state(g.get_state(learner))
        accio, _ = eval_agent.eval_step(rl_state)
        accio = int(accio)

        hand = g.players[learner].initial_hand
        sc = TrucJudger.get_envit_ma(hand)
        ft = _forca_ma(hand)

        if sc <= ENVIT_BAIX:
            n_env_baix += 1
            env_baix += (accio == _AP_E)
        elif sc >= ENVIT_ALT:
            n_env_alt += 1
            env_alt += (accio == _AP_E)
        if ft >= TRUC_FORTA:
            truc_forta_n += 1
            truc_forta_si += (accio == _AP_T)
        elif ft < TRUC_FEBLE:
            truc_feble_n += 1
            truc_feble_si += (accio == _AP_T)

    p_env_baix = 100.0 * env_baix / n_env_baix if n_env_baix else float("nan")
    p_env_alt = 100.0 * env_alt / n_env_alt if n_env_alt else float("nan")
    p_truc_forta = 100.0 * truc_forta_si / truc_forta_n if truc_forta_n else float("nan")

    calib_envit = (p_env_alt - p_env_baix) if not (np.isnan(p_env_alt) or np.isnan(p_env_baix)) else float("nan")

    return {
        "calib_envit": calib_envit,
        "calib_truc": p_truc_forta,
        "p_env_baix": p_env_baix,
        "p_env_alt": p_env_alt,
        "n_env_baix": n_env_baix,
        "n_env_alt": n_env_alt,
        "n_truc_forta": truc_forta_n,
    }


def es_no_degenerat(calib: dict[str, float]) -> bool:
    """True si el model manté la calibració d'envit (no ha col·lapsat)."""
    ce = calib.get("calib_envit", float("nan"))
    return (not np.isnan(ce)) and ce >= CALIB_ENVIT_MIN
