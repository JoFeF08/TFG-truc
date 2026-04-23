"""Pool de variants d'AgentRegles per entrenar agents amb memòria (Fase 4).
"""
import random
from RL.models.model_propi.agent_regles import AgentRegles


POOL_OPONENTS = [
    ("conservador", dict(truc_agressio=0.5, envit_agressio=0.5,
                         farol_prob=0.02, resposta_truc=0.6)),
    ("agressiu",    dict(truc_agressio=1.8, envit_agressio=1.8,
                         farol_prob=0.30, resposta_truc=1.5)),
    ("truc_bot",    dict(truc_agressio=2.0, envit_agressio=0.4,
                         farol_prob=0.15, resposta_truc=1.2)),
    ("envit_bot",   dict(truc_agressio=0.4, envit_agressio=2.0,
                         farol_prob=0.05, resposta_truc=0.7)),
    ("faroler",     dict(truc_agressio=1.3, envit_agressio=1.3,
                         farol_prob=0.40, resposta_truc=1.3)),
    ("equilibrat",  dict(truc_agressio=1.0, envit_agressio=1.0,
                         farol_prob=0.12, resposta_truc=1.0)),
]

NOMS_VARIANTS = [n for n, _ in POOL_OPONENTS]


def sample_oponent(rng: random.Random) -> tuple[str, AgentRegles]:
    """Samplea una variant del pool amb random.Random."""
    nom, params = rng.choice(POOL_OPONENTS)
    return nom, AgentRegles(seed=rng.randint(0, 2**31 - 1), **params)


def crear_oponent(nom: str, seed=None) -> AgentRegles:
    """Crea una variant del pool pel seu nom. Útil per avaluació contra cada variant."""
    for n, params in POOL_OPONENTS:
        if n == nom:
            return AgentRegles(seed=seed, **params)
    raise ValueError(f"Variant desconeguda: {nom}. Disponibles: {NOMS_VARIANTS}")
