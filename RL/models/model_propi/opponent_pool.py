"""Pool d'oponents per a l'entrenament amb currículum i self-play amb lliga.

Mostreja, a cada `reset()` del wrapper d'entrenament, un oponent entre:
acció aleatòria, `AgentRegles` amb paràmetres variats (més o menys
agressiu/farol), o un checkpoint de política congelada carregat des d'un
directori de pool (self-play amb versions anteriors de l'aprenent).
Generalitza el patró d'oponent fix que ja hi havia a `TrucGymEnvMa`.
"""
import random
from pathlib import Path
from typing import Optional

from joc.entorn.cartes_accions import ACTION_LIST
from RL.models.model_propi.random_agent import RandomAgent
from RL.models.model_propi.agent_regles import AgentRegles, ESTILS

N_ACTIONS = len(ACTION_LIST)

# Pes de mostreig entre estils d'AgentRegles. No es uniforme a proposit:
# tot i el calibratge, 'agressiu'/'farol' segueixen guanyant una mica mes
# sovint en enfrontaments regles-vs-regles (~52-55% de mitjana). Per no
# esbiaixar l'exposicio d'entrenament cap a "blofejar sempre", es baixa
# el seu pes de mostreig i es puja el de 'conservador'/'equilibrat'.
PESOS_ESTIL = {'conservador': 0.30, 'equilibrat': 0.30, 'agressiu': 0.22, 'farol': 0.18}


def _marcador_aleatori(rng: random.Random) -> tuple:
    """Mostreja un escenari de marcador (propi, rival) variat perquè la
    lògica adaptativa d'AgentRegles s'exerciti durant l'entrenament (les
    mans d'entrenament sempre comencen 0-0 i mai ho reflectirien soles)."""
    escenari = rng.choices(
        ['empat', 'avantatge', 'desavantatge', 'final'],
        weights=[0.3, 0.25, 0.25, 0.2], k=1,
    )[0]
    if escenari == 'empat':
        p = rng.randint(0, 18)
        return (p, p)
    if escenari == 'avantatge':
        propi = rng.randint(6, 23)
        return (propi, max(0, propi - rng.randint(3, 12)))
    if escenari == 'desavantatge':
        rival = rng.randint(6, 23)
        return (max(0, rival - rng.randint(3, 12)), rival)
    # final: algú a prop de guanyar (24)
    if rng.random() < 0.5:
        return (rng.randint(18, 23), rng.randint(0, 20))
    return (rng.randint(0, 20), rng.randint(18, 23))


class OpponentPool:
    """Mostreja oponents `eval_step()`-compatibles per a l'entrenament.

    `pesos` controla la proporció de mostreig entre 'random', 'regles' i
    'pool' (checkpoints congelats a `pool_dir`, si n'hi ha). Els checkpoints
    es carreguen de manera diferida (només quan se sortegen) i es cachegen.
    """

    def __init__(self, pool_dir: Optional[str] = None, pesos: Optional[dict] = None, seed: Optional[int] = None):
        self.pool_dir = Path(pool_dir) if pool_dir else None
        self.pesos = pesos or {'random': 0.2, 'regles': 0.5, 'pool': 0.3}
        self.rng = random.Random(seed)
        self._cache_checkpoints: dict = {}

    def _checkpoints_disponibles(self):
        if not self.pool_dir or not self.pool_dir.exists():
            return []
        return sorted(self.pool_dir.glob("*.zip"))

    def _carregar_checkpoint(self, path: Path):
        key = str(path)
        if key not in self._cache_checkpoints:
            from sb3_contrib import MaskablePPO
            from RL.models.sb3.sb3_adapter import SB3PPOEvalAgent
            model = MaskablePPO.load(str(path), device="cpu")
            self._cache_checkpoints[key] = SB3PPOEvalAgent(model, n_actions=N_ACTIONS)
        return self._cache_checkpoints[key]

    def sample(self):
        """Sorteja i retorna un agent oponent nou (eval_step-compatible)."""
        opcions = ['random', 'regles']
        pesos = [self.pesos.get('random', 0.0), self.pesos.get('regles', 0.0)]

        checkpoints = self._checkpoints_disponibles()
        if checkpoints:
            opcions.append('pool')
            pesos.append(self.pesos.get('pool', 0.0))

        if sum(pesos) <= 0:
            opcions, pesos = ['random'], [1.0]

        tipus = self.rng.choices(opcions, weights=pesos, k=1)[0]

        if tipus == 'random':
            return RandomAgent(num_actions=N_ACTIONS, seed=self.rng.randrange(1 << 30))

        if tipus == 'regles':
            estil = self.rng.choices(list(PESOS_ESTIL.keys()), weights=list(PESOS_ESTIL.values()), k=1)[0]
            base = ESTILS[estil]

            def _jitter(v):
                return v * self.rng.uniform(0.9, 1.1)

            return AgentRegles(
                seed=self.rng.randrange(1 << 30),
                truc_agressio=_jitter(base['truc_agressio']),
                envit_agressio=_jitter(base['envit_agressio']),
                farol_prob=_jitter(base['farol_prob']),
                resposta_truc=_jitter(base['resposta_truc']),
                marcador_simulat=_marcador_aleatori(self.rng),
            )

        path = self.rng.choice(checkpoints)
        return self._carregar_checkpoint(path)

    def refresh(self) -> None:
        """Invalida la cache de checkpoints (per recollir nous checkpoints escrits a pool_dir)."""
        self._cache_checkpoints.clear()
