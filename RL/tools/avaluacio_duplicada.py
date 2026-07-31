"""Avaluació amb repartiments duplicats (com el bridge duplicat).

Comparar agents amb un win-rate sobre mans independents és sorollós: la
sort de qui ha rebut les cartes bones domina el senyal i cal moltíssimes
mans per detectar diferències reals de decisió. Aquí, cada repartiment de
cartes es juga DUES vegades, intercanviant qui seu a cada seient (mateixa
llavor del barrejador a totes dues), de manera que cada agent juga
exactament les mateixes dues mans que el rival. La sort es cancel·la sola
i només queda la diferència de qualitat de les decisions.

Ús com a script:
    python -m RL.tools.avaluacio_duplicada --model ruta/model.zip --rival agressiu
    python -m RL.tools.avaluacio_duplicada --model ruta/A.zip --rival model --model_y ruta/B.zip

Ús com a mòdul:
    from RL.tools.avaluacio_duplicada import avaluar_duplicat
"""
import argparse
import os
import sys

if '__file__' in globals():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import random
import numpy as np

RIVALS_REGLES = ("conservador", "equilibrat", "agressiu", "farol")


def _jugar_ma(env, agents) -> int:
    """Juga una mà i retorna el marge de punts des del punt de vista del seient 0."""
    state, pid = env.reset()
    steps = 0
    while pid is not None and steps < 80:
        action, _ = agents[pid].eval_step(state)
        state, pid = env.step(action)
        steps += 1
    return env.game.score[0] - env.game.score[1]


def avaluar_duplicat(env, factory_x, factory_y, n_repartiments: int, rng: random.Random):
    """Avalua l'agent X contra Y amb repartiments duplicats.

    `factory_x`/`factory_y` són callables `seed -> agent` (agent amb
    `.eval_step(state)`). Retorna (marge_mitjà, error_estàndard,
    fraccio_repartiments_guanyats_per_x, fraccio_perduts).
    """
    marges_x = []
    for _ in range(n_repartiments):
        deal_seed = rng.randrange(1 << 30)
        seed_x = rng.randrange(1 << 30)
        seed_y = rng.randrange(1 << 30)

        env.game.np_random = np.random.RandomState(deal_seed)
        marge1 = _jugar_ma(env, [factory_x(seed_x), factory_y(seed_y)])

        # Mateix repartiment exacte, seients intercanviats.
        env.game.np_random = np.random.RandomState(deal_seed)
        marge2 = _jugar_ma(env, [factory_y(seed_y), factory_x(seed_x)])

        marges_x.append(marge1 - marge2)

    marges = np.array(marges_x, dtype=np.float64)
    marge_mig = float(marges.mean())
    error = float(marges.std(ddof=1) / np.sqrt(len(marges))) if len(marges) > 1 else 0.0
    return marge_mig, error, float((marges > 0).mean()), float((marges < 0).mean())


def _rival_factory(nom: str, model_y_path: str | None):
    if nom == "random":
        from RL.models.model_propi.random_agent import RandomAgent
        return lambda s: RandomAgent(num_actions=19, seed=s)
    if nom in RIVALS_REGLES:
        from RL.models.model_propi.agent_regles import AgentRegles
        return lambda s, e=nom: AgentRegles(seed=s, estil=e)
    if nom == "probabilistic":
        from RL.models.model_propi.agent_probabilistic import AgentProbabilistic
        return lambda s: AgentProbabilistic(seed=s)
    if nom == "cfr":
        from RL.models.model_propi.agent_cfr import AgentCFR
        return lambda s: AgentCFR(seed=s)
    if nom == "model":
        if not model_y_path:
            raise ValueError("--model_y és obligatori quan --rival model")
        from sb3_contrib import MaskablePPO
        from RL.models.sb3.sb3_adapter import SB3PPOEvalAgent
        model = MaskablePPO.load(model_y_path, device="cpu")
        return lambda s: SB3PPOEvalAgent(model)
    raise ValueError(f"--rival desconegut: {nom!r}")


def main() -> None:
    from joc.entorn_ma.env_ma import TrucEnvMa
    from RL.models.sb3.sb3_adapter import SB3PPOEvalAgent
    from sb3_contrib import MaskablePPO

    parser = argparse.ArgumentParser(description="Avaluació amb repartiments duplicats per al Truc")
    parser.add_argument("--model", required=True, help="Checkpoint .zip de MaskablePPO a avaluar (agent X)")
    parser.add_argument("--rival", default="agressiu",
                         choices=["random", "model", "probabilistic", *RIVALS_REGLES],
                         help="Rival (agent Y): random | conservador|equilibrat|agressiu|farol | "
                              "probabilistic (determinització/minimax) | model (--model_y)")
    parser.add_argument("--model_y", default=None, help="Checkpoint .zip per a --rival model")
    parser.add_argument("--n_repartiments", type=int, default=250)
    parser.add_argument("--num_jugadors", type=int, default=2)
    parser.add_argument("--cartes_jugador", type=int, default=3)
    parser.add_argument("--permetre_apostes", action="store_true", help="Per defecte False (etapa 'cartes')")
    parser.add_argument("--permetre_truc", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    env_config = {
        "num_jugadors": args.num_jugadors,
        "cartes_jugador": args.cartes_jugador,
        "permetre_apostes": args.permetre_apostes,
        "permetre_truc": args.permetre_truc,
    }
    env = TrucEnvMa(config=env_config)
    rng = random.Random(args.seed)

    model_x = MaskablePPO.load(args.model, device="cpu")
    factory_x = lambda s: SB3PPOEvalAgent(model_x)
    factory_y = _rival_factory(args.rival, args.model_y)

    marge, err, pwin, plos = avaluar_duplicat(env, factory_x, factory_y, args.n_repartiments, rng)
    print(f"X ({args.model}) vs {args.rival}: marge mitjà = {marge:+.3f} ± {err:.3f} pts/repartiment "
          f"(X guanya el repartiment {pwin*100:.1f}%, en perd {plos*100:.1f}%, "
          f"empat exacte {(1-pwin-plos)*100:.1f}%)")


if __name__ == "__main__":
    main()
