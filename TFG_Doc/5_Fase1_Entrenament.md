# 5. Fase 1: Comparativa Homogènia d'Algorismes

Aquesta primera fase experimental, descrita al fitxer `RL/entrenament/entrenamentsComparatius/fase1/entrenament_comparatiu.py`, compara **quatre algorismes de Reinforcement Learning** sobre l'entorn del Truc **en igualtat de condicions** i partint de zero (*from scratch*, sense pre-entrenament ni transferència de pesos). L'objectiu és establir una base rigorosa sobre la qual comparar posteriorment tècniques més sofisticades (Fase 2 — curriculum learning).

## Algorismes Avaluats

| Algorisme | Llibreria | On/Off-policy | Entorn | Paral·lelisme |
|:--|:--|:--|:--|:--|
| **DQN-RLCard** | RLCard | Off-policy | `TrucEnv` (seqüencial) | No |
| **NFSP-RLCard** | RLCard | Off-policy + SL | `TrucEnv` (seqüencial) | No |
| **DQN-SB3** | Stable-Baselines3 | Off-policy | `TrucGymEnv` | 1 entorn |
| **PPO-SB3** | Stable-Baselines3 | On-policy | `TrucGymEnv` | 48 entorns `SubprocVecEnv` |

Tots els agents comparteixen:

- **Dimensió d'observació**: 239 (aplanament de `(6,4,9) + (23,)`).
- **Arquitectura**: MLP `[256, 256]` + ReLU, sense *feature extractor* convolucional. Es volia que les diferències s'atribuïssin al **algorisme**, no a l'arquitectura.
- **Pressupost**: 24 milions de timesteps (o alternativament, temps màxim fix de 4 h per comparar velocitat).
- **Funció d'avaluació**: `evaluar_agent(agent, ENV_CONFIG, regles_eval)` — 50 partides contra `RandomAgent` + 100 partides contra `AgentRegles`.
- **Mètrica comuna**: `metric = 0.25 · wr_random + 0.75 · wr_regles`.
- **Freqüència d'avaluació**: cada `EVAL_EVERY_STEPS = 500_000`.
- **Format del log**: `training_log.csv` amb columnes `step, games_played, loss, eval_wr_random, eval_wr_regles, eval_metric, elapsed_s`.

La mètrica pondera pesadament la victòria contra l'`AgentRegles` (0.75) perquè guanyar al `RandomAgent` és trivial (el sostre natural és molt alt) i l'interès real és batre un oponent amb estratègia coherent. Vegeu [[4_Estructura_Models]] per la descripció de l'agent de regles estocàstic.

## 1. DQN-RLCard

DQN off-policy clàssic amb un sol `TrucEnv` seqüencial. El replay buffer gran (2M) garanteix la diversitat de mostres sense necessitat de paral·lelitzar l'entorn.

| Paràmetre | Valor |
|:--|:--|
| Learning rate | `1e-4` |
| Batch size | 512 |
| Replay memory | 2 000 000 |
| Warmup | 100 000 |
| Train every | 128 steps |
| Update target | cada 2 000 steps |
| Epsilon decay | 80% dels timesteps fins a `0.05` |
| Polyak tau | 0.05 (cada 5 000 partides) |

**Distribució d'oponents per episodi**: `10% Random + 60% AgentRegles + 30% self-play Polyak`. L'oponent Polyak és una còpia del DQN principal que s'actualitza lentament (`tau=0.05`) per evitar que l'agent aprengui a explotar una versió fixa de si mateix i quedi atrapat en un equilibri fràgil.

## 2. NFSP-RLCard

NFSP (Neural Fictitious Self-Play) combina un *best response* via DQN amb una *supervised policy* entrenada sobre un *reservoir buffer* de les pròpies jugades històriques. L'equilibri teòric aproximat és el de Nash (política mitjana vs best response).

| Paràmetre | Valor |
|:--|:--|
| RL learning rate | `1e-4` |
| SL learning rate | `1e-4` |
| Batch size | 512 |
| Q-replay | 2 000 000 |
| Reservoir | 2 000 000 |
| η (anticipabilitat) | 0.25 |
| Warmup | 100 000 |
| Epsilon min | 0.05 |

L'hiperparàmetre η = 0.25 controla el pes entre les dues polítiques: el 25% de les decisions provenen del mòdul best response agressiu, i el 75% de la política supervisada conservadora. Això estabilitza l'entrenament però ralenteix molt l'aprenentatge dels conceptes tàctics locals en comparació amb un DQN pur.

**Distribució d'oponents**: `10% Random + 60% AgentRegles + 30% self-play NFSP natiu`.

## 3. DQN-SB3

Rèplica del DQN-RLCard amb la implementació de Stable-Baselines3 i l'entorn `TrucGymEnv`. La idea és controlar si les diferències entre les dues implementacions del mateix algorisme (RLCard vs SB3) són significatives.

| Paràmetre | Valor |
|:--|:--|
| Learning rate | `1e-4` |
| Batch size | 512 |
| Buffer size | 2 000 000 |
| Warmup | 100 000 |
| Train freq | 4 steps |
| Target update | 2 000 |
| ε inicial | 1.0 |
| ε final | 0.05 |
| ε fracció | 0.8 del total |
| Gamma | 0.99 |

A diferència de DQN-RLCard, aquí no s'implementa *self-play Polyak*: l'oponent es defineix en temps de construcció de cada `TrucGymEnv` i és fix durant tot l'entrenament (majoritàriament `AgentRegles`).

## 4. PPO-SB3

L'únic algorisme on-policy de la comparativa. Utilitza 48 instàncies paral·leles de `TrucGymEnv` via `SubprocVecEnv` natiu de SB3, sense necessitat de cap infraestructura pròpia.

| Paràmetre | Valor |
|:--|:--|
| Nombre d'entorns | 48 |
| Learning rate | `3e-4` |
| Gamma | 0.995 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Entropy coef | 0.01 |
| Value coef | 0.5 |
| N epochs | 7 |
| Minibatch | 1 024 |
| N steps | 256 |

**Distribució d'oponents**: ~5% Random, la resta `AgentRegles` (no hi ha self-play natiu a PPO). La distribució s'implementa repartint els 48 subprocessos entre factories amb oponent `random` i factories amb oponent `regles`.

PPO té un comportament molt diferent als tres anteriors: com que és on-policy, cada *rollout* s'ha de generar amb la política actual i consumeix molts més steps per unitat de temps d'optimització. L'alta paral·lelització compensa aquesta ineficiència relativa.

## Evaluació Comuna

Per garantir que els quatre algorismes es comparin exactament amb la mateixa vara de mesurar, tots passen per la mateixa funció `evaluar_agent()`:

```python
def evaluar_agent(agent, env_config, regles_agent,
                  n_random=50, n_regles=100):
    eval_env = wrap_env_aplanat(TrucEnv(env_config))
    rand_opp = RandomAgent(num_actions=N_ACTIONS)

    wins_r = sum(1 for i in range(n_random)
                 if _run_eval(eval_env, agent, rand_opp, i%2))
    wr_random = 100.0 * wins_r / n_random

    wins_g = sum(1 for i in range(n_regles)
                 if _run_eval(eval_env, agent, regles_agent, i%2))
    wr_regles = 100.0 * wins_g / n_regles

    metric = 0.25 * wr_random + 0.75 * wr_regles
    return wr_random, wr_regles, metric
```

Nota important: l'agent alterna la seva posició (jugador 0 / jugador 1) cada partida per evitar biaix posicional.

Els agents DQN/NFSP de RLCard ja implementen `eval_step()` nativament; els de SB3 es fan compatibles a través de **`SB3EvalAgent`**, un adaptador definit al mateix fitxer i equivalent al `SB3PPOEvalAgent` de `RL/models/sb3/sb3_adapter.py` (vegeu [[4_Estructura_Models]]).

## Execució

Els scripts bash `run_fase1.sh` i `run_fase1_temps.sh` llancen els quatre agents seqüencialment:

- **`run_fase1.sh`**: 5M timesteps per agent (pressupost fix per steps).
- **`run_fase1_temps.sh`**: 14 400 s (4 h) per agent (pressupost fix per temps).

Executar els dos experiments permet observar alhora:

1. Quin algorisme assoleix millor mètrica per un nombre d'steps donat (**qualitat per mostra**).
2. Quin algorisme assoleix millor mètrica per un pressupost de temps donat (**throughput real**).

Els resultats concrets i la seva interpretació es presenten a [[6_Fase1_Resultats]].
