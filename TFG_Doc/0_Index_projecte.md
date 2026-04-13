# Resum Entorn

**Josep Ferriol Font**


## Índex Modular (Obsidian)

Aquesta documentació està estructurada en mòduls temàtics formant un graf interactiu. Els documents s'organitzen seguint la lectura natural del projecte: primer l'arquitectura general, després la lògica del joc i els entorns RL, després els models disponibles, i finalment les dues fases experimentals (Fase 1 — comparativa d'algorismes, i Fase 2 — curriculum learning).

- [[1_Arquitectura_MVC]]: Arquitectura MVC del joc interactiu, contractes de Vista i Model.
- [[2_Logica_Joc]]: Motor lògic del joc (`TrucGame` i `TrucGameMa`) i sistema de *reward shaping*.
- [[3_Entorns_Simulacio_RL]]: Adaptadors RLCard (`TrucEnv`, `TrucEnvMa`) i wrappers Gymnasium per SB3 (`TrucGymEnv`, `TrucGymEnvMa`).
- [[4_Estructura_Models]]: Organització de `RL/models/` (`core/`, `rlcard_legacy/`, `sb3/`, `model_propi/`) i en profunditat l'`AgentRegles` estocàstic.
- [[5_Fase1_Entrenament]]: Comparativa d'algorismes DQN-RLCard / NFSP-RLCard / DQN-SB3 / PPO-SB3 amb condicions homogènies.
- [[6_Fase1_Resultats]]: Resultats dels experiments de la Fase 1 (per steps fixos i per temps fix).
- [[7_Entorn_Ma_Curriculum]]: Entorn per mans (`TrucGameMa`/`TrucEnvMa`/`TrucGymEnvMa`) i el concepte de *curriculum learning* aplicat al Truc.
- [[8_Fase2_Curriculum]]: Fase 2 experimental: control vs curriculum (mans → partides).

---

## Resum de l'Estructura

El projecte **TFG-truc** implementa el joc de cartes Truc amb una arquitectura modular preparada per a l'entrenament d'agents de Reinforcement Learning (RL). S'utilitzen tant els algorismes de la llibreria **RLCard** (DQN, NFSP) com els de **Stable-Baselines3** (DQN, PPO), comparant-los en igualtat de condicions.

### Visió General

L'objectiu principal és proporcionar un entorn robust per simular partides de Truc i entrenar agents intel·ligents, comparant algorismes clàssics i investigant l'efecte del *curriculum learning* (entrenar primer en mans individuals i després en partides senceres) sobre la seva convergència.

### Estructura de Directoris

- `joc/`: Nucli del joc sota una arquitectura MVC.
  - `entorn/`: Motor de simulació per als agents (partides senceres, Single-Agent).
    - `game.py`: Motor lògic del joc (`TrucGame`). Gestiona estats, regles, rondes, mans, apostes (Truc i Envit) i *reward shaping*.
    - `env.py`: Adaptador RLCard (`TrucEnv`). Tradueix l'estat a tensors (6,4,9) + (23,).
    - `gym_env.py`: Wrapper Gymnasium (`TrucGymEnv`) per utilitzar els entorns amb Stable-Baselines3 (SB3).
    - `cartes_accions.py`: Constants compartides (cartes, llistat d'accions, senyals).
    - `rols/`: `dealer.py`, `judger.py`, `player.py`.
  - `entorn_ma/`: Variant de l'entorn per mans individuals (1 episodi = 1 mà).
    - `game_ma.py`: Motor lògic per mans (`TrucGameMa`). Reward net normalitzat final de mà.
    - `env_ma.py`: Adaptador RLCard per mans (`TrucEnvMa`).
    - `gym_env_ma.py`: Wrapper Gymnasium per mans (`TrucGymEnvMa`).
    - `parallel_env_ma.py`: Versió paral·lela heretada (no utilitzada al pipeline actual, que depèn de `SubprocVecEnv` de SB3).
  - `controlador/`: Gestors i classes de control (arquitectura MVC).
  - `vista/`: Interfícies gràfiques (consola i escriptori, amb recursos a `img_iu/`).
- `RL/`: Flux de treball de Reinforcement Learning.
  - `models/`: Arquitectures i agents.
    - `core/`: `feature_extractor.py` (`CosMultiInput`, `ModelPreEntrenament`), `obs_adapter.py`, `loader.py`.
    - `rlcard_legacy/`: `model_adapter.py` (wrapper per connectar agents RLCard amb el nostre loader).
    - `sb3/`: `sb3_adapter.py` (`SB3PPOEvalAgent` per avaluar models SB3 dins el pipeline de RLCard).
    - `model_propi/`: `agent_regles.py` (agent Rule-Based estocàstic).
  - `entrenament/`:
    - `entrenamentEstatTruc/`: Preentrenament supervisat del "Cos" CNN+MLP (`preentrenar_cos.py`).
    - `entrenamentsComparatius/fase1/`: Script de la Fase 1 (`entrenament_comparatiu.py`) i llançadors bash.
    - `entrenamentsComparatius/fase2/`: Scripts de Fase 2 (`entrenament_fase2_curriculum.py`, etc.) i llançadors bash.
  - `tools/`: Utilitats generals.
- `demo.py`: Script de demostració interactiu per jugar una partida (humà vs bot).
- `TFG_Doc/`: Documentació teòrica i llibretes de resultats.
  - `notebooks/1_comparacio_inicial/`: Notebook i carpetes `resultats_fase1_*`.
  - `notebooks/2_curriculum_learning/`: Notebook Fase 2.
