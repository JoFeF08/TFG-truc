# Resum Entorn

**Josep Ferriol Font**


## Índex Modular (Obsidian)

Aquesta documentació està estructurada en mòduls temàtics formant un graf interactiu. Podeu navegar pels següents enllaços per endinsar-vos en la teoria arquitectònica del projecte:

- [[1_Arquitectura_MVC]]: Orquestració del joc interactiu, contractes de Vista i Model.
- [[2_Logica_Joc]]: Arbre de decisions i regles motores (`TrucGame` i `TrucGameMa`).
- [[3_Entorns_Simulacio_RL]]: Adaptadors d'estat humà a vectors/tensors formals (`TrucEnv` i `TrucEnvMa`).
- [[4_Entorns_Parallelisme]]: Execució *Scatter-gather* multi-procés pur per reduir els colls d'ampolla.
- [[5_Models_RLCard]]: Entrenament *off-policy* baselines (DQN, NFSP) i el "Cos" extractor de CNN.
- [[6_Model_PPO_Propi]]: Autoria de xarxes recurrents *on-policy* PPO amb GRU des de zero.

---

## Resum de l'Estructura

El projecte **TFG-truc** implementa el joc de cartes Truc amb una arquitectura modular preparada per a l'entrenament d'agents de Reinforcement Learning (RL).

### Visió General

L'objectiu principal és proporcionar un entorn robust per simular partides de Truc i entrenar agents intel·ligents utilitzant la llibreria RLCard.

### Estructura de Directoris

- `joc/`: Nucli del joc sota una arquitectura MVC.
  - `entorn/`: Motor de simulació per als agents de Reinforcement Learning.
    - `game.py`: Motor lògic del joc. Gestiona els estats de la partida, les regles, la progressió de rondes i mans, i les apostes (Truc i Envit).
    - `env.py`: Adaptador per a RLCard. Tradueix l'estat intern del joc a observacions numèriques i exposa l'API estàndard de l'entorn (`reset`, `step`).
    - `parallel_env.py`: Entorns vectorials paral·lels (`SubprocVecEnv`) que executen N instàncies de `TrucEnv` en processos separats via `multiprocessing`. Detalls exhaustius a [[4_Entorns_Parallelisme]].
    - `cartes_accions.py`: Fitxer de constants compartides.
    - `rols/`: Implementacions dels rols de l'entorn.
      - `dealer.py`: Gestiona la baralla (creació, barreja i repartiment de cartes als jugadors).
      - `judger.py`: Conté la lògica d'arbitratge: determina el guanyador d'una ronda, d'una mà i de l'envit.
      - `player.py`: Classe base que defineix la interfície comuna dels agents o jugadors aleatoris.
  - `entorn_ma/`: Variant de l'entorn per a entrenament per mans individuals (1 episodi = 1 mà). Detalls exhaustius a [[4_Entorns_Parallelisme]].
    - `game_ma.py`: Motor lògic per mans (`TrucGameMa`). Cada mà és un episodi complet amb reward net normalitzat.
    - `env_ma.py`: Adaptador RLCard per mans (`TrucEnvMa`). Observació idèntica a `TrucEnv` (6,4,9)+(23,).
    - `parallel_env_ma.py`: Entorn vectorial paral·lel per mans (`SubprocVecEnvMa`). Mateixa arquitectura que `SubprocVecEnv`.
  - `controlador/`: Gestors i classes de control (arquitectura MVC) que interactuen amb la simulació.
  - `vista/`: Interfícies gràfiques o de consola (MVC) on s'hi desenvolupen i mostren les partides (inclou visualitzador per consola i escriptori).
- `RL/`: Flux de treball de Reinforcement Learning.
  - `entrenament/`: Scripts i codi dedicat a realitzar els entrenaments i l'avaluació.
  - `models/`: Models amb els pesos i punts de control per als diferents agents.
  - `notebooks/`: Llibretes Jupyter per elaborar proves, estadístiques i avaluacions sobre l'entrenament dels agents.
  - `tools/`: Utilitats generals pel tractament de les simulacions i xarxes neuronals de l'entorn.
- `demo.py`: Script de demostració interactiu per jugar. Permet configurar el nombre de jugadors, tipus de partides, rols (Humà o Agent) a través del terminal o finestra d'escriptori.
- `TFG_Doc/`: Documentació teòrica fragmentada i codi del projecte en Markdown preparat per Obsidian.
