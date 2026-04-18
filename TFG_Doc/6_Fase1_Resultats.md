# 6. Fase 1: Resultats Experimentals

Aquesta secció presenta els resultats dels dos experiments de la Fase 1, dissenyats per respondre dues preguntes complementàries:

1. **Experiment per steps** (`resultats_fase1_09_04_1014h/`): donat el mateix nombre de timesteps, quin algorisme aprèn més?
2. **Experiment per temps** (`resultats_fase1_temps_09_04_2141h/`): donat el mateix temps de rellotge, quin algorisme arriba més lluny?

Tots dos experiments utilitzen la configuració descrita a [[5_Fase1_Entrenament]] i els logs (`training_log.csv`) i pesos entrenats (`best.pt`, `final.pt`) viuen dins de subcarpetes per agent: `dqn_rlcard/`, `nfsp_rlcard/`, `dqn_sb3/`, `ppo_sb3/`.

L'anàlisi s'elabora al notebook `TFG_Doc/notebooks/1_comparacio_inicial/comparacio_fase1.ipynb`, que carrega els CSVs, els alinea per `step` o `elapsed_s`, i genera les corbes d'evolució de la mètrica.

## Experiment 1 — Pressupost fix per steps (5M steps)

Pressupost per agent: **5 000 000 timesteps**.

### Temps de rellotge consumit

Extret de `resum_temps.txt`:

| Agent       |                Temps total | Throughput (steps/s) |
| :---------- | -------------------------: | -------------------: |
| PPO-SB3     |    `511 s` (≈ 8 m 31 s) |               ~9 780 |
| DQN-SB3     |      `2 584 s` (≈ 43 m) |               ~1 940 |
| DQN-RLCard  |      `3 272 s` (≈ 54 m) |               ~1 530 |
| NFSP-RLCard | `13 943 s` (≈ 3 h 52 m) |                 ~360 |

**Observacions clau**:

- PPO-SB3 és **un ordre de magnitud més ràpid** que els altres gràcies als 48 entorns paral·lels.
- NFSP-RLCard és dràsticament més lent perquè manté dos buffers enormes (RL + reservoir) i fa dos *forward/backward passes* per update.
- Els dos DQN tenen throughput comparable; SB3 és lleugerament més ràpid que l'implementació nativa de RLCard.

### Mètriques finals aproximades (últimes 5 files del log)

| Agent       | `wr_random` (darreres eval) | `wr_regles` (darreres eval) | `metric` |
| :---------- | ----------------------------: | ----------------------------: | ---------: |
| DQN-RLCard  |                       58–80% |                       11–20% |    ~23–35 |
| NFSP-RLCard |                       68–76% |                       14–36% |    ~30–46 |
| DQN-SB3     |                       44–50% |                       24–42% |    ~30–43 |
| PPO-SB3     |                       10–24% |                       20–30% |    ~18–26 |

**Lectura**:

- Amb només 5M steps, **NFSP-RLCard obté el millor pic** contra l'AgentRegles (fins a 36%) però al preu d'haver dedicat 3h 52m a fer-los. És l'agent més eficient per mostra però el menys eficient per temps.
- **DQN-RLCard** convergeix bé contra Random però no aconsegueix dominar contra l'AgentRegles — típic del trade-off off-policy + replay prioritari sense shaping fi del reward.
- **DQN-SB3** té una mètrica final similar a NFSP però amb un perfil més estable i 5× més ràpid: per aquest pressupost és **la millor relació qualitat/cost**.
- **PPO-SB3 no convergeix dins 5M steps**: és massa pocs steps per un algorisme on-policy que necessita molts més *rollouts* per explorar l'espai d'accions discret del Truc (18 accions). El wr_random queda atrapat al voltant del 15% — clarament no ha après la política òptima bàsica. Aquesta és una observació important: **PPO necessita un pressupost molt més gran per ser competitiu** en steps, però compensa amb un throughput brutal.

## Experiment 2 — Pressupost fix per temps (4 h)

Pressupost per agent: **14 400 s (4 h)**.

### Steps consumits en 4 hores

| Agent       |                Steps aconseguits |
| :---------- | -------------------------------: |
| PPO-SB3     | **~157 000 000** (≈ 157M) |
| DQN-SB3     |             ~34 000 000 (≈ 34M) |
| DQN-RLCard  |             ~17 000 000 (≈ 17M) |
| NFSP-RLCard |               ~5 000 000 (≈ 5M) |

La diferència de throughput és brutal: PPO-SB3 fa **30× més steps** que NFSP-RLCard en el mateix temps. Aquest és precisament el motiu pel qual les comparacions purament per steps penalitzen injustament els algorismes paral·lelitzats.

### Mètriques finals aproximades (darreres 5 files)

| Agent       | `wr_random` (darreres) | `wr_regles` (darreres) | `metric` |
| :---------- | -----------------------: | -----------------------: | ---------: |
| DQN-RLCard  |                  60–86% |                   7–21% |    ~21–34 |
| NFSP-RLCard |                  64–84% |                  17–23% |    ~30–35 |
| DQN-SB3     |                  22–44% |                  13–23% |    ~16–28 |
| PPO-SB3     |                  34–54% |                  17–32% |    ~21–35 |

**Lectura**:

- Contra qualsevol expectativa, **donar 4h a cada algorisme no ha millorat substancialment la mètrica** respecte a l'experiment de 5M steps per la majoria d'agents. Els dos DQN han oscil·lat entorn del mateix sostre, i NFSP també s'ha estancat.
- Això suggereix que els quatre algorismes (tal com estan configurats, sense shaping addicional) toquen un **sostre estructural** a l'entorn *partides senceres*: 24 punts és molt temps fins a la recompensa final, i el *reward shaping* intermedi no és prou per donar direcció estable.
- PPO-SB3 arriba a competir amb els altres quan li donen el throughput suficient, però **el sostre efectiu segueix al voltant del 30–35% de la mètrica**.

## Conclusions de la Fase 1

1. **Cap dels quatre algorismes arriba a batre l'`AgentRegles` de manera consistent** dins les condicions de la Fase 1 (la mètrica millor estimada queda entre 30 i 40 punts). L'agent de regles, amb les seves heurístiques adaptatives i l'estocasticitat deliberada, és una barrera molt més difícil del que semblaria inicialment.
2. **PPO té un problema de sample efficiency** a aquest entorn: amb 5M steps és l'agent pitjor, amb 157M steps tot just iguala els altres. L'entorn de partida sencera (horizon llarg, reward final molt sparse, 18 accions) no és la condició ideal per PPO.
3. **El coll d'ampolla no és el temps de càlcul**, és el **senyal d'aprenentatge**. Donar més temps als algorismes no ha canviat qualitativament els resultats, només els ha fet oscil·lar al voltant del mateix sostre.
4. **Conclusió estratègica**: cal canviar la formulació del problema, no empènyer més fort el mateix hammer. Aquesta és la motivació de la Fase 2 — utilitzar **curriculum learning** per aprendre primer la tàctica local (1 episodi = 1 mà, reward dens), i després fer finetune a partides senceres. Vegeu [[7_Fase2_MarcTeoric]] i [[8_Fase2_Implementacio]].
