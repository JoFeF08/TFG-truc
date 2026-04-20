# 11. Fase 3: Implementació i Resultats

Aquest document descriu la implementació tècnica de la Fase 3 definida a [[10_Fase3_MarcTeoric]] i recull els resultats numèrics obtinguts. L'objectiu és respondre:

> **Q1:** Aporta valor el feature extractor COS (arquitectura CNN + densa) respecte al MLP estàndard de SB3, en els tres protocols d'entrenament?

## Disseny experimental

**8 runs nous** a `TrucGymEnv` o `TrucGymEnvMa` (depenent del protocol), cadascun de 24M steps (o 12M+12M per curriculum). A aquests s'afegeixen **4 sèries reutilitzades de la Fase 2**, de manera que el total de comparació és de 12 sèries:

| Protocol   | Agent   | COS           | MLP           |
| :--------- | :------ | :------------ | :------------ |
| Control    | DQN-SB3 | **nou** | F2 control    |
| Control    | PPO-SB3 | **nou** | F2 control    |
| Curriculum | DQN-SB3 | **nou** | F2 curriculum |
| Curriculum | PPO-SB3 | **nou** | F2 curriculum |
| Mans       | DQN-SB3 | **nou** | **nou** |
| Mans       | PPO-SB3 | **nou** | **nou** |

Tots els runs comparteixen la mateixa funció d'avaluació (`evaluar_agent()`, 100+100 partides senceres cada 500 000 steps) i els mateixos hiperparàmetres RL que la Fase 2.

## Fitxers afectats

**Nous o modificats per aquesta fase:**

- `RL/entrenament/entrenamentsComparatius/fase3/fase30/entrenament_fase3.py` — script principal.
- `RL/entrenament/entrenamentsComparatius/fase3/fase30/run_fase3.sh` — llançador dels 6 runs.
- `TFG_Doc/notebooks/3_feature_extractor/30_comparacio_sense/comparacio_fase30.ipynb` — notebook d'anàlisi.

**Reutilitzats sense modificació:**

- `RL/models/sb3/sb3_features_extractor.py` — `CosMultiInputSB3`.
- `RL/models/core/feature_extractor.py` — `CosMultiInput` (sense canvis).
- `RL/entrenament/entrenamentsComparatius/fase2/entrenament_fase2_curriculum.py` — importa tots els hiperparàmetres i helpers.

## Script `entrenament_fase3.py`

L'script accepta els arguments:

```
--agent      {ppo_sb3, dqn_sb3}
--protocol   {control, curriculum, mans}
--cos        flag booleà (True → CosMultiInputSB3 scratch; absent → MLP estàndard)
--steps      total steps (defecte 24 000 000; per curriculum: 12M+12M)
--num_envs   entorns PPO (defecte 32)
--save_dir   carpeta de sortida
```

### Construcció del model

La diferència entre la variant COS i la variant MLP és exclusivament el `policy_kwargs`:

```python
# Variant COS (scratch, pesos aleatoris):
policy_kwargs = dict(
    features_extractor_class=CosMultiInputSB3,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=...,          # [256,256] per DQN, dict(pi=...,vf=...) per PPO
)

# Variant MLP estàndard:
policy_kwargs = dict(net_arch=...)  # CombinedExtractor per defecte de SB3
```

No hi ha cap càrrega de pesos preentrenats: la variant COS d'aquesta fase inicialitza el cos amb els pesos aleatoris de SB3. Això aïlla l'efecte **arquitectural** del COS del possible efecte del preentrenament supervisat.

### Protocols

**Control** (`--protocol control`): crida `_dqn_loop` o `_ppo_loop` una sola vegada amb `use_mans=False` i `timesteps = steps`. Entorn: `TrucGymEnv` (partides senceres).

**Curriculum** (`--protocol curriculum`): executa dues fases seqüencials:

1. `_dqn_loop` / `_ppo_loop` amb `use_mans=True`, `log_name='training_log_mans.csv'`, `timesteps = steps // 2`. Entorn: `TrucGymEnvMa` (mans).
2. Carrega `best_mans.zip`, `_dqn_loop` / `_ppo_loop` amb `use_mans=False`, `log_name='training_log_partides.csv'`, `timesteps = steps // 2`. Entorn: `TrucGymEnv`.

Les corbes del notebook mostren els steps acumulats (0–24M) amb l'offset de l'etapa 1 sumades als steps de l'etapa 2.

**Mans** (`--protocol mans`): crida `_dqn_loop` o `_ppo_loop` una sola vegada amb `use_mans=True` i `timesteps = steps`. Entorn: `TrucGymEnvMa` (mans individuals).

### Avaluació i guardat

Idèntics als de `entrenament_fase2_curriculum.py`:

- `EvalCallback` cada `EVAL_EVERY_STEPS = 500 000` steps → `evaluar_agent()` → `training_log.csv`.
- `best.*` quan la `metric` millora; `final.*` al final del run.
- L'avaluació és sempre sobre partides senceres (no sobre mans), per mantenir comparabilitat amb Fase 1 i Fase 2.

## Script `run_fase3.sh`

```bash
bash RL/entrenament/entrenamentsComparatius/fase3/fase30/run_fase3.sh [STEPS]
```

Llança 6 experiments en seqüència:

```
ppo_sb3 + control + cos
dqn_sb3 + control + cos
ppo_sb3 + curriculum + cos
dqn_sb3 + curriculum + cos
ppo_sb3 + mans + mlp
dqn_sb3 + mans + mlp
```

Sortida: `TFG_Doc/notebooks/3_feature_extractor/resultats/resultats_fase3_<TIMESTAMP>/` amb una subcarpeta per experiment (`{agent}_{protocol}_{tag}/`).

## Layout de registres

```
TFG_Doc/notebooks/3_feature_extractor/resultats/
├── resum_fase3.txt
├── ppo_sb3_control_cos/
│   ├── training_log.csv
│   ├── best.zip
│   └── final.zip
├── dqn_sb3_control_cos/
├── ppo_sb3_curriculum_cos/
│   ├── training_log_mans.csv
│   ├── training_log_partides.csv
│   ├── best_mans.zip
│   └── final_partides.zip
├── dqn_sb3_curriculum_cos/
├── ppo_sb3_mans_mlp/
└── dqn_sb3_mans_mlp/
```

## Notes metodològiques

- **Observació**: vector pla de **240 dimensions** (216 cartes + 24 context). Veure `RL/tools/obs_utils.py`.
- **CosMultiInputSB3**: extractor que divideix 240 dims en `(6,4,9)` cartes i `(24,)` context, CNN + Linear fusionat a 256 dims. Definit a `RL/models/sb3/sb3_features_extractor.py`.
- **MLP estàndard SB3**: `CombinedExtractor` per defecte (aplana i concatena totes les entrades del dict).
- **PPO-SB3**: `NUM_ENVS = 32` entorns paral·lels (SubprocVecEnv), `n_steps=256`, `batch_size=2048`.
- **DQN-SB3**: entorn únic, `learning_starts=50 000`, `buffer_size=200 000`, `exploration_fraction=0.3`.
- **Protocol curriculum**: fase 1 = `TrucGymEnvMa` (mans), 12M steps; fase 2 = finetune sobre `TrucGymEnv` (partides), 12M steps. Les corbes mostren els steps acumulats (0–24M); l'avaluació es fa sempre sobre partides senceres.
- **Avaluació**: 100 partides contra agent aleatori + 100 contra agent de regles cada 500 000 steps. `eval_metric = 0.25 × WR_random + 0.75 × WR_regles`.
- **Execucions reutilitzades**: `control mlp` i `curriculum mlp` provenen de la Fase 2.

## Resultats (run complet, 24M steps)

Run executat el 18 abr 2026. Dades a `TFG_Doc/notebooks/3_feature_extractor/resultats/`.

### Taula resum

| Protocol   | Agent   |       COS (pic) |       MLP (pic) |        Δ |
| :--------- | :------ | --------------: | --------------: | --------: |
| Control    | DQN-SB3 | **71.0%** |           60.5% |  +10.5 pp |
| Control    | PPO-SB3 |           32.5% | **35.0%** |  −2.5 pp |
| Curriculum | DQN-SB3 |           56.2% | **56.0%** |   +0.2 pp |
| Curriculum | PPO-SB3 | **63.0%** |           75.0% | −12.0 pp |
| Mans       | DQN-SB3 | **82.2%** |           85.8% |  −3.6 pp |
| Mans       | PPO-SB3 | **89.0%** |           87.2% |   +1.8 pp |

### Lectures principals

**Control**: DQN-SB3 es beneficia clarament del COS (+10.5 pp). PPO-SB3 no millora amb COS scratch (−2.5 pp), cosa coherent amb la dificultat que ja mostrava PPO amb l'entorn de partides a Fase 2.

**Curriculum**: Les diferències COS vs MLP s'esvaeixen per a DQN (pràcticament empat). Per a PPO, la variant MLP segueix sent clarament superior (+12 pp). El COS scratch en protocol curriculum no aporta avantatge sobre el MLP: l'etapa 1 (mans) entrena l'agent i el cos conjuntament des de zero, però el finetune a l'etapa 2 afecta el cos de manera prou significativa com per perdre l'avantatge inicial.

**Mans**: Amb un senyal de recompensa dens (episodis curts), ambdues arquitectures convergeixen a resultats molt similars (diferències <4 pp en tots els casos). El protocol de mans no discrimina entre COS i MLP.

### Interpretació global

El COS aporta valor mesurable **en el protocol de control per a DQN-SB3**. En la resta de combinacions, la diferència amb el MLP és marginal o negativa per al COS scratch. Dues conclusions:

1. La inducció estructural del COS no és universalment millor: quan l'entrenament és prou dens (mans) o quan l'agent és on-policy (PPO amb els seus múltiples passos de gradient), el MLP estàndard és competitiu.
2. L'impacte del preentrenament supervisat (frozen/finetune) és una pregunta separada que es tracta en la fase següent.

## Decisió i motivació de la fase següent

> **Decisió**: El **protocol mans** és el que ofereix els millors resultats en tots els escenaris (>82% per a totes les combinacions). La tria entre COS i MLP per a PPO+mans és marginal (+1.8 pp); s'optarà per provar diverses tècniques per aprofitar millor el COS.

El protocol mans és l'escenari on el senyal d'aprenentatge és més ric i les polítiques convergeixen més ràpidament. Però la Fase 3 ha usat el COS sempre amb **pesos aleatoris** (scratch), la qual cosa implica que l'arquitectura CNN ha d'aprendre la representació de les cartes des de zero, en competència directa amb l'aprenentatge de la política. Hi ha raons per creure que **inicialitzar el COS amb pesos supervisats** podria desbloquejar un rendiment superior al scratch:

- Un cos preentrenat ja distingeix rangs, pals i accions legals des del primer step de RL; la política pot centrar-se immediatament en la presa de decisions en lloc d'aprendre la representació.
- Al protocol de mans, on els episodis són curts (~15 steps), el gradient per actualitzar el cos és poc freqüent en termes absoluts. Un cos preentrenat arranca d'un punt millor i pot arribar a un sostre més alt.

Per explorar-ho, la fase següent **fixa el protocol mans** (el millor de Fase 3) i compara tres estratègies d'inicialització del cos preentrenat: **scratch** (aleatori, línia base equivalent al millor resultat de Fase 3), **frozen** (pesos supervisats congelats, la política aprèn sola) i **finetune** (pesos supervisats descongelats, cos i política s'adapten conjuntament). Això permet respondre la pregunta complementària a Q1:

> **Q2:** Dona valor el preentrenament supervisat del COS, i en cas afirmatiu, quin règim (frozen vs finetune) és preferible?

## Procediment d'ús

1. **Llançar els 8 runs nous**:
   ```bash
   bash RL/entrenament/entrenamentsComparatius/fase3/fase30/run_fase3.sh
   ```

2. **Smoke test** (recomanat abans del run complet):
   ```bash
   bash RL/entrenament/entrenamentsComparatius/fase3/fase30/run_fase3.sh 200000
   ```
   Cada run hauria de completar-se en <5 min i generar el `training_log.csv` corresponent.

3. **Anàlisi**: obrir `TFG_Doc/notebooks/3_feature_extractor/30_comparacio_sense/comparacio_fase30.ipynb` i executar totes les cel·les. El notebook carrega automàticament la carpeta de resultats i combina les sèries noves amb les reutilitzades de Fase 2.

## Enllaços creuats

- [[10_Fase3_MarcTeoric]] — fonaments conceptuals i justificació del disseny.
- [[8_Fase2_Implementacio]] — font dels hiperparàmetres i de les sèries MLP reutilitzades.
- [[9_Checkpoint1]] — selecció dels dos models i pla de les fases 3–5.
- [[12_Fase35_MarcTeoric]] — Q2: estratègies d'inicialització del COS (fase següent).
