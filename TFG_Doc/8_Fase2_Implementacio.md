# 8. Fase 2: Implementació i Resultats

Aquesta fase implementa i avalua el *curriculum learning* descrit a [[7_Fase2_MarcTeoric]] sobre els mateixos quatre algorismes de la Fase 1. L'objectiu experimental és respondre a una sola pregunta:

> Donat un pressupost fix de 24M timesteps, entrenar primer 12M en mans i després 12M en partides, supera a entrenar directament 24M en partides?

L'script responsable és `RL/entrenament/entrenamentsComparatius/fase2/entrenament_fase2_curriculum.py`.

## Disseny experimental

Per a cada algorisme es llancen **dues** execucions independents:

| Mode | Etapa 1 | Etapa 2 | Total |
|:--|:--|:--|:--|
| **`control`** | — | 24M steps en `TrucGymEnv` (partides) | 24M |
| **`curriculum`** | 12M steps en `TrucGymEnvMa` (mans) | 12M steps en `TrucGymEnv` (partides) amb pesos carregats | 24M |

Això dóna **8 execucions totals** (4 algorismes × 2 modes), comparables amb els mateixos hiperparàmetres i la mateixa funció d'avaluació `evaluar_agent()` que a la Fase 1.

### Canvis respecte a Fase 1

- **Sense self-play per cap agent.** A Fase 1, DQN i NFSP de RLCard usaven self-play amb ~30% de les partides. Aquí s'elimina per aïllar l'efecte del curriculum.
- **Distribució d'oponents unificada**: `PCT_RANDOM = 0.10` + `PCT_REGLES = 0.90` per als quatre agents.
- **Sense shaping multiplicat**: els hiperparàmetres dels algorismes són els mateixos de la Fase 1 (mateixa arquitectura `[256, 256]`, mateix learning rate, mateix buffer…), excepte les modificacions específiques per al finetune.

### Modificacions per al finetune (mode curriculum)

Al pas d'etapa 1 → etapa 2, els pesos de l'etapa 1 s'han de carregar i el loop d'entrenament ha de continuar, però amb **menys exploració** per no desfer el que s'ha après:

- **DQN-RLCard**: `q_estimator.qnet.load_state_dict(torch.load(best_mans.pt))`. L'epsilon segueix el seu calendari normal sobre els 12M restants.
- **NFSP-RLCard**: similar, amb càrrega separada per l'average policy i el best response.
- **DQN-SB3**: `SB3DQN.load(best_mans.zip, env=...)` i a sobre:
  - `learning_starts = SB3DQN_WARMUP_FT = 10_000` (warmup curt, no 100k).
  - `exploration_rate = SB3DQN_EPS_FINETUNE = 0.10` (no 1.0 ni 0.05).
- **PPO-SB3**: `PPO.load(best_mans.zip, env=...)` amb tota la resta d'hiperparàmetres idèntics.

Els noms dels fitxers de pesos segueixen el patró `best_mans.pt` / `best_mans.zip` (millor mètrica durant l'etapa 1) i `best_partides.*` / `final_partides.*` (etapa 2).

## Estructura de l'script

L'script `entrenament_fase2_curriculum.py` exporta quatre funcions top-level:

```python
run_dqn(save_dir, mans_steps, partides_steps, device, mode)
run_nfsp(save_dir, mans_steps, partides_steps, device, mode)
run_dqn_sb3(save_dir, mans_steps, partides_steps, device, mode)
run_ppo(save_dir, mans_steps, partides_steps, device, mode)
```

Cadascuna internament:

1. Si `mode == 'control'`: crida el *loop* genèric una sola vegada amb `use_mans=False` i `timesteps = mans_steps + partides_steps`.
2. Si `mode == 'curriculum'`:
   - Primer loop amb `use_mans=True`, `log_name='training_log_mans.csv'`, durada `mans_steps`.
   - Segon loop amb `use_mans=False`, `log_name='training_log_partides.csv'`, durada `partides_steps`, passant el path a `best_mans.*` com a `load_path`.

Els dos loops comparteixen les funcions helper (`_make_gym_env_fn` vs `_make_gym_env_ma_fn`, `evaluar_agent`, `append_log`) i ambdues avaluacions es fan **sempre sobre partides senceres** (`ENV_CONFIG`, no `ENV_CONFIG_MA`), perquè és on volem mesurar la competència real.

Això últim és clau: encara que l'agent estigui aprenent sobre mans a l'etapa 1, l'avaluació és sobre partides. Així podem veure directament al CSV si l'entrenament en mans està transferint coneixement útil a partides o no.

## Scripts de llançament

Dins de `RL/entrenament/entrenamentsComparatius/fase2/` hi ha:

- **`run_fase2.sh`**: llança tots els agents × tots els modes.
- **`run_fase2_control.sh`**: només el grup de control.
- **`run_fase2_curriculum.sh`**: només el grup curriculum.

Els resultats es desen automàticament a `fase2/registres/<agent>_<mode>_<timestamp>/` amb subcarpetes per log i pesos.

## Arxiu auxiliar `entrenament_fase2_cos.py`

Aquest script és un experiment paral·lel (no integrat al curriculum principal) per investigar l'efecte del **cos pre-entrenat** (`CosMultiInput` amb pesos obtinguts via `ModelPreEntrenament`, vegeu [[4_Estructura_Models]]) sobre un PPO propi. Compara tres modes: `scratch` (cos aleatori), `frozen` (cos congelat) i `finetune` (cos descongelat al 15%).

> **Important**: aquest script depèn de components històrics del `model_propi/model_ppo/` que ja no formen part del pipeline principal. Tanmateix, es manté per referència a l'estudi del pre-entrenament supervisat. El curriculum "actiu" de Fase 2 és el de `entrenament_fase2_curriculum.py`.

## Resultats esperats

El notebook `TFG_Doc/notebooks/2_curriculum_learning/comparacio_fase2.ipynb` carrega els `training_log_*.csv` de les 8 execucions i compara les corbes. Les hipòtesis a validar són:

1. **Acceleració inicial**: al principi de l'etapa 2, l'agent curriculum ha de tenir una mètrica superior al control (en el mateix step global) gràcies al coneixement tàctic previ.
2. **Convergència final comparable o millor**: el pic de la mètrica del curriculum a la fi dels 24M ha de ser ≥ pic del control. Si és significativament millor, el curriculum val la pena.
3. **Shift inicial a l'etapa 2**: és probable observar una **caiguda temporal** de la mètrica just quan comença l'etapa 2 (shift de distribució). Si és petita i ràpidament recuperada, no és problema; si és gran i persistent, el curriculum és perjudicial per aquell algorisme.
4. **Diferència entre algorismes**:
   - **PPO-SB3** és el més susceptible al *catastrophic forgetting* per ser on-policy. Podria requerir un learning rate reduït al finetune.
   - **DQN-SB3** és el que més hauria de beneficiar-se'n gràcies al warmup curt i la baixa exploració inicial (la política transferida guia el replay buffer immediatament).
   - **DQN-RLCard** i **NFSP-RLCard** haurien de mostrar un perfil similar a DQN-SB3, però amb el sostre estructural de RLCard que ja vam observar a Fase 1.
5. **Comparació amb Fase 1**: cal també contrastar les corbes del grup control amb les de Fase 1. Com que Fase 2 elimina el self-play, és previsible que algun agent tingui **pitjors resultats al control que a Fase 1** — la qual cosa aïlla encara més el benefici del curriculum.

### Interpretació normativa

- Si el curriculum millora la mètrica final en ≥ 5 punts per almenys dos dels quatre algorismes, es pot afirmar que és una **tècnica efectiva a aquest entorn**.
- Si només accelera l'inici però el sostre final és el mateix, és un guany de **sample efficiency** però no de *skill* — útil per entrenaments limitats en temps però no per determinar el potencial màxim.
- Si perjudica sistemàticament, cal reconsiderar el disseny de les etapes: potser l'etapa 1 hauria de ser més curta, o el reward de mans hauria d'estar escalat diferent.

Els valors numèrics concrets i les conclusions finals s'extreuran del notebook `comparacio_fase2.ipynb` un cop els entrenaments finalitzin.
