# 13. Fase 3.5: Implementació i Resultats

Aquest document descriu la implementació tècnica de la Fase 3.5 definida a [[12_Fase35_MarcTeoric]]. L'objectiu és respondre:

> **Q2:** Dona valor el preentrenament supervisat del COS, i quin règim d'inicialització —frozen o finetune— és preferible respecte a scratch i respecte a no usar COS?

## Disseny experimental

El protocol és **mans** (`TrucGymEnvMa`, 24M steps) per a tots els runs. Es comparen quatre règims:

| Règim | Agent | Pesos inicials | `requires_grad(cos)` | Font |
|:--|:--|:--|:--|:--|
| Sense COS (MLP) | DQN-SB3 | — | — | F3 reutilitzat |
| Sense COS (MLP) | PPO-SB3 | — | — | F3 reutilitzat |
| Scratch | DQN-SB3 | Aleatoris | True | F3 reutilitzat |
| Scratch | PPO-SB3 | Aleatoris | True | F3 reutilitzat |
| Frozen | DQN-SB3 | Preentrenats | **False** | **nou** |
| Frozen | PPO-SB3 | Preentrenats | **False** | **nou** |
| Finetune | DQN-SB3 | Preentrenats | True | **nou** |
| Finetune | PPO-SB3 | Preentrenats | True | **nou** |

Els règims **sense COS** i **scratch** es reutilitzen directament de la Fase 3 (subcarpetes `{agent}_mans_mlp` i `{agent}_scratch` a `TFG_Doc/notebooks/3_feature_extractor/resultats/`). Només frozen i finetune requereixen nous runs.

Tots els runs comparteixen la mateixa funció d'avaluació (`evaluar_agent()`, 100+100 partides senceres cada 500 000 steps, sempre sobre `TrucGymEnv`) i els mateixos hiperparàmetres RL que la Fase 2.

## Prereqüisit: preentrenar el cos

Abans de llançar frozen/finetune cal disposar de `best_pesos_cos_truc.pth`. Si no existeix:

```bash
python RL/entrenament/entrenamentEstatTruc/preentrenar_cos.py
```

El fitxer es desarà a `RL/entrenament/entrenamentEstatTruc/registres/<timestamp>/models/best_pesos_cos_truc.pth`. Apuntar-se la ruta, que es passa com a argument obligatori a `run_fase35.sh`.

## Fitxers afectats

**Nous o modificats per aquesta fase:**

- `RL/entrenament/entrenamentsComparatius/fase3/fase35/entrenament_fase35.py` — script principal.
- `RL/entrenament/entrenamentsComparatius/fase3/fase35/run_fase35.sh` — llançador dels 4 runs nous.
- `TFG_Doc/notebooks/3_feature_extractor/35_init_cos/comparacio_fase35.ipynb` — notebook d'anàlisi.

**Reutilitzats sense modificació:**

- `RL/models/sb3/sb3_features_extractor.py` — `CosMultiInputSB3` amb `carregar_pesos_preentrenats()` i `congelar_cos()`.
- `RL/models/core/feature_extractor.py` — `CosMultiInput`.
- `RL/entrenament/entrenamentEstatTruc/preentrenar_cos.py` — genera `best_pesos_cos_truc.pth`.

## L'embolcall `CosMultiInputSB3`

L'adaptador fa de pont entre l'API de SB3 (que espera un `BaseFeaturesExtractor` sobre l'observació plana) i `CosMultiInput` (que espera dos tensors separats):

```python
class CosMultiInputSB3(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cos = CosMultiInput()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cartes = observations[:, :216].view(-1, 6, 4, 9)
        context = observations[:, 216:]
        return self.cos(cartes, context)

    def carregar_pesos_preentrenats(self, ruta: str) -> None:
        state = torch.load(ruta, map_location="cpu")
        self.cos.load_state_dict(state)

    def congelar_cos(self) -> None:
        for p in self.cos.parameters():
            p.requires_grad = False
```

## Script `entrenament_fase35.py`

Arguments:

```
--agent      {ppo_sb3, dqn_sb3}
--variant    {scratch, frozen, finetune}
--pesos_cos  <ruta>   # obligatori per frozen i finetune
--steps      total steps (defecte 24 000 000)
--save_dir   carpeta de sortida
```

### Construcció i aplicació del règim

```python
# Tots els règims usen CosMultiInputSB3 (scratch inclòs)
policy_kwargs = dict(
    features_extractor_class=CosMultiInputSB3,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=...,
)
model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)  # o PPO

# Aplicació del règim — SEMPRE després del constructor
if args.variant in ("frozen", "finetune"):
    model.policy.features_extractor.carregar_pesos_preentrenats(args.pesos_cos)

if args.variant == "frozen":
    model.policy.features_extractor.congelar_cos()
    entrenables = [p for p in model.policy.parameters() if p.requires_grad]
    model.policy.optimizer = torch.optim.Adam(entrenables, lr=LR)
```

**Per què després del constructor**: SB3 inicialitza els submòduls dins `model.__init__()`. Si carreguéssim els pesos abans, SB3 els sobreescriuria. L'ordre correcte és: construir → carregar → congelar → reconstruir optimitzador.

**Per què reconstruir l'optimitzador al frozen**: si l'optimitzador original manté tots els paràmetres, PyTorch calcularia gradients (zeros) i actualitzaria els *momentum buffers* dels paràmetres congelats innecessàriament.

## Script `run_fase35.sh`

```bash
bash RL/entrenament/entrenamentsComparatius/fase3/fase35/run_fase35.sh <PESOS_COS> [STEPS]
```

Llança 4 runs nous en seqüència (scratch s'omet perquè es reutilitza de F3):

```
dqn_sb3 + frozen
dqn_sb3 + finetune
ppo_sb3 + frozen
ppo_sb3 + finetune
```

Sortida: `TFG_Doc/notebooks/3_feature_extractor/resultats/` amb subcarpetes `{agent}_{variant}/`.

## Layout de registres

```
TFG_Doc/notebooks/3_feature_extractor/resultats/
├── dqn_sb3_scratch/          ← reutilitzat de F3
│   └── training_log.csv
├── ppo_sb3_scratch/          ← reutilitzat de F3
├── dqn_sb3_mans_mlp/         ← reutilitzat de F3 (sense COS)
├── ppo_sb3_mans_mlp/         ← reutilitzat de F3 (sense COS)
├── dqn_sb3_frozen/           ← nou
│   ├── training_log.csv
│   ├── best_frozen.zip
│   └── final_frozen.zip
├── ppo_sb3_frozen/           ← nou
├── dqn_sb3_finetune/         ← nou
└── ppo_sb3_finetune/         ← nou
```

## Notes metodològiques

- **Entorn**: `TrucGymEnvMa` (episodi = una mà, ~15 steps), oponent 10% Random + 90% AgentRegles.
- **Avaluació**: sempre sobre partides senceres, cada 500 000 steps. `eval_metric = 0.25 × WR_random + 0.75 × WR_regles`.
- **PPO-SB3**: `NUM_ENVS = 32`, `n_steps = 256`, `batch_size = 2048`.
- **DQN-SB3**: entorn únic, `learning_starts = 50 000`, `buffer_size = 200 000`, `exploration_fraction = 0.3`.
- **Pesos preentrenats**: `best_pesos_cos_truc.pth` generat per `preentrenar_cos.py` sobre 200 000 mostres, targets: envit (MSE), accions legals (BCE), forces (MSE).

## Resultats (run complet, 24M steps)

Runs executats l'abril 2026. Dades a `TFG_Doc/notebooks/3_feature_extractor/resultats/`.

### Taula resum (protocol mans)

| Règim | DQN-SB3 (pic) | PPO-SB3 (pic) |
|:--|--:|--:|
| Sense COS (MLP) | 85.8% | 87.2% |
| Scratch | 82.2% | 89.0% |
| Frozen | **91.2%** | **89.5%** |
| Finetune | 88.5% | 88.0% |

### Lectures principals

**Frozen vs scratch**: el preentrenament aporta guanys clars per a DQN (+9 pp) i lleugers per a PPO (+1 pp). Per a DQN, el cos preentrenat i congelat actua com a extractor estable que permet a la Q-network aprendre sense interferències.

**Frozen vs sense COS**: frozen supera el MLP en tots dos agents. Confirma que el preentrenament supervisat, no l'arquitectura sola, és el que marca la diferència per a DQN. Per a PPO, la diferència és modesta (+3 pp).

**Finetune vs frozen**: per a DQN el finetune és **inferior** al frozen (88.5% vs 91.2%, −2.7 pp). El gradient del RL degrada parcialment la representació preentrenada malgrat el *replay buffer*. Per a PPO el finetune queda lleument per sota del frozen (88.0% vs 89.5%), consistent amb un forgetting moderat induït pels múltiples `n_epochs` per actualització.

**Scratch vs sense COS**: la diferència és <3 pp en tots dos agents i no és consistent (DQN scratch < MLP; PPO scratch > MLP). Confirma el que va establir la Fase 3: l'arquitectura COS sense preentrenament no supera sistemàticament el MLP al protocol mans.

### Interpretació global

El preentrenament supervisat aporta valor real, especialment per a DQN. El règim **frozen és el millor per a DQN** (91.2%, +5.4 pp sobre MLP, +9 pp sobre scratch); el finetune no millora el frozen perquè el gradient RL degrada parcialment la representació preentrenada. Per a PPO, tots els règims convergen entre 87–89% i el scratch és pràcticament equivalent a frozen, de manera que el preentrenament no aporta millora significativa. El MLP estàndard és una base molt competitiva al protocol mans, i superar-lo requereix el preentrenament, no l'arquitectura sola.

## Procediment d'ús

1. **Preentrenar el cos**:
   ```bash
   python RL/entrenament/entrenamentEstatTruc/preentrenar_cos.py
   ```

2. **Llançar els runs nous** (frozen + finetune × 2 agents):
   ```bash
   bash RL/entrenament/entrenamentsComparatius/fase3/fase35/run_fase35.sh \
     RL/entrenament/entrenamentEstatTruc/registres/<timestamp>/models/best_pesos_cos_truc.pth
   ```

3. **Smoke test**:
   ```bash
   bash RL/entrenament/entrenamentsComparatius/fase3/fase35/run_fase35.sh <PESOS> 200000
   ```

4. **Anàlisi**: `TFG_Doc/notebooks/3_feature_extractor/35_init_cos/comparacio_fase35.ipynb`. El notebook carrega les 8 sèries (4 reutilitzades + 4 noves) i mostra les corbes i la taula resum.

## Enllaços creuats

- [[12_Fase35_MarcTeoric]] — fonaments teòrics del preentrenament i dels quatre règims.
- [[10_Fase3_MarcTeoric]] — Q1 (arquitectura COS vs MLP), motivació de la Q2.
- [[11_Fase3_Implementacio]] — font de les sèries scratch i sense COS reutilitzades.
- [[9_Checkpoint1]] — selecció dels models i pla de fases.
