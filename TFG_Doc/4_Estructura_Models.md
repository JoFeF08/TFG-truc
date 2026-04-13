# 4. Estructura de la Carpeta Models

Aquesta secció detalla l'estructura i organització de la subcarpeta `RL/models/`, que encapsula l'arquitectura de xarxes neuronals, els agents i els adaptadors necessaris per interoperar amb les dues llibreries de RL utilitzades al projecte: **RLCard** (per DQN i NFSP) i **Stable-Baselines3** (per DQN-SB3 i PPO-SB3).

```
RL/models/
├── core/
│   ├── feature_extractor.py   # CosMultiInput + ModelPreEntrenament
│   ├── obs_adapter.py         # Wrappers per aplanar observacions
│   └── loader.py              # Factoria de models a partir d'una spec
├── rlcard_legacy/
│   └── model_adapter.py       # Pont entre agents RLCard i TrucModel
├── sb3/
│   └── sb3_adapter.py         # SB3PPOEvalAgent per avaluació uniforme
└── model_propi/
    └── agent_regles.py        # Agent Rule-Based estocàstic
```

## `core/` — Peces centrals compartides

### `feature_extractor.py`

Defineix la xarxa extractora de característiques **`CosMultiInput`**, pensada per ser compartida per qualsevol algorisme que vulgui aprofitar representacions denses del Truc:

```python
class CosMultiInput(nn.Module):
    """
    Entrades:
      · cartes  : Tensor (batch, 6, 4, 9)  — Mapa de cartes 2D
      · context : Tensor (batch, 23)       — Informació contextual
    Sortida:
      · Tensor (batch, 256) — Representació latent del joc
    """
```

- **Branca CNN** sobre el mapa de cartes (6 canals × 4 pals × 9 rangs): dues capes `Conv2d` (kernel (1,3) i (3,3)) amb ReLU i `Flatten`, produint un vector de **320** dimensions.
- **Branca densa** sobre el context continu (23 valors): `Linear(23 → 32)` + ReLU.
- **Fusió**: concatenació `(320 + 32 = 352)` seguida d'un `Linear(352 → 256)` + ReLU.

També es defineix **`ModelPreEntrenament`**, que combina el `CosMultiInput` amb tres caps de regressió/classificació per al **pre-entrenament supervisat** (script `RL/entrenament/entrenamentEstatTruc/preentrenar_cos.py`):

| Cap | Mida sortida | Objectiu | Pèrdua |
|:--|:--|:--|:--|
| `cap_envido` | 1 | Puntuació d'envit de la mà | MSE |
| `cap_accions_legals` | 19 | Màscara d'accions legals | BCE |
| `cap_forces` | 3 | Força individual de cada carta a la mà | MSE |

Aquests tres objectius auxiliars forcen el cos a aprendre representacions útils del joc (coneixement d'envit, legalitat, força relativa) abans del pas a RL pur.

### `obs_adapter.py`

Conté utilitats per **aplanar l'observació** del diccionari nadiu `{obs_cartes, obs_context}` a un vector pla de 239 dimensions. S'utilitza via *monkey-patching* del mètode `_extract_state` de l'entorn:

```python
def wrap_env_aplanat(env):
    original = env._extract_state
    def _extract_patched(self, state):
        extracted = original(state)
        if isinstance(extracted['obs'], dict):
            extracted['obs'] = np.concatenate([
                extracted['obs']['obs_cartes'].flatten(),
                extracted['obs']['obs_context'],
            ], axis=0).astype(np.float32)
        return extracted
    env._extract_state = types.MethodType(_extract_patched, env)
    return env
```

Aquest pattern és necessari perquè tant els agents DQN/NFSP de **RLCard** com les `MlpPolicy` de **SB3** esperen un vector pla, no un diccionari multi-entrada.

### `loader.py`

Factory `crear_model(spec, env_config)` que construeix un model segons un diccionari d'especificació (`{"tipus": "...", "ruta": "..."}`). Retorna una instància que compleix el protocol **`TrucModel`** (mètode `triar_accio(estat) -> int`), utilitzat per la capa MVC (`ModelInteractiu`) per injectar bots a les partides interactives.

## `rlcard_legacy/` — Pont amb agents RLCard

### `model_adapter.py`

Inclou `_RLCardModelAdapter`, un *embolcall* (wrapper) molt prim que fa compatible qualsevol agent amb interfície `eval_step(state)` de RLCard (`DQNAgent`, `NFSPAgent`, `AgentRegles`) amb el protocol `TrucModel`:

```python
class _RLCardModelAdapter:
    def __init__(self, agent, state_extractor):
        self._agent = agent
        self._extract = state_extractor

    def triar_accio(self, estat):
        rlcard_state = self._extract(estat)
        action, _ = self._agent.eval_step(rlcard_state)
        return int(action)
```

La seva existència permet que el mateix loader i el mateix pipeline interactiu puguin jugar qualsevol agent de RLCard contra un jugador humà.

## `sb3/` — Pont amb Stable-Baselines3

### `sb3_adapter.py`

Defineix **`SB3PPOEvalAgent`**, un *adaptador invers* al cas anterior: agafa un model SB3 (PPO, DQN…) i li dóna la interfície `eval_step(state)` que espera la funció d'avaluació `evaluar_agent()` dels scripts d'entrenament.

```python
class SB3PPOEvalAgent:
    use_raw = False

    def __init__(self, model, n_actions: int = N_ACTIONS):
        self.model = model
        self.num_actions = n_actions

    def eval_step(self, state):
        obs = state['obs']
        if isinstance(obs, dict):
            obs_flat = np.concatenate(
                [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
            ).astype(np.float32)
        else:
            obs_flat = np.asarray(obs, dtype=np.float32)
        action, _ = self.model.predict(obs_flat[np.newaxis], deterministic=True)
        return int(action[0]), {}
```

Aquest adaptador és el que fa possible que a la Fase 1 els quatre algorismes comparats (DQN-RLCard, NFSP-RLCard, DQN-SB3, PPO-SB3) s'avaluin amb **la mateixa funció `evaluar_agent()`** i el mateix format de log. Vegeu [[5_Fase1_Entrenament]].

## `model_propi/` — Components desenvolupats específicament

Conté únicament l'agent propi codificat a mà pel projecte:

### `agent_regles.py` — L'Agent de Regles Estocàstic

Aquest agent és un dels pilars del projecte: s'utilitza simultàniament com a **oponent d'entrenament** i com a **oponent d'avaluació** per a totes dues fases. No té cap xarxa neuronal; tota la seva intel·ligència prové d'heurístiques manuals que combinen informació visible del joc (mà actual, historial de rondes, puntuació, cartes visibles del rival).

#### Estats de decisió

El mètode `eval_step(state)` divideix la decisió en tres branques segons l'estat de resposta pendent:

| Estat | Branca | Accions possibles |
|:--|:--|:--|
| `response_state_val == 2` | `_respondre_envit` | `vull_envit` / `fora_envit` / `apostar_envit` |
| `response_state_val == 1` | `_respondre_truc` | `vull_truc` / `fora_truc` / `apostar_truc` |
| Altrament | `_torn_normal` | `apostar_envit` / `apostar_truc` / `play_card_X` |

#### Funcions de lectura de l'estat

L'agent interpreta l'estat brut (`raw_obs`) mitjançant un conjunt d'helpers:

| Funció | Retorn | Descripció |
|:--|:--|:--|
| `_forces(raw)` | `list[int]` | Força numèrica de cada carta de la mà (via `TrucJudger.get_forca_carta`) |
| `_n_top(raw)` | `int` | Quantes cartes "top" té (`força ≥ 90`: 3s, manilles, asos forts, B11, O10) |
| `_best_force(raw)` | `int` | Força màxima de la mà |
| `_envit_score(raw)` | `int` | Punts d'envit de la mà (via `TrucJudger.get_envit_ma`) |
| `_rondes_info(raw)` | `(guanyades, perdudes)` | Nombre de rondes guanyades/perdudes pel propi equip |
| `_rival_carta_taula(raw)` | `int \| None` | Força de la carta que el rival acaba de jugar (si n'hi ha) |
| `_avantatge_puntuacio(raw)` | `int` | `punts_propis - punts_rivals` en el marcador |
| `_score_context(raw)` | `(per_propi, per_rival)` | Punts que falten a cada equip per guanyar |

#### Llindars adaptatius

Les decisions no són llistes de regles rígides: els llindars s'ajusten dinàmicament segons el context del marcador. Per exemple, a `_respondre_envit`:

```python
llindar_puja    = 30
llindar_accepta = 25

# Rival a punt de guanyar -> conservadors (no regalar punts)
if perill_rival >= 1.0:
    llindar_puja    += 8
    llindar_accepta += 8
elif perill_rival >= 0.6:
    llindar_puja    += 4
    llindar_accepta += 4

# Nosaltres a punt de guanyar -> agressius
if necessitat >= 1.0:
    llindar_puja    -= 10
    llindar_accepta -= 10
elif necessitat >= 0.6:
    llindar_puja    -= 5
    llindar_accepta -= 5
```

Aquest mateix patró s'aplica a `_respondre_truc` amb un `ajust_forca` que puja amb el nombre de pujades del rival (+10 per pujada) i baixa fort quan el truc decideix la partida (−20 si `necessitat ≥ 1.0`).

#### Estocasticitat

Aquest és un punt **essencial**. Si l'agent fos purament determinista, els algorismes de RL aprendrien ràpidament a explotar les seves respostes fixes i el guanyarien amb facilitat. Això faria que la mètrica d'avaluació `wr_regles` pujés artificialment i no reflectís la qualitat real de l'agent entrenat (se'n diu *reward hacking* contra un oponent fix). Per evitar-ho s'injecta soroll aleatori a tots els punts de decisió sensibles:

| Punt de decisió | Font d'aleatorietat | Efecte |
|:--|:--|:--|
| Llindars d'envit (resposta) | `rng.randint(-2, 2)` sobre `llindar_puja` i `llindar_accepta` | Zona grisa d'acceptació/rebuig |
| `ajust_forca` (resposta truc) | `rng.randint(-5, 5)` | Variabilitat en l'acceptació del truc |
| Acceptació truc després d'una ronda guanyada | `rng.random() < 0.20` | 20% d'acceptar sense mà forta (bluff defensiu) |
| Idem ronda 0 amb `n_top == 1` | `rng.random() < 0.20` | 20% d'acceptar amb mà mitjana |
| Envit inicial (llindars variables) | `rng.randint(28, 30)` i `rng.randint(24, 26)` | El mateix agent no aposta igual cada mà |
| Envit en zona grisa | `rng.random() < 0.50` si és mà, `< 0.35` en general | Decisions probabilístiques al límit |
| Bluff `apostar_truc` | `rng.random() < 0.12` (tenint ronda guanyada) | Pujades ocasionals sense cartes |
| Truc amb 2 top havent perdut ronda | `rng.random() < 0.40` | Agressivitat parcial |
| Carta inicial forta a ronda 0 | `rng.random() < 0.15` | Surt fort ocasionalment per enganyar |

L'efecte acumulat és que l'agent és coherent en el gruix del seu comportament (té una línia tàctica clara) però **no és mecànicament predictible**, la qual cosa el fa molt més semblant a un oponent humà i obliga els algorismes de RL a generalitzar.

#### Ús dins del projecte

- **Com a oponent d'entrenament**: a Fase 1, el 60% (DQN-RLCard / NFSP-RLCard) o la fracció dominant (DQN-SB3, PPO-SB3) de les partides es juga contra `AgentRegles`. A Fase 2, la barreja passa a ser **10% Random + 90% AgentRegles** (sense self-play).
- **Com a oponent d'avaluació**: totes les comparatives utilitzen `evaluar_agent()`, que mesura 100 partides contra `AgentRegles` i 50 contra `RandomAgent`, produint la mètrica `metric = 0.25·wr_random + 0.75·wr_regles` on el pes dominant és precisament la victòria contra l'agent de regles.
- **Com a model jugable**: via el `loader.py` i el `_RLCardModelAdapter`, es pot instanciar a `demo.py` per jugar contra ell des de la UI.
