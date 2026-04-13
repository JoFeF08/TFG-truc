# 3. Entorns de Simulació per RL


`TrucEnv` és l'entorn que adapta la lògica del joc (`TrucGame`) a la interfície estàndard de **RLCard**. La seva funció principal és **transformar l'estat del joc** (diccionaris llegibles per humans) en **vectors numèrics** (observacions) que un agent de Reinforcement Learning pot processar.

##### Arquitectura

```mermaid
flowchart TD
    %% Connexió Superior
    Agent["Agent RL"] <-->|"observació, acció"| Env["TrucEnv<br/>(env.py)"]
    
    %% Connexió de Lògica
    Env --->|"extracció estat, step()"| Game["TrucGame<br/>(game.py)"]
    
    %% Nucli Intern
    subgraph "Motor Lògic Aïllat"
        Game -.-> Dealer["TrucDealer"]
        Game -.-> Judger["TrucJudger"]
        Game -.-> Player["TrucPlayer"]
    end
    
    %% Estils
    style Env fill:#0984e3,color:#fff
    style Game fill:#00b894,color:#fff
    style Agent fill:#e84393,color:#fff
```

`TrucEnv` hereta de `rlcard.envs.Env`, que proporciona el bucle estàndard `reset()` -> `step()` -> `get_payoffs()`.

##### Constructor de l'Entorn `__init__`

```python
TrucEnv(config)
```

##### Paràmetres de configuració (diccionari `config`)

| Clau                | Defecte | Descripció                                                                                           |
| :------------------ | :------ | :---------------------------------------------------------------------------------------------------- |
| `num_jugadors`    | 2       | Nombre de jugadors                                                                                    |
| `cartes_jugador`  | 3       | Cartes per jugador                                                                                    |
| `puntuacio_final` | 24      | Punts per guanyar                                                                                     |
| `senyes`          | False   | Activar fase de senyals                                                                               |
| `player_class`    | None    | Classe dels jugadors (e.g.`HumanPlayer`) o **diccionari** `{id: Classe}` per barrejar tipus |
| `allow_step_back` | False   | Permetre desfer passos (heretat d'RLCard)                                                             |
| `seed`            | None    | Seed per reproducibilitat                                                                             |
| `verbose`         | False   | Mode de debug                                                                                         |

---

##### Variables Internes de l'Entorn

###### Mapeig de Cartes i Senyals

| Variable       | Tipus              | Descripció                                             |
| :------------- | :----------------- | :------------------------------------------------------ |
| `cartes`     | `list[str]`      | Llista de totes les cartes del joc (36 cartes)          |
| `carta_map`  | `dict[str, int]` | Mapa `carta -> index` per codificar cartes a one-hot  |
| `signal_map` | `dict[str, int]` | Mapa `senya -> index` per codificar senyals a one-hot |

###### Dimensions de l'Espai d'Estat

| Variable                 | Valor (exemple 2J, 3C, -senyes) | Descripció                                                    |
| :----------------------- | :------------------------------ | :------------------------------------------------------------- |
| `num_cartes`           | 36                              | Cartes úniques a la baralla                                   |
| `espai_joc_cartes`     | 37                              | `num_cartes + 1` (inclou slot "buit")                        |
| `espai_hist_cartes`    | 6                               | `num_jugadors * cartes_jugador` (slots historial)            |
| `espai_senya`          | 10                              | `len(ACTIONS_SIGNAL) + 1` (9 senyals + buit)                 |
| `espai_hist_senyes`    | 0 o 6                           | 0 si `senyes=False`, sinó `num_jugadors * cartes_jugador` |
| `espai_info_publica`   | 10                              | Puntuacions + apostes + situació                              |
| **`state_size`** | **343** (sense senyes)    | Mida total del vector d'observació                            |

###### Espais per a RLCard

| Variable         | Descripció                                                                 |
| :--------------- | :-------------------------------------------------------------------------- |
| `state_shape`  | `[[state_size]] * num_jugadors` - dimensions de l'observació per jugador |
| `action_shape` | `[[len(ACTION_LIST)]] * num_jugadors` - 18 accions possibles              |

---

##### Estructura de l'Observació `_extract_state(state)`

Transforma el diccionari d'estat del joc en un tensor numèric i context de tipus `np.float32`. L'estructura s'ha canviat a un format **multi-entrada**:

L'observació extreta és un diccionari amb dues claus rellevants globals sota la sub-clau `obs`:

1. **`obs_cartes`**: Un tensor 3D de dimensions `(6 canals, 4 pals, 9 rangs)`. Les posicions marcades són valors one-hot a l'índex corresponent.
2. **`obs_context`**: Un tensor 1D (vector) de 23 dimensions flotants (`(23,)`) amb variables contínues (escalables) i valors one-hot del context.

**Total variables `state_size`**: `6 * 4 * 9 + 23` = **239** mides.

###### 1. Canals de Cartes (`obs_cartes` tensor (6,4,9))

Codificació on-hot segons la utilitat i propietat de destí per cada carta segons l'observador:

- **Canal 0**: Mà actual del jugador
- **Canals 1-4**: Historial de les cartes jugades d'ell mateix (1), Rival 1 (2), Company en mode 4 jugadors (3) i Rival 2 (4).
- **Canal 5**: Cartes assenyalades per les senyes (si actives) a l'historial del company.

###### 2. Vector de Context (`obs_context` tensor (23,))

| Posició | Contingut                 | Descripció i format                      |
| :------- | :------------------------ | :---------------------------------------- |
| 0        | `puntuacio` Equip Propi | Escalat `punts / 24`                    |
| 1        | `puntuacio` Equip Rival | Escalat `punts / 24`                    |
| 2        | `estat_truc.level`      | Nivell actual del Truc (`level / 24`)   |
| 3        | `estat_envit.level`     | Nivell actual de l'Envit (`level / 24`) |
| 4        | `fase_torn`             | Fase actual (0 o 1)                       |
| 5        | `comptador_ronda`       | Rondes completades (`ronda / cartes`)   |
| 6-9      | `ma_offset`             | Qui és mà (one-hot relatiu)             |
| 10-13    | `truc_owner_offset`     | Qui ha cantat Truc (one-hot relatiu)      |
| 14-16    | `envit_owner_offset`    | Qui ha cantat Envit (one-hot relatiu)     |
| 17       | `rondes_guanyades_prop` | Rondes guanyades propi (`rondes / 3`)    |
| 18       | `rondes_guanyades_riv`  | Rondes guanyades rival (`rondes / 3`)    |
| 19       | `winner_r1`             | Guanyador R1 (1.0=jo, -1.0=rival, 0.0=empat/no jugada) |
| 20       | `winner_r2`             | Guanyador R2 (1.0=jo, -1.0=rival, 0.0=empat/no jugada) |
| 21       | `envit_accepted`        | 1.0 si acceptat, 0.0 si no               |
| 22       | `response_state`        | 0.0/0.5/1.0 (NO/TRUC/ENVIT PENDING)      |

###### Retorn de `_extract_state`

```python
{
    'obs': {
        'obs_cartes': np.array([...]),  # Tensor (6,4,9) per la xarxa neuronal espacial
        'obs_context': np.array([...])  # Vector de context (23,) de variables contínues 
    },
    'legal_actions': OrderedDict,     # Accions legals com a OrderedDict
    'raw_obs': state,                 # Estat original (diccionari llegible)
    'raw_legal_actions': ['passar', 'apostar_truc', ...],  # Noms de les accions
    'action_record': self.action_recorder  # Historial d'accions (d'RLCard)
}
```

---

##### Altres Mètodes i Fites

L'entorn cedeix tota l'autonomia al mecanisme subjacent (`TrucGame`) per al càlcul de les recompenses (per més detalls relatius al disseny asimètric dels rewards, refèrir-se a [[2_Logica_Joc]]):

1. **Reward final `get_payoffs()`**: Obté la referència matemàtica de +1.0 o -1.0 directament recollint-la del motor de joc.
2. **El Problema del Reward Intermedi a RLCard**: La llibreria RLCard està dissenyada per només emprar el *Payoff* final en l'última transició, assignant recursivament una recompensa de `0.0` a qualsevol step intermedi. Per solucionar l'Sparse Reward, el projecte implementa un encaminador personal.

###### `reorganize_amb_rewards(trajectories, payoffs)`

Funció personalitzada (tècnica de *Reward Injection*) dissenyada per sobreescriure el tractament de trajectòries per defecte d'RLCard `[estat_actual, acció, state_next]`.

L'encaminador llegeix la propietat `reward_intermedis` continguda de forma bruta en els raw_obs i transforma la memòria RAM local substituint aquells zero cecs precaris inserint manualment la recompensa que tocava per cada segment individual del procés, deixant un tensor impecable llest per a ser empassat pels aprenents RL d'alt rendiment:  `[estat_actual, acció, *recompensa_intermedia_real*, next_state_raw, done]`.

###### `_decode_action(action_id)` -> Descodificar Acció

Converteix un índex d'acció numèric al seu nom string:

```python
ACTION_LIST[action_id]  # e.g. 0 -> 'play_card_0', 4 -> 'apostar_truc'
```

###### `_get_legal_actions()` -> Accions Legals

Aquest mètode de l'entorn delega directament a `TrucGame.get_legal_actions()`, que és on realment es calcula la llista d'accions permeses. Retorna una **llista d'índexs** (enters) que representen les accions vàlides.


## Wrappers Gymnasium per Stable-Baselines3

Els adaptadors RLCard (`TrucEnv`/`TrucEnvMa`) treballen amb un loop seqüencial propi i una interfície multi-agent basada en `set_agents()`. Per entrenar amb **Stable-Baselines3 (SB3)** — que espera la interfície estàndard `gymnasium.Env` (`reset()` / `step(action)` sense saber res dels oponents) — s'han creat dos wrappers:

| Fitxer | Classe | Motor subjacent |
|:--|:--|:--|
| `joc/entorn/gym_env.py` | `TrucGymEnv` | `TrucEnv` (partides senceres) |
| `joc/entorn_ma/gym_env_ma.py` | `TrucGymEnvMa` | `TrucEnvMa` (mans) |

### Arquitectura

Els dos wrappers segueixen el mateix patró:

```python
class TrucGymEnv(gymnasium.Env):
    def __init__(self, env_config: dict, opponent=None, learner_pid: int = 0):
        self.rlcard_env = TrucEnv(env_config)
        self.learner_pid = learner_pid
        self.opponent    = opponent or RandomAgent(num_actions=n_actions)
        ...
        self.observation_space = spaces.Box(-inf, +inf, shape=(obs_dim,), dtype=float32)
        self.action_space      = spaces.Discrete(n_actions)
```

La idea clau és que **un sol jugador és l'aprenent** (`learner_pid`), i l'altre jugador es mou automàticament dins del `step()` del wrapper, cridant `opponent.eval_step(state)`. Així l'algorisme de SB3 veu un entorn monojugador convencional.

### Observació Aplanada

A diferència de `TrucEnv`, que retorna l'observació com un diccionari `{'obs_cartes': (6,4,9), 'obs_context': (23,)}`, els wrappers **aplanen** l'observació en un únic vector de **239 dimensions** (`6*4*9 + 23 = 239`) per poder usar una `MlpPolicy` de SB3:

```python
def _flatten_obs(self, state) -> np.ndarray:
    obs = state['obs']
    if isinstance(obs, dict):
        return np.concatenate(
            [obs['obs_cartes'].flatten(), obs['obs_context']], axis=0
        ).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)
```

### Loop de `step()`

Un `step()` del wrapper:

1. Aplica l'acció de l'aprenent al motor subjacent.
2. Si la partida ha acabat, retorna el *payoff* final com a recompensa (`done=True`).
3. Altrament, llegeix el *reward shaping* intermedi (`reward_intermedis[equip_aprenent] × 5.0`) i el suma a la recompensa del pas.
4. Executa accions de l'oponent fins que torni a tocar a l'aprenent (bucle intern), acumulant els rewards intermedis pel camí.
5. Retorna `(obs_aprenent, reward_acumulat, done, truncated, info)`.

El factor ×5.0 aplicat al *reward intermedi* serveix per donar-li més pes respecte al soroll natural del *policy gradient* de PPO (sense ell els rewards de shaping —tots < 1— quedarien ofegats per la variància del gradient).

### Ús amb SB3 Vectoritzat

Aquests wrappers s'instancien dins factories que es passen a `stable_baselines3.common.vec_env.SubprocVecEnv`. Exemple simplificat extret de Fase 1:

```python
def _make_gym_env_fn(opponent_type, learner_pid, seed):
    def _init():
        cfg = ENV_CONFIG.copy()
        cfg['seed'] = seed
        opp = RandomAgent(n_actions) if opponent_type == 'random' \
            else AgentRegles(n_actions, seed=seed+1000)
        return TrucGymEnv(cfg, opponent=opp, learner_pid=learner_pid)
    return _init

env_fns = [_make_gym_env_fn('regles', i%2, SEED+i) for i in range(48)]
vec_env = SB3SubprocVecEnv(env_fns)
```

D'aquesta manera, la paral·lelització s'aconsegueix directament amb el `SubprocVecEnv` natiu de SB3 (48 subprocessos per defecte en PPO), sense necessitat de cap entorn paral·lel propi.

### Diferència entre `TrucGymEnv` i `TrucGymEnvMa`

Codi gairebé idèntic: l'única cosa que canvia és el motor subjacent (`TrucEnv` vs `TrucEnvMa`). Amb `TrucGymEnv` cada episodi és una partida sencera fins a 24 punts; amb `TrucGymEnvMa` cada episodi és una sola mà. Això és crucial per la Fase 2 — vegeu [[7_Entorn_Ma_Curriculum]] i [[8_Fase2_Curriculum]].

