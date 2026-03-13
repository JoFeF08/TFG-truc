# Resum Entorn

**Josep Ferriol Font**

## Resum de l'Estructura

El projecte **TFG-truc** implementa el joc de cartes Truc amb una arquitectura modular preparada per a l'entrenament d'agents de Reinforcement Learning (RL).

### Visió General

L'objectiu principal és proporcionar un entorn robust per simular partides de Truc i entrenar agents intel·ligents utilitzant la llibreria RLCard.

### Estructura de Directoris

- `joc/`: Nucli del joc sota una arquitectura MVC.
  - `entorn/`: Motor de simulació per als agents de Reinforcement Learning.
    - `game.py`: Motor lògic del joc. Gestiona els estats de la partida, les regles, la progressió de rondes i mans, i les apostes (Truc i Envit).
    - `env.py`: Adaptador per a RLCard. Tradueix l'estat intern del joc a observacions numèriques i exposa l'API estàndard de l'entorn (`reset`, `step`).
    - `cartes_accions.py`: Fitxer de constants compartides.
    - `rols/`: Implementacions dels rols de l'entorn.
      - `dealer.py`: Gestiona la baralla (creació, barreja i repartiment de cartes als jugadors).
      - `judger.py`: Conté la lògica d'arbitratge: determina el guanyador d'una ronda, d'una mà i de l'envit.
      - `player.py`: Classe base que defineix la interfície comuna dels agents o jugadors aleatoris.
  - `controlador/`: Gestors i classes de control (arquitectura MVC) que interactuen amb la simulació.
  - `vista/`: Interfícies gràfiques o de consola (MVC) on s'hi desenvolupen i mostren les partides (inclou visualitzador per consola i ascriptori).
- `RL/`: Flux de treball de Reinforcement Learning.
  - `entrenament/`: Scripts i codi dedicat a realitzar els entrenaments i l'avaluació.
  - `models/`: Models amb els pesos i punts de control per als diferents agents.
  - `notebooks/`: Llibretes Jupyter per elaborar proves, estadístiques i avaluacions sobre l'entrenament dels agents.
  - `tools/`: Utilitats generals pel tractament de les simulacions i xarxes neuronals de l'entorn.
- `demo.py`: Script de demostració interactiu per jugar. Permet configurar el nombre de jugadors, tipus de partides, rols (Humà o Agent) a través del terminal o finestra d'escriptori.
- `doc/`: Documentació tècnica en Markdown i/o LaTeX.

## Arquitectura del joc del Truc

Aquest projecte implementa una arquitectura Model—Vista—Controlador (MVC) que separa clarament tres rols:

- **Controlador**: orquestra el flux; demana dades i accions a la vista, consulta i modifica l'estat a través del model, i no depèn de cap implementació concreta de vista o model.
- **Model**: lògica del joc (estat, regles, jugadors bots).
- **Vista**: entrada i sortida amb l'usuari (configuració, mostrat d'estat, selecció d'accions, resultats).

El controlador depèn només d'**interfícies** (contractes): qualsevol vista i qualsevol model que implementin aquests contractes poden ser utilitzats sense canviar el controlador.

### El Controlador

El **controlador** (`controlador/controlador.py`) és l'únic punt que coneix tant la vista com el model. La seva funció és:

1. Obtenir la configuració inicial mitjançant la vista.
2. Inicialitzar el model amb aquesta configuració.
3. En un bucle, mentre la partida no hagi acabat:
   - Saber quin jugador juga (model).
   - Si és humà: mostrar l'estat (vista), demanar una acció (vista), aplicar-la (model) i informar la vista.
   - Si és bot: obtenir l'acció del model, aplicar-la i informar la vista.
4. Un cop acabada la partida, obtenir el resultat del model i mostrar-lo per la vista.
5. Preguntar si es vol repetir (vista) i, si cal, mostrar el missatge de sortida.

El controlador **no** conté lògica de joc ni lògica d'interfície: només coordina crides entre vista i model segons el contracte.

#### Contracte amb el Model

El controlador parla amb el model a través del **protocol** `Model` (`controlador/interficie_model.py`). Qualsevol classe que implementi aquests mètodes pot fer de model.

| Mètode                        | Signatura                                       | Descripció                                                                                             |
| :----------------------------- | :---------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| `iniciar`                    | `(self, config: dict) -> None`                | Crea i inicialitza una partida amb la configuració donada.                                             |
| `get_estat`                  | `(self, jugador_id: int) -> dict`             | Retorna l'estat visible per al jugador amb aquest id.                                                   |
| `get_jugador_actual`         | `(self) -> int`                               | Retorna l'id del jugador que ha de jugar ara.                                                           |
| `es_huma`                    | `(self, jugador_id: int) -> bool`             | Indica si el jugador és humà (l'accions vindran de la vista).                                         |
| `get_accio_bot`              | `(self, jugador_id: int) -> tuple[int, str]`  | Retorna `(codi_accio, nom_accio)` triada pel bot.                                                     |
| `aplicar_accio`              | `(self, accio: int) -> None`                  | Aplica l'acció amb codi donat i avança l'estat del joc.                                               |
| `get_guanyador_envit_recent` | `(self) -> tuple[int, int, list[int]] \| None` | Retorna `(equip, punts, punts_detall)` de l'envit que s'acaba de tancar, si n'hi ha.                  |
| `get_guanyador_truc_recent`  | `(self) -> tuple[int, int] \| None`            | Retorna `(equip, punts)` del truc (mà) que s'acaba de tancar, si n'hi ha.                            |
| `es_final`                   | `(self) -> bool`                              | Indica si la partida ha acabat.                                                                         |
| `get_resultat`               | `(self) -> dict`                              | Retorna un diccionari amb `score` i `payoffs` (per exemple `{'score': [...], 'payoffs': [...]}`). |

La implementació actual del contracte `Model` és `ModelInteractiu` (`controlador/model_interactiu.py`), que encapsula el joc (`TrucGame`) i adapta les seves crides als mètodes del protocol.

#### Contracte amb la Vista

El controlador parla amb la vista a través del **protocol** `Vista` (`vista/interficie_vista.py`). Qualsevol classe que implementi aquests mètodes pot fer de vista.

| Mètode                     | Signatura                                                           | Descripció                                                                                       |
| :-------------------------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------ |
| `demanar_config`          | `(self) -> dict`                                                  | Demana (per la UI) i retorna la configuració del joc.                                            |
| `mostrar_estat`           | `(self, estat: dict) -> None`                                     | Mostra l'estat actual del joc (taula, mà, puntuació, info de ronda, etc.).                      |
| `escollir_accio`          | `(self, accions_legals: list, estat: dict) -> int`                | Presenta les accions legals; l'usuari en tria una; retorna el**codi** d'acció (enter).     |
| `mostrar_accio`           | `(self, jugador_id: int, nom_accio: str, es_bot: bool) -> None`   | Informa quina acció ha fet un jugador. Si `es_bot=True`, la vista pot afegir un retard visual. |
| `mostrar_guanyador_envit` | `(self, equip: int, punts: int, punts_detall: list[int]) -> None` | Comunica qui ha guanyat l'envit temporal parcial de la mà en joc.                                |
| `mostrar_guanyador_truc`  | `(self, equip: int, punts: int) -> None`                          | Comunica qui ha guanyat el truc i el repartiment sota una mà temporal tancada.                   |
| `mostrar_fi_partida`      | `(self, score: list, payoffs: list) -> None`                      | Mostra el resultat final (marcador i payoffs).                                                    |
| `demanar_repetir`         | `(self) -> bool`                                                  | Pregunta si es vol jugar una altra partida. Retorna `True` si sí.                              |
| `mostrar_sortint`         | `(self) -> None`                                                  | Indica que l'usuari surt de l'aplicació.                                                         |

### Model

#### Motor Lògic (`game.py`)

`TrucGame` és la classe central que implementa tota la lògica del joc del Truc. Gestiona l'estat complet de la partida: jugadors, cartes, apostes (Truc i Envit), rondes, i puntuació global. Actua com a **motor del joc**, independent de la interfície d'entorn (`env.py`).

##### Constructor `__init__`

```python
TrucGame(num_jugadors=2, cartes_jugador=3, senyes=False, puntuacio_final=24, player_class=TrucPlayer)
```

##### Paràmetres de configuració

| Paràmetre          | Defecte        | Descripció                                                                                       |
| :------------------ | :------------- | :------------------------------------------------------------------------------------------------ |
| `num_jugadors`    | 2              | Nombre de jugadors                                                                                |
| `cartes_jugador`  | 3              | Cartes repartides per mà                                                                         |
| `senyes`          | False          | Activa la fase de senyals                                                                         |
| `puntuacio_final` | 24             | Punts per guanyar la partida                                                                      |
| `player_class`    | `TrucPlayer` | Classe o**diccionari** `{id: Classe}` dels jugadors (permet partides mixtes Human vs Bot) |
| `verbose`         | False          | Mode de debug per consola                                                                         |

---

##### Variables Internes del Joc

###### Jugadors i Rols

| Variable      | Tipus                | Descripció                                   |
| :------------ | :------------------- | :-------------------------------------------- |
| `players`   | `list[TrucPlayer]` | Llista d'instàncies de jugadors              |
| `dealer`    | `TrucDealer`       | Gestiona la baralla i el repartiment          |
| `judger`    | `TrucJudger`       | Determina guanyadors de rondes, envits i mans |
| `np_random` | `RandomState`      | Generador aleatori per reproducibilitat       |

###### Control del Torn

| Variable           | Tipus             | Descripció                                                                |
| :----------------- | :---------------- | :------------------------------------------------------------------------- |
| `ma`             | `int`           | ID del jugador que és**mà** (comença jugant). Rota cada mà nova. |
| `current_player` | `int`           | ID del jugador que ha d'actuar ara                                         |
| `turn_player`    | `int`           | Jugador amb el torn "real" (es restaura després d'una aposta)             |
| `turn_phase`     | `int`           | Fase actual del torn: 0=Senyals, 1=Apostes i Joc                           |
| `response_state` | `ResponseState` | Estat de resposta pendent                                                  |

**Significat dels valors de l'estat de resposta pendent:**

| Valor                 | Significat                                                       |
| :-------------------- | :--------------------------------------------------------------- |
| `NO_PENDING` (0)    | No hi ha cap aposta pendent de resposta                          |
| `TRUC_PENDING` (1)  | S'ha cantat Truc i s'espera resposta (Vull / Fora / Re-apostar)  |
| `ENVIT_PENDING` (2) | S'ha cantat Envit i s'espera resposta (Vull / Fora / Re-apostar) |

**Important:** Mentre hi ha una resposta pendent, el jugador contrari **només** pot respondre a l'aposta (vull/fora/re-apostar). No pot jugar cartes ni fer altres accions.

###### Puntuació i Historial

| Variable          | Tipus           | Descripció                                        |
| :---------------- | :-------------- | :------------------------------------------------- |
| `score`         | `list[int]`   | Puntuació global `[equip_0, equip_1]`           |
| `hist_cartes`   | `list[tuple]` | Historial de cartes jugades:`(player_id, carta)` |
| `hist_senyes`   | `list[tuple]` | Historial de senyals:`(player_id, senya)`        |
| `round_counter` | `int`         | Nombre de rondes completades a la mà actual       |
| `cartes_ronda`  | `list[tuple]` | Cartes jugades a la ronda en curs                  |
| `ronda_winners` | `list[int]`   | Guanyadors de cada ronda (-1 = empat)              |
| `payoffs`       | `list[int]`   | Recompenses finals per a l'entorn RL               |

###### Estat del Truc (Aposta de cartes)

| Variable                | Tipus   | Descripció                                                               |
| :---------------------- | :------ | :------------------------------------------------------------------------ |
| `truc_level`          | `int` | Nivell actual del Truc (punts en joc per cartes)                          |
| `truc_owner`          | `int` | Qui ha cantat l'última aposta de Truc (-1 = ningú)                      |
| `previous_truc_level` | `int` | Nivell anterior (per si diuen "Fora", saben quants punts guanya el rival) |

**Seqüència d'escalada del Truc:**
`1 (per defecte) -> 3 (Truc) -> 6 (Retruc) -> 9 (Val Nou) -> 24 (Joc Fora)`

###### Estat de l'Envit (Aposta d'envit)

| Variable                 | Tipus    | Descripció                                          |
| :----------------------- | :------- | :--------------------------------------------------- |
| `envit_level`          | `int`  | Nivell actual de l'Envit                             |
| `envit_owner`          | `int`  | Qui ha cantat l'últim Envit (-1 = ningú)           |
| `envit_accepted`       | `bool` | Si l'envit ha estat acceptat                         |
| `previous_envit_level` | `int`  | Nivell anterior (per calcular punts si diuen "Fora") |

**Seqüència d'escalada de l'Envit:**
`0 (no cantat) -> 2 (Envit) -> 4 (Un mes) -> 6 (Dos mes) -> Tots (Falta Envit)`

**Regla "Tots" (Falta Envit):**
Si cap equip supera els 12 punts, val 24 (guanya partida). Si algun equip supera els 12, val `24 - puntuacio_del_lider`.

---

##### Mètodes Principals

###### `init_game()` -> Inicialitzar Partida

1. Crea els jugadors amb `player_class`
2. Crea el dealer i el judger
3. Barreja i reparteix cartes
4. Inicialitza totes les variables d'estat
5. Retorna `(estat, current_player)`

###### `step(action)` -> Avançar el Joc

El mètode principal. Rep una acció (int o string) i actualitza l'estat.

**Flux de decisions:**

```text
step(action)
      |
      +--[Resposta pendent?]
      |        |
      |   ENVIT_PENDING --> vull_envit | fora_envit | apostar_envit
      |   TRUC_PENDING  --> vull_truc  | fora_truc  | apostar_truc
      |
      +--[NO_PENDING - Torn normal]
               |
               +-- Fase 0 (Senyals) --> passar  (avanca a Fase 1)
               |                    --> senya_* (registra, avanca a Fase 1)
               |
               +-- Fase 1 (Apostes) --> apostar_envit --> ENVIT_PENDING
               |                    --> apostar_truc  --> TRUC_PENDING
               |                    --> fora_truc     --> Rival guanya, reset ma
               |                    --> passar        --> avanca a Fase 2
               |                    --> play_card_X   --> (Es pot jugar directament si escau la lògica)
               |
               +-- Fase 2 (Jugar)   --> play_card_X
                                         |
                                    Fi ronda? -> Fi ma? -> Fi partida?
```

###### `get_state(player_id)` -> Construir Estat

Retorna un diccionari amb tota la informació visible pel jugador.

###### `get_legal_actions()` -> Accions Legals

Determina quines accions pot fer el jugador actual segons la situació:

| Situació         | Accions disponibles                                                          |
| :---------------- | :--------------------------------------------------------------------------- |
| `ENVIT_PENDING` | `vull_envit`, `fora_envit`, (+ `apostar_envit` si `level <= 6`)      |
| `TRUC_PENDING`  | `vull_truc`, `fora_truc`, (+ `apostar_truc` si `level < 24`)         |
| Fase 0 (Senyals)  | `passar` + totes les `senya_*`                                           |
| Fase 1 (Apostes)  | `apostar_envit`¹ , `apostar_truc`², `fora_truc`, `passar`, jugada* |

> ¹ Només si `envit_level == 0` i `round_counter == 0` (primera ronda, sense envit previ)
> ² Només si el jugador actual no és el propietari de l'última aposta de Truc

###### `_reset_hand_state()` -> Reset per Nova Mà

Quan acaba una mà (per guanyador o per "Fora"):

1. Avança `ma` al següent jugador
2. Barreja i reparteix de nou
3. Reseteja totes les variables de Truc, Envit, rondes i historials

###### `is_ma_over()` i `is_over()` -> Fi de Mà / Partida

Distingeix els dos nivells d'acabament:

- `is_ma_over()`: retorna `True` si la **mà actual** ha acabat (hi ha guanyador o s'han esgotat les rondes).
- `is_over()`: retorna `True` si la **partida** ha acabat (`max(score) >= puntuacio_final`).

---

##### Fases del torn i Joc

En les darreres versions de l'entorn les fases es redueixen a 0 (Senyals) i 1 (Joc i Apostes). Això simplifica l'arbre de decisions permetent cridar `play_card_X` directament en fase 1 sent equivalent a la fase 2 original.

```text
  Fase 0: Senyals
      |  passar / senya_*
      v
  Fase 1: Apostes i Joc
      |  passar / apostar_* / play_card_X
      v
  Nova Ronda (torna a Fase 0 si senyes=True, sinó Fase 1)
```

**Nota:** Si `senyes=False`, la fase 0 se salta completament i es comença directament a la fase 1.

#### Entorn de Simulació

`TrucEnv` és l'entorn que adapta la lògica del joc (`TrucGame`) a la interfície estàndard de **RLCard**. La seva funció principal és **transformar l'estat del joc** (diccionaris llegibles per humans) en **vectors numèrics** (observacions) que un agent de Reinforcement Learning pot processar.

##### Arquitectura

```text
                    observació, acció
  +------------+  <------------------>  +------------+
  |  Agent RL  |                        |  TrucEnv   |
  +------------+                        |  (env.py)  |
                                        +-----+------+
                                              | estat, step
                                              v
                                        +------------+
                                        |  TrucGame  |
                                        |  (game.py) |
                                        +--+--+--+---+
                                           |  |  |
                              +------------+  |  +------------+
                              v               v               v
                        +----------+   +-----------+   +-----------+
                        |TrucDealer|   |TrucJudger |   |TrucPlayer |
                        +----------+   +-----------+   +-----------+
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

###### 2. Vector de Context (`obs_context` tensor (17,))

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
        'obs_context': np.array([...])  # Vector de context (17) de variables contínues 
    },
    'legal_actions': OrderedDict,     # Accions legals com a OrderedDict
    'raw_obs': state,                 # Estat original (diccionari llegible)
    'raw_legal_actions': ['passar', 'apostar_truc', ...],  # Noms de les accions
    'action_record': self.action_recorder  # Historial d'accions (d'RLCard)
}
```

---

##### Altres Mètodes

Les recompenses son el fruit d'una partida (Victòria/Derrota) i de fites intermèdies:

1. **Reward final (get_payoffs)**: senyal de victòria/derrota normalitzada:
   `R = sign(delta) * sqrt(|delta| / objectiu_punts)`
   
2. **Rewards intermedis (step)**: senyals parcials emesos a cada fi de ronda (±0.07 × pes_ronda), resolució d'envit (±0.05 × pts/8) i resolució de truc (±0.05 × pts/24).

###### `_decode_action(action_id)` -> Descodificar Acció

Converteix un índex d'acció numèric al seu nom string:

```python
ACTION_LIST[action_id]  # e.g. 0 -> 'play_card_0', 4 -> 'apostar_truc'
```

###### `_get_legal_actions()` -> Accions Legals

Aquest mètode de l'entorn delega directament a `TrucGame.get_legal_actions()`, que és on realment es calcula la llista d'accions permeses. Retorna una **llista d'índexs** (enters) que representen les accions vàlides.

## Entrenament i Models d'Aprenentatge per Reforç

L'entrenament dels agents per jugar al Truc es basa en arquitectures de Reinforcement Learning adaptades a l'entorn de simulació del joc d'informació imperfecta.

### Models de decisió (RLCard)

S'utilitzen dos grans algorismes proporcionats i adaptats des d'RLCard:

1. **Deep Q-Network (DQN)**:

   - Utilitza una xarxa neuronal ($qnet$) per estimar la funció de valor $Q(s, a)$. Sovint s'emprèn una xarxa MLP profunda, en el nostre cas de `[256, 256]` neurones ocultes cap amunt.
   - L'entrenament segueix un mètode de *Self-Play* contra una versió congelada d'ell mateix. Durant l'entrenament, l'agent reajusta els pesos mitigant l'error quadràtic entre les prediccions i les recompenses emmagatzemades al *Replay Buffer*.
   - Quan l'agent principal assoleix un alt rendiment validat en fase d'avaluació, els seus nous pesos es transfereixen a l'oponent "congelat", obligant-lo a superar-se constantment pas a pas en aquesta guerra armamentística per lluitar contra l'estancament.
2. **Neural Fictitious Self-Play (NFSP)**:

   - Algorisme més complex i avançat dissenyat explícitament per cercar i assolir Equilibris de Nash en jocs competitius d'informació imperfecta.
   - Manté i entrena dues xarxes separades paral·leles:
     - **Xarxa RL (Q-Network)**: Es focalitza absolutament a trobar i potenciar l'estratègia temporal més *explotadora* contra el rival concret actual.

### La Xarxa Unificada (El "Cos")

Atès que l'observació de l'entorn del joc vectoritza mapes d'informació temporal de cartes (tensors multicapa espacials) i variables contextuals escalars (puntuacions, apostes actives de truc), **les capes de decisió dels models de RLCard incorporen a sota un extractor de característiques integrat referit com "el Cos"**.

Implementat a `xarxa_unificada.py`, està estructurat internament per PyTorch en:

- **Branca A (Cartes)**: Xarxa Conv2D (CNN) per extreure complexitats tàctiques del tensor principal de mida `(6, 4, 9)`.
- **Branca B (Context)**: Perceptró Multicapa (MLP) lineal al vector de configuració `(17,)`.

Les dues branques conflueixen de forma encadenada cap a un únic espai latent densament connectat que es transfereix cap als algoritmes de dalt (DQN o NFSP).

#### Preentrenament Supervisat del Cos

Abans d'iniciar l'aprenentatge per reforç, s'ha implementat una fase de preentrenament supervisat mitjançant l'script `preentrenar_cos.py`. L'objectiu d'aquesta fase és dotar a la xarxa d'extracció de característiques d'una comprensió fonamental de la semàntica del joc.

- **Dataset**: S'ha generat un dataset sintètic de **200.000 mostres** utilitzant l'entorn de joc real (`TrucEnv`). Les dades es divideixen en un **80% per a entrenament** i un **20% per a validació**.
- **Regeneració de dades**: Per evitar el *sobreajustament* a un dataset estàtic, s'ha implementat un mecanisme de **regeneració iterativa** que substitueix el dataset complet cada 20 èpoques amb dades noves generades de forma pseudo-aleatòria.
- **Multitask Learning**: El model s'entrena per predir simultàniament tres objectius crítics:
  1. **Punts d'Envit (MSE)**: Predicció de la puntuació d'envit de la mà inicial (normalitzat per 24).
  2. **Força de Truc (MSE)**: Estimació de la potència de les cartes per a la fase de Truc (normalitzat per 330).
  3. **Accions Legals (BCE with Logits)**: Identificació de quines de les 19 accions possibles són vàlides en un estat donat.
- **Hiperparàmetres i Regularització**:
  - Optimitzador: **Adam** amb un Learning Rate de **0.001**.
  - Regularització: **Weight Decay (L2)** de **1e-4** i capes de **Dropout**.
  - **Early Stopping**: Aturada de l'entrenament si la pèrdua de validació no millora durant 10 èpoques consecutives.

#### Metodologia d'Entrenament per Reforç (DQN i NFSP)

L'entrenament principal es gestiona des de `entrenaments_unificats.py`, que permet alternar entre diferents algorismes i modes d'integració dels pesos preentrenats.

- **Modes d'Integració**:

  - **Scratch**: Pesos inicialitzats aleatòriament.
  - **Frozen**: El "Cos" es manté bloquejat amb els pesos preentrenats, i només s'entrena el "Cap" (MLP).
  - **Fine-tune**: S'entrena tota la xarxa, però amb un Learning Rate reduït per al "Cos" (**1e-5**) per preservar el coneixement previ.
- **DQN (Deep Q-Network)**:

  - **Opponent Pool**: Durant el *Self-Play*, l'agent no només juga contra ell mateix, sinó contra un ventall d'oponents per millorar la robustesa:
    - **20% Random**: Contra un agent aleatori per exploració bàsica.
    - **40% Polyak**: Contra una versió suavitzada (soft update, $\tau = 0.05$) dels pesos actuals.
    - **40% Historical**: Contra versions passades desades en un pool de models.
  - **Reward Scheduling**: Aplicació d'una evolució de la variable **Beta** (de **0.5 a 1.0**) que modula les recompenses per prioritzar primer la supervivència i després la victòria.
  - **Learning Rate Decay**: En els punts 25%, 50% i 75% del total, la taxa base d'aprenentatge es redueix a la meitat (factor 0.5) per estabilitzar la convergència.
- **NFSP (Neural Fictitious Self-Play)**:

  - Implementa una arquitectura de **duplicitat de xarxes** per assolir Equilibris de Nash:
    - **Xarxa RL (DQN)**: Per aprendre la millor resposta (Best Response) contra el rival actual.
    - **Xarxa SL (Supervised Learning)**: Aprenentatge per generar una "Política Mitjana" (*Average Policy*) estable i menys fràgil.
  - Paràmetre **Eta (0.3)**: Determina la probabilitat de seleccionar l'acció de la xarxa RL versus la xarxa SL.

#### Anàlisi de Resultats i el Notebook (Conclusions)

Segons les dades del registre de `Comparativa_Experiments.ipynb`, s'han extret conclusions determinants:

1. **Eficiència del Preentrenament**: Els models en mode **Frozen** o **Fine-tune** assoleixen victòries molt abans que el mode **Scratch**, confirmant que la transferència de coneixement és efectiva.
2. **DQN vs NFSP**: En l'entorn del Truc, el **DQN supera el NFSP**. L'agressivitat del DQN en optimitzar els $Q$-values és més eficaç en partides curtes que l'estratègia mitjana conservadora del NFSP.
3. **Optimització del Fine-tune**: La combinació d'un Learning Rate molt baix per a l'extractor ($1e-5$) i un ritme normal per al cap de decisió ($5e-4$) permet assolir els millors resultats.

---

- **Per què el NFSP perd avantatge?**: El NFSP busca un Equilibri de Nash mitjançant una "Política Mitjana" (Average Policy) molt conservadora. En el Truc, aquesta cautela "segura" es tradueix en una manca d'agressivitat que permet al DQN dominar les mans clau a base de risc calculat. L'estabilitat del NFSP és ideal per a jocs llargs i complexos como el Poker, però peca de passivitat en el dinamisme del Truc.

### Contingut del Directori de Treball `RL/`

A l'arrel de solucions trobem el desglossament del directori específic destinat a l'estratègia dels models:

| Carpeta          | Descripció                                                                                                                                                                    |
| :--------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entrenament/` | Scripts iteratius i codi font utilitzat per realitzar les formacions (ex. el nucli de `entrenaments_unificats`).                                                             |
| `models/`      | Definicions PyTorch de les xarxes d'extracció i pesos globals (`xarxa_unificada.py`, classes d'agents de sortida i dades generades serialitzades model final al directori). |
| `notebooks/`   | Documents clau estadístics de Jupyter (`ipynb`/`html`) per realitzar els anàlisis i estudis comparatius a fons.                                                          |
| `tools/`       | Utilitats de reajustament (scripts `test_comparativa.py` per lluitar contra pesos ja tancats o eines auxiliars `exportar_pesos.py`).                                       |
