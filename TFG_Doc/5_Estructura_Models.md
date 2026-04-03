# 5. Estructura de la Carpeta Models

Aquesta secció detalla l'estructura i organització de la subcarpeta `RL/models`, que encapsula tota l'arquitectura de xarxes neuronals, definició d'agents i adaptadors del sistema. S'ha dividit en tres subcarpetes principals per modularitzar el codi.

## `core/`

Conté les peces centrals de processament d'informació compartides per tot el repositori:

- **`feature_extractor.py`**: Defineix la xarxa extractora de característiques (`CosMultiInput`). Aquesta xarxa té una branca CNN per processar la matriu de cartes 2D i una branca de perceptró multicapa (MLP) paral·lela per processar la informació d'estat (context) escalar. Després, ambdues branques es fusionen en un espai latent de dimensió 256. També s'hi inclou `ModelPreEntrenament`, utilitzat només durant la fase de pre-entrenament supervisat per generar a la xarxa els comportaments latents (mitjançant la simulació predictiva d'objectius com la quantitat de punts d'envit d'una mà, la probabilitat legalística de qualsevol de les 19 accions i la sortida del càlcul de potència d'emparellament entre valors individuals).
- **`obs_adapter.py`**: Aportador d'utilitats relacionades amb l'adaptació i el mapeig de l'observació o de llistes que han estat pre-vectoritzades pel funcionament adequat com tensors preparats pel "Cos".
- **`loader.py`**: Utilitats orientades a la càrrega eficient dels diferents pesos interns del model complet i fraccionat a partir de fitxers pre-entrenats `.pt`.

## `rlcard_legacy/`

Conté el codi de compatibilitat desenvolupada o copiada originalment dels agents d'intel·ligència de `RLCard` (per suportar DQN i NFSP propulsats directament pel nucli establert sense modificacions d'altres àrees obertes on-policy):

- **`model_adapter.py`**: Inclou un tipus d'embolcall que processa o aplana algunes operacions a nivell local, simulant adaptabilitat als paràmetres originals d'experiència esperada dels components no compatibles nativament per l'entrada multi-tensors (requerida al Deep-Q o les parts paral·leles del best response supervisor a l'NFSP).

## `model_propi/`

Conté tota la implementació i organització d'algorítmica feta originalment i directament pel programador i feta de forma particularitzada pel projecte Truc.

- **`agent_regles.py`**: Defineix l'agent oponent de rendiment moderat-alt (Rule-Based) i altament determinista gràcies a l'exposició als jocs coneguts per heurístiques de referència d'escala real base (a partir de la codificació per regles humanes i d'inferència abstractes a curt i llarg termini), el qual s'utilitza majorment per avaluar a l'estudiant iterativament o per fer que aquest hi jugui entrenaments d'agressivitat directes i reals dins l'espai d'auto-joc paral·lel de Fase PPO.
- **`model_ppo/`**:
  - Les implementacions directes referents al _Proximal Policy Optimization_ (PPO) i components derivats de la política on-policy per defecte escollida.
  - El codi està organitzat segons les evolucions respectives tractades: `ppo/` (versió de Perceptró simple/MLP), `ppo_gru/` (versió basada fortament en cèl·lules recurrents per fer record local sobre partides en memòria activa respecte iteracions temporals anteriors als mateixos jocs), i `ppo_gru_nash/`.
  - **`ppo_loss.py`**: Formules específicament per fer estimacions o ajustos directes de talls (clipings PPO), incloent el desenvolupament matemàtic asíncron de la funció d'Avantatges per Estimació Generalitzada (GAE: _Generalized Advantage Estimation_).
  - **`ppo_utils.py`**: Funcions o ajudes generals matemàtiques que donen suport al càlcul o operacions.
  - **`ppo_loaders.py`**: Moduls orientats exclusivament en l'exportació o arrencada i re-creació correcte del complex de tensors associats per inferència durant la fase interactiva del Truc Engine (l'execució i jocs UI en temps real de Flask-SockerIO HTTP local i/o producció).
