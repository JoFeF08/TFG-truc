# 1. Arquitectura MVC i Controlador

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

##### Implementació Actual: `ModelInteractiu`

La classe `ModelInteractiu` (`controlador/model_interactiu.py`) és l'adaptador principal que implementa el contracte `Model`. Va més enllà de simplement traduir mètodes; concretament s'encarrega de:

- **Abstreure el TFG**: Manté la instància oculta i purgada de `TrucGame` actuant de tallafocs.
- **Injecció de Models d'Aprenentatge**: Mitjançant un directiu de configuració (ex: `{"tipus": "ppo"}`), extreu arquitectures preentrenades gràcies a l'encaminador `crear_model(...)` de `RL.models`, instanciant i fixant directament la lògica per cada jugador bot.
- **Resolució Humana (`_SlotHuma`)**: Assigna slots silenciosos abstractes per aquells jugadors que decideixen via interfície UI. En cas freqüentar l'execució d'un d'aquests, frena la cadena delegant l'input manual de l'usuari pel Controlador. En contrast, les crides autònomes (`get_accio_bot`) disparen inferències vectorials respectant l'aïllament del domini.

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

##### Grafics i Vistes Actuals Disponibles

El repositori compta actualment amb dues interfícies oficials independents i plenament madures ubicades al subdirectori `joc/vista/`:

- **Vista per Consola (`vista_consola.py`)**: Interfície d'entrada/sortida de text (CLI). És una visualització tècnica, molt lleugera, àgil i directa. Renderitza pas a pas per terminal l'estat dels tensors ocults, historial de torns de forma declarativa i desplega el menú interactiu per números simples. És extraordinàriament recomanable per validar execucions estructurals i auditar comportaments directes en entorns *headless* (servidors SSH capitius).
- **Vista d'Escriptori (`vista_desktop/vista_desktop.py`)**: Experiència d'usuari (UI) gràfica dissenyada per a l'ús i entreteniment o avaluació de prestacions d'un rival sintètic RL de forma estètica. Transformarà estats tècnics de programari directament a l'àmbit visual i semàntic gestionant renderitzacions sobre gràfics reals de naips, botons (recursos carregats de l'arbre intern de `/img_iu`) capturant els events d'acció directament de clics de ratolí. Per donar emulació humana, les interfícies s'encarreguen d'intercalar *delays* artificials davant qualsevol moviment que el Controlador recuperi sota l'etiqueta automàtica `es_bot=True`. Aquesta interfície ha estat feta de manera independent amb IA.
