# Entrenament dels models de Truc

## 1. DQN (Deep Q-Network)

DQN aprèn a estimar de forma directa quant "val" cada acció possible donat l'estat en què es troba partida. Ho fa amb una xarxa neuronal de diverses capes.

### Bucle d'entrenament (DQN)

El sistema d'entrenament del DQN està pensat perquè el model no s'estanqui per haver jugat massa estona contra un rival aleatori i dolent. En comptes d'això, juga contra ell mateix en un bucle tancat:

1. Es creen dos agents idèntics (mateixa xarxa i mateixos pesos inicials). L'agent principal és el que aprèn i altera els seus propis pesos partint de les recompenses.
2. Comença a jugar partides completes (episodis). Al finalitzar, les *trajectories* generades les interioritza **només** l'agent principal.
3. El segon agent fa d'oponent "congelat". Té l'exploració (`epsilon`) desactivada per jugar sempre amb la millor estratègia de què disposa i no altera pas els seus pesos.
4. Cada certs episodis (fase d'**avaluació**), l'agent principal juga una gran quantitat de partides purament aleatòries contra un rival de veritat que llença cartes a l'atzar. L'objectiu d'això no és aprendre'n res, els pesos no es modifiquen; només es fa servir per mesurar com d'alt és el seu *winrate* base i desar la mètrica.
5. El punt clau ve aquí: Si ha destrossat el seu anterior rècord de victòries netes a l'avaluació (ha millorat globalment respecte a com jugava hores enrere), **s'agafen els nous pesos del model principal i se li trasplanten a l'oponent congelat**.
6. Així l'agent principal, de sobte, passa a enfrontar-se a una versió de si mateix molt millor i més enginyosa, i l'entrenament reprèn exigint-li més intel·ligència. Això crea una guerra armamentística constant de model vs model històric.
7. Al final es desa un document `.pt` només de la millor iteració global que ha donat la fase d'avaluació.

---

## 2. NFSP (Neural Fictitious Self-Play)

L'algorisme NFSP és teòricament capaç de trobar equilibris de Nash en jocs amb informació imperfecta sense perdre's. Utilitza dues xarxes separades simultàniament:

* Una xarxa **RL** (entrenada amb Q-learning) que computa quina és l'acció amb major retorn de victòria (més explotadora/greedy) si coneix a la perfecció l'estratègia mitjana del contrari.
* Una xarxa **SL** (Supervised Learning) que registra i interioritza els moviments dels propis agents de la partida d'abans, consolidant l'"Average Policy".

### Bucle d'entrenament (NFSP)

La particularitat de NFSP és que tots dos jugadors avancen junts:

1. S'instancien els dos agents d'Aprenentatge Profund a la partida.
2. Comença el gruix d'episodis: una partida sencera rere l'altra. Al acabar, les observacions (trajectories) s'enfilen de retorn als "buffers" dels dos agents **alhora**. Tant P0 com P1 s'entrenen del feedback de les cartes guanyades en la partida.
3. Quan toca fer la fase d'**avaluació** per monitoritzar mètriques (freqüència marcada per codi), s'atura l'aprenentatge un moment i tots dos canvien la seva política de generació de cartes d'"exploratòria" a "determinista". L'Enviroment els enfronta per separat contra un jugador Aleatori durant unes sub-partides per extreure com de bé van realment contra el jugador "benchmark" mut i extreure el famós "Reward Mig".
4. Si la taxa d'encert p0 va en rècord respecte a estones anteriors d'entrenament, desen l'instantània de model en aquell precís checkpoint com a "Millor Model P0". Es guarda el de "Millor Model P1" en consonància per estar entrenant contra P0 i haver-lo forçat i viceversa. L'entrenament es represa normalment després de gravar el *log*.
5. A diferència d'ara fa una estona, la fi de NFSP té "El gran duel final" (`playoff`): Els agents Best_P0 i Best_P1 carreguen els models desats de quan eren al seu *prime* històric de performance i s'enfronten entre ells diverses milers de vegades canviant les posicions J0 i J1 fins a la mort. El desèmil-porcent guanyador general dels combats destrueix i devora a la perdedora i guanya el lloc permanent al disc de `nfsp_truc.pt` consolidat per a tota la producció futura o inferència del TFG de fer de bot.

---

## Paràmetres clau i variables rellevants

A l'inici dels scripts hi ha les constants principals que defineixen com funciona el bucle:

* **`NUM_EPISODES`**: Total de partides del joc de principi a final (`12` pedres per defecte) que faran saltar l'entrenament (Per defecte `200_000` als darrers runs complets, depèn del temps d'espera i CPU/GPU disponible que puguis suportar abans de ser fumat).
* **`EVALUATE_EVERY`**: Nombre de loops en sec i lliures de molèsties entre cada parada a gravar els csv històrics d'entrenament per mesurar l'estat del reward sense entrenar i fix. Molt alt vol dir poca pèrdua de velocitat però poques dades d'aprenentatge.
* **`EVALUATE_NUM`**: Si t'atares a avaluar, n partides per reduir la variància contra aleatori i fer mitjanes (Ex `500`).
* **`SAVE_EVERY`**: Desament incondicional pur del .pt del moment si rebentes la consola amb Control C almenys tens salvament cada X episodis de la darrera fita.
* **`LAYERS`** / **`HIDDEN_LAYERS`**: Mides de la xarxa neuronal que resol la partida, normalment dos capes amagades grosses tipus `[256, 256]`.
* **`BATCH_SIZE`** i **`MEMORY_SIZE`**: `256` de batch normal d'entrenament xarxa, guardant la broma de buffers alts fins a les `100_000` per NFSP o DQN i no aturar-se en l'overfitting continuu de memòries recents i estancar el valor a la curta.
* **`LR_DECAY_AT`**: S'usa "Learning Rate Scheduling", el model comença en sec i fort a moure els pesos per trobar el sol abric al pantà enorme de la pèrdua teòrica de la partida, però de cop, el sistema d'autocontrol de l'entrenador frena els Learning rate per meitat i el dexa en pas de formiga si passa de `25%`, `50%` i `75%` del tram de num episodes planificat prèviament o previst i fer un asintota local suau a l'aprenentatge i finalitzar sense oscil·lar.
