# Checkpoint 1 — Tancament de Fase 1 i Fase 2

Aquest document resumeix les conclusions de les dues primeres fases experimentals i justifica quins models continuen a partir de la Fase 3.

## Context

Quatre algorismes avaluats en totes dues fases: **DQN-RLCard**, **NFSP-RLCard**, **DQN-SB3**, **PPO-SB3**. La mètrica principal és `metric = 0.25·wr_random + 0.75·wr_regles`, amb èmfasi al win-rate contra `AgentRegles` (oponent heurístic fort).

## Fase 1 — Comparativa base (partides senceres)

Dos experiments: pressupost fix per steps (5M) i pressupost fix per temps (4h). Detall complet a [6_Fase1_Resultats.md](6_Fase1_Resultats.md).

**Resultats aproximats (sostre `metric`):**

| Agent       | 5M steps | 4h temps |
| :---------- | -------: | -------: |
| DQN-RLCard  |  ~23–35 |  ~21–34 |
| NFSP-RLCard |  ~30–46 |  ~30–35 |
| DQN-SB3     |  ~30–43 |  ~16–28 |
| PPO-SB3     |  ~18–26 |  ~21–35 |

**Conclusions:**

1. Cap algorisme supera de manera consistent el ~35% de `metric` contra `AgentRegles`.
2. PPO té mala *sample efficiency* en aquest entorn (18 accions, recompensa esparsa, horitzó llarg) però compensa amb throughput (~30× NFSP).
3. Donar més temps no canvia qualitativament els resultats → el coll d'ampolla és el **senyal d'aprenentatge**, no el còmput.
4. Cal canviar la formulació del problema → motivació directa de la Fase 2.

## Fase 2 — Curriculum learning (mà → partida)

Dos experiments paral·lels (12M+12M steps curriculum vs 24M steps control directe). Detall a [7_Fase2_MarcTeoric.md](7_Fase2_MarcTeoric.md) i [8_Fase2_Implementacio.md](8_Fase2_Implementacio.md).

**Resultats aproximats de `metric` a les darreres avaluacions:**

| Agent             |      Curric. mans |  Curric. partides |  Control partides |
| :---------------- | ----------------: | ----------------: | ----------------: |
| DQN-RLCard        |           ~17–18 |           ~15–20 |           ~15–18 |
| NFSP-RLCard       |           ~19–24 |           ~20–23 |           ~20–22 |
| **DQN-SB3** | **~68–74** | **~37–41** | **~49–54** |
| **PPO-SB3** | **~74–84** |             ~5–8 |           ~11–18 |

**Observacions clau:**

1. **Entrenar per mà (recompensa densa) és dràsticament més fàcil**: els dos agents SB3 arriben al 70–85% de `metric` contra regles, quelcom inassolible a la Fase 1.
2. **La transferència mà → partida no és gratuïta**:
   - DQN-SB3 transfereix raonablement (74 → 40): perd la meitat del rendiment però es manté com a millor model a partides.
   - PPO-SB3 **col·lapsa** (84 → 6): patró compatible amb *catastrophic forgetting* i amb el *mismatch* entre reward-shaping per mà i reward esparsa a nivell de partida.
3. **Sorpresa del control**: per a DQN-SB3, entrenar 24M steps directament a partides (control, ~54) supera el curriculum a partides (~40). Per a aquest agent, el curriculum **només val la pena si el que importa és el sostre a nivell de mà**, no la partida final.
4. **Els dos agents RLCard no es beneficien del curriculum**: es mantenen al voltant del sostre de la Fase 1 en totes dues configuracions. La limitació és estructural (xarxa + pipeline RLCard), no de formulació.

**Quan ajuda el curriculum learning, en resum:**

- **Per mà**: ajuda sempre als models capaços d'aprendre (SB3). La recompensa densa i l'horitzó curt fan trivial el que abans era impossible.
- **Per partida**: només ajuda si el model pot retenir i transferir el coneixement. DQN-SB3 hi perd, PPO-SB3 hi col·lapsa → cal arquitectura amb memòria o millor representació (motivació directa de Fase 3 i 4).

## Decisió: models que continuen a la Fase 3

Els dos models seleccionats són:

### 1. DQN-SB3 — *el baseline robust*

- Millor `metric` global a partides (fins a 54% en control, 41% en curriculum).
- Bona estabilitat, *throughput* raonable (~3h 30m per experiment).
- Transfereix decentment mà → partida, fet que el converteix en el candidat natural per mesurar **guanys nets** de les millores arquitectòniques de les fases 3–4.
- Representa la família *value-based off-policy*.

### 2. PPO-SB3 — *el cas interessant a millorar*

- **Millor aprenent a nivell de mà** (fins a 84% de `metric`), però amb el pitjor comportament a partides.
- El seu problema no és capacitat sinó **context**: no reté estructura temporal a llarg termini ni manté coherència entre mans d'una mateixa partida.
- És **precisament** el tipus de model que hauria de beneficiar-se més de:
  - **Fase 3**: un *feature extractor* preentrenat que li doni una representació rica i estable de l'estat del joc.
  - **Fase 4**: un mòdul recurrent (LSTM/GRU) que resolgui el problema de memòria entre mans.
- Representa la família *policy-gradient on-policy* i dona la **millor narrativa experimental** del TFG: un model amb sostre molt alt localment que s'arregla progressivament amb millores arquitectòniques.
- Throughput excel·lent (~37 min per experiment), permet iterar ràpid a les fases següents.

### Models descartats

- **DQN-RLCard**: mateixa família que DQN-SB3 amb resultats consistentment pitjors en tots els experiments. No aporta informació addicional.
- **NFSP-RLCard**: teòricament atractiu per jocs d'informació imperfecta, però a la pràctica és l'algorisme **més lent** (17h per experiment), amb una `metric` mitjana que no justifica el cost, i cap evidència que escali bé amb més còmput (Fase 1 experiment per temps ja ho va mostrar).
