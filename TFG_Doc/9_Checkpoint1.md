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

Dos experiments paral·lels (12M+12M steps curriculum vs 24M steps control directe). Detall complet a [7_Fase2_MarcTeoric.md](7_Fase2_MarcTeoric.md), [8_Fase2_Implementacio.md](8_Fase2_Implementacio.md) i `TFG_Doc/notebooks/2_curriculum_learning/comparacio_fase2.ipynb`.

### Resultats reals (run 11–13 abr 2026)

**Taula A — Comparació a 12M steps (mateix pressupost, avaluació sobre partides):**

| Agent | Control 12M | Curriculum 12M mans | Δ |
|:--|--:|--:|--:|
| DQN-RLCard | 18.2% | 26.0% | +7.8 pp |
| NFSP-RLCard | 25.2% | 27.2% | +2.0 pp |
| DQN-SB3 | 44.0% | **86.2%** | **+42.2 pp** |
| PPO-SB3 | 30.8% | **87.2%** | **+56.5 pp** |

**Taula B — Comparació a 24M steps totals (control vs curriculum mans+finetune):**

| Agent | Control 24M | Curriculum complet | Δ |
|:--|--:|--:|--:|
| DQN-RLCard | 52.0% | 22.2% | −29.8 pp ❌ |
| NFSP-RLCard | 55.5% | 60.0% | +4.5 pp |
| DQN-SB3 | 60.5% | 56.0% | −4.5 pp ≈ |
| **PPO-SB3** | 35.0% | **75.0%** | **+40.0 pp** ✓✓ |

### Conclusions

**El curriculum millora el rendiment final condicionat a l'agent i al pressupost.** A 12M steps, tots 4 agents milloren entrenant sobre mans. A 24M steps (pressupost total), només 2 de 4 en surten beneficiats. El coll d'ampolla és la transferència, no l'aprenentatge inicial.

**Els agents que es beneficien més són els on-policy (PPO).** L'expectativa inicial era la contrària — es temia oblit catastròfic de PPO — però el resultat és l'invers: PPO és precisament qui més necessita reward dens per arrencar. Amb episodis curts i reward dens, PPO arriba al 87.2% sobre mans i conserva el 75% al finetune sobre partides: **el millor resultat del TFG fins a la data**.

**Sorpresa del control DQN-SB3:** entrenar 24M steps directament sobre partides (60.5%) supera el curriculum (56.0%). Per a DQN-SB3, el curriculum només aporta si el que es vol maximitzar és el sostre a mans (86%), no el resultat final a partides.

## Decisió: models que continuen a la Fase 3

Els dos models seleccionats per a les fases següents són **DQN-SB3** i **PPO-SB3**.

### 1. PPO-SB3 — *el gran beneficiari i nou baseline*

- **Millor resultat del TFG**: 75% contra `AgentRegles` amb curriculum complet (+40 pp sobre el seu propi control).
- **Millor aprenent a mans**: 87.2%, empatat amb DQN-SB3 però amb la meitat del cost computacional (~40 min vs 5.5h per run).
- La narrativa experimental és clara: PPO no tenia problema de capacitat, sinó de senyal d'aprenentatge. El curriculum ho resol.
- Throughput excel·lent permet iterar ràpidament a Fase 3.

### 2. DQN-SB3 — *el baseline robust*

- Millor rendiment a partides sense curriculum (60.5% control a 24M).
- Sostre de 86.2% a mans: el màxim assolit en entrenament sobre mans aïllades.
- Representa la família *value-based off-policy* i és el contrapunt natural a PPO.
- Throughput acceptable (~5.5h per run) per a experiments llargs com Fase 3.

### Models descartats

- **DQN-RLCard**: oblit catastròfic sever al finetune (col·lapse a 22%), resultats consistentment inferiors a DQN-SB3 en tots els experiments. Descartada la seva integració amb càrrega de pesos externs fins que s'estabilitzi el pipeline.
- **NFSP-RLCard**: l'algorisme més lent (~17h per run), `metric` que no supera el 60%, i cap evidència d'escalabilitat amb més còmput (confirmat ja a Fase 1). Benefici massa modest per al cost.
