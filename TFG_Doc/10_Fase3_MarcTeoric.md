# 10. Fase 3: Marc Teòric (Valor del Feature Extractor COS)

Aquest document introdueix la pregunta experimental de la Fase 3 i els fonaments teòrics que la justifiquen. La Fase 3 és la primera de dues fases dedicades a estudiar el *feature extractor* `CosMultiInput` en el context dels agents **DQN-SB3** i **PPO-SB3**.

> **Q1 (Fase 3):** Aporta valor el feature extractor COS respecte al MLP estàndard de SB3? O és simplement un artefacte de la configuració d'entrenament?

La resposta a la pregunta complementària —quina estratègia d'inicialització és millor: scratch, frozen o finetune?— correspon a una fase posterior.

## Motivació

Els resultats de la Fase 2 ([[8_Fase2_Implementacio]]) revelen un patró clar: els dos agents SB3 mostren capacitats molt asimètriques entre protocols:

| Model | Control MLP | Curriculum MLP | Mans MLP |
|:--|--:|--:|--:|
| DQN-SB3 | 60.5% | 56.0% | — |
| PPO-SB3 | 35.0% | 75.0% | — |

Tots els agents de Fase 2 usaven la `MlpPolicy` per defecte de SB3 amb `net_arch=[256,256]`. Internament, quan l'entorn retorna un espai d'observació `Dict`, SB3 aplica el `CombinedExtractor`: aplana cadascuna de les entrades del diccionari i les concatena en un vector pla que rep el MLP.

El problema d'aquest enfoc per al Truc és que l'observació `obs_cartes` és naturalment **2D**: `(6, 4, 9)` on les tres dimensions representen canals (mà pròpia, historial, senyes), pals (4) i rangs (9). Un MLP pla ha de *redescobrir* per ell mateix que el "4 de copas" i el "4 de oros" comparteixen rang, o que el "7 de bastos" i el "1 de espades" estan relacionats per la jerarquia de forces del Truc. Una convolució sobre la grid pal×rang captura aquest coneixement estructural de manera directa.

La hipòtesi de la Fase 3 és:

> *Canviar l'arquitectura del feature extractor per un disseny que respecta l'estructura 2D de les cartes (COS) ha de millorar la qualitat final de la política, independentment de si els seus pesos s'inicialitzen aleatòriament o amb preentrenament supervisat.*

## L'arquitectura `CosMultiInput` i el seu adaptador SB3

### Observació i partició

L'observació de l'entorn és un vector pla de **240 dimensions** (veure `RL/tools/obs_utils.py`):

- Dimensions `[0:216]` → mapa de cartes `(6, 4, 9)` en format one-hot aplanat.
- Dimensions `[216:240]` → vector de context (24 dims): puntuació, fase del torn, rondes guanyades, nivell truc/envit, etc.

L'adaptador `CosMultiInputSB3` (a `RL/models/sb3/sb3_features_extractor.py`) rep el vector de 240 dims i el desempaqueta internament:

```python
cartes  = observations[:, :216].view(-1, 6, 4, 9)
context = observations[:, 216:]
return self.cos(cartes, context)
```

### Arquitectura `CosMultiInput` (dues branques)

**Branca A — CNN sobre el mapa de cartes** (`(B, 6, 4, 9)`):
- `Conv2d(6 → 16, kernel=(1,3))` + ReLU — patrons locals dins d'un pal.
- `Conv2d(16 → 32, kernel=(3,3))` + ReLU — relacions creuades entre pals i rangs.
- `Flatten()` → **320 dimensions**.

**Branca B — Densa sobre el context** (`(B, 24)`):
- `Linear(24 → 32)` + ReLU → **32 dimensions**.

**Fusió**:
- `concat` → 352 dims → `Linear(352 → 256)` + ReLU → **256 dimensions**.

La sortida de 256 dims coincideix exactament amb l'amplada de la primera capa oculta (`net_arch=[256,256]`), de manera que l'extractor substitueix el `CombinedExtractor` per defecte sense cap canvi a la resta de l'arquitectura.

### MLP estàndard SB3 (referència)

Sense COS, SB3 usa el `CombinedExtractor` integrat: aplana i concatena totes les entrades del diccionari d'observació. El resultat és un vector de 240 dims que entra directament al MLP `[256, 256]`. No hi ha cap inducció estructural sobre la forma de les cartes.

## Disseny experimental

La Fase 3 compara **COS vs MLP** en els tres protocols definits a la Fase 2, per als dos agents seleccionats al [[9_Checkpoint1]].

| Protocol | Entorn principal | Steps | Descripció |
|:--|:--|--:|:--|
| **Control** | `TrucGymEnv` (partides) | 24M | Entrenament directe sobre partides senceres. |
| **Curriculum** | `TrucGymEnvMa` → `TrucGymEnv` | 12M+12M | Fase 1: mans. Fase 2: finetune sobre partides. |
| **Mans** | `TrucGymEnvMa` (mans) | 24M | Entrenament exclusiu sobre mans individuals. |

Per a cada combinació (2 agents × 3 protocols), s'executa la variant **COS** i la variant **MLP**, cosa que dóna en principi 12 sèries. Però les quatre sèries MLP de control i curriculum **es reutilitzen de la Fase 2** per evitar reentrenar innecessàriament:

| Sèrie | Font |
|:--|:--|
| DQN-SB3 control MLP | Fase 2 (control) |
| PPO-SB3 control MLP | Fase 2 (control) |
| DQN-SB3 curriculum MLP | Fase 2 (curriculum) |
| PPO-SB3 curriculum MLP | Fase 2 (curriculum) |

Això deixa **8 runs nous**: control COS × 2 agents, curriculum COS × 2 agents, mans COS × 2 agents i mans MLP × 2 agents.

### Configuració d'oponents i avaluació

Igual que a Fase 2:

- **Oponent**: 10% Random + 90% AgentRegles (sense self-play).
- **Avaluació**: 100 partides contra agent aleatori + 100 contra agent de regles cada 500 000 steps, **sempre sobre partides senceres** (`TrucGymEnv`).
- **Mètrica composta**: `eval_metric = 0.25 × WR_random + 0.75 × WR_regles`.
- **Corbes curriculum**: els steps es mostren acumulats (0–24M); l'offset entre etapes s'aplica a la visualització.

### Hiperparàmetres principals

**PPO-SB3**: `NUM_ENVS = 32` entorns paral·lels (`SubprocVecEnv`), `n_steps = 256`, `batch_size = 2048`. Mateixos valors que Fase 2.

**DQN-SB3**: entorn únic, `learning_starts = 50 000`, `buffer_size = 200 000`, `exploration_fraction = 0.3`. Mateixos valors que Fase 2.

L'única cosa que canvia entre la variant COS i la variant MLP és el `features_extractor_class` al `policy_kwargs`.

## Expectatives teòriques

**Control i curriculum**: S'espera que la variant COS superi la MLP en tots dos agents, ja que l'estructura 2D del mapa de cartes és immediatament explotable per la CNN. L'efecte ha de ser especialment visible per a DQN, que aprèn sobre episodis llargs on la representació rica de l'estat és crítica.

**Mans**: El protocol de mans dona un senyal de recompensa molt més dens (fi d'episodi cada ~15 steps). Amb tant de senyal disponible, un MLP pla pot compensar la manca d'inducció estructural. La diferència COS vs MLP hauria de ser menor que als altres dos protocols.

**Protocol curriculum COS**: A l'etapa 2, quan el model passa de mans a partides, el fet que el cos (COS) hagi après una representació rica de les cartes podria reduir el *catastrophic forgetting*, perquè la política no ha d'adaptar-se alhora a un nou entorn *i* a una nova representació. Això hauria de traduir-se en un finetune més estable que la variant MLP.

## Riscos específics

- **COS scratch no convergeix al control**: l'arquitectura CNN té més paràmetres que el `CombinedExtractor` pla. Amb el protocol de control (partides llargues, reward sparse), pot ser que el COS scratch tardi més a convergir i no arribi al nivell del MLP dins del pressupost de 24M. Mesura: comparar les corbes en els primers 5M i en els darrers 5M.

- **COS perjudicial al curriculum** (per PPO): el gradient on-policy del finetune de l'etapa 2 pot afectar la representació del COS i produir un col·lapse si la CNN s'adapta massa ràpid a una distribució canviant d'estats. Mesura: monitoritzar la `wr_regles` a l'inici de l'etapa 2 (steps 12–14M).

- **Diferència inapreciable a mans**: si el protocol de mans ja és suficientment dens per a qualsevol arquitectura, el COS no aporta avantatge mesurable. Conclusió operativa: si `|metric_cos - metric_mlp| < 2%` als 24M, el valor del COS és arquitecturalment neutre per a mans.

## Per què COS en mode scratch

La Fase 3 usa el COS amb **pesos aleatoris** per a totes les variants noves. Aquesta elecció és deliberada: l'efecte que mesurem és el de **l'arquitectura** (CNN vs MLP pla), no el del preentrenament supervisat. Qualsevol diferència observada entre COS i MLP s'atribueix exclusivament al disseny estructural del feature extractor.

## Enllaços creuats

- [[7_Fase2_MarcTeoric]] — definició dels protocols control/curriculum/mans i dels entorns.
- [[8_Fase2_Implementacio]] — resultats numèrics que motiven la pregunta Q1.
- [[9_Checkpoint1]] — selecció dels dos models i pla de les fases 3–5.
- [[11_Fase3_Implementacio]] — detalls tècnics i resultats numèrics de la Fase 3.
