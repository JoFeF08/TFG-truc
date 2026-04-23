# 12. Fase 3.5: Marc Teòric (Estratègia d'Inicialització del COS)

Aquest document introdueix els fonaments teòrics de la Fase 3.5. Partint dels resultats de la Fase 3 ([[11_Fase3_Implementacio]]), la Fase 3.5 vol respondre:

> **Q2:** Dona valor el preentrenament supervisat del COS, i en cas afirmatiu, quin règim d'inicialització —frozen o finetune— és preferible respecte a scratch?

## Motivació

La Fase 3 va establir que el **protocol mans** és el que ofereix els millors resultats en tots els escenaris (>82%) i va fixar la comparació COS scratch vs MLP estàndard. Per al protocol mans, la diferència entre COS scratch i MLP és marginal (≤4 pp per a ambdós agents). Això no significa que el COS no tingui potencial, sinó que usant-lo amb **pesos aleatoris** (scratch) l'arquitectura CNN no arrenca amb cap avantatge representacional: ha d'aprendre la representació de les cartes des de zero, en competència directa amb l'aprenentatge de la política.

La Fase 3.5 introdueix una variable que la Fase 3 deliberadament va excloure: el **preentrenament supervisat del cos**. Abans de começar el RL, els pesos de `CosMultiInput` s'inicialitzen via aprenentatge supervisat sobre un dataset d'estats de Truc. La hipòtesi és:

> *Un cos preentrenat que ja distingeix rang, pal i força de les cartes des del primer step de RL permet a la política centrar-se immediatament en la presa de decisions, sense haver de reaprendre la representació. Això hauria de traduir-se en un sostre més alt i/o una convergència més ràpida.*

## El preentrenament supervisat del cos

### Per què no s'aplica a Fase 3

La Fase 3 comparava **arquitectura COS vs arquitectura MLP**. Per aïllar l'efecte arquitectural, era imprescindible que el COS s'inicialitzés amb pesos aleatoris, igual que el MLP. Introduir el preentrenament a Fase 3 hauria confós dos efectes independents: *arquitectura* i *inicialització*. Fase 3.5 treballa exclusivament sobre l'arquitectura COS (ja validada com a mínimament competitiva) i estudia l'efecte de la inicialització.

### Objectiu del preentrenament

El cos es preentrena amb `RL/entrenament/entrenamentEstatTruc/preentrenar_cos.py`. L'objectiu és que el cos aprengui, de manera supervisada, a extreure les quantitats rellevants del joc **a partir d'observacions reals de Truc**:

- **Punts d'envit** de la mà del jugador en torn.
- **Accions legals** disponibles en l'estat actual (vector binari de 19 posicions).
- **Força de cada carta** de la mà (ordenades, normalitzades).

Aquests tres targets cobreixen el coneixement tàctic bàsic necessari per jugar una mà: saber quines cartes pots jugar, quant val el teu envit i com de fortes són les teves cartes relativament. Si el cos ja extreu bé aquests conceptes, la política RL té una base representacional molt més rica des del principi.

### Arquitectura del model de preentrenament

S'instancia un `ModelPreEntrenament` compost per `CosMultiInput` + tres caps lineals:

- `cap_envit: Linear(256, 1)` — entrenada amb **MSE**.
- `cap_accions_legals: Linear(256, 19)` — entrenada amb **BCE-logits** (multi-label).
- `cap_forces: Linear(256, 3)` — entrenada amb **MSE**.

**Loss combinada**: `1.0·loss_envit + 2.0·loss_accions + 2.5·loss_forces`. Els pesos reflecteixen la importància de cada target: les forces de les cartes és el senyal més ric (3 valors continus), les accions legals el més immediatament útil per la política, i l'envit el més simple.

### Generació del dataset

Es juguen episodis amb una política pseudo-aleatòria sobre `TrucEnv` real (80% acció qualsevol aleatòria, 20% accions no-carta per forçar variació en apostes i envits). A cada estat visitat s'extreuen `obs_cartes (6,4,9)`, `obs_context (24,)` i els tres targets. El dataset complet té **200 000 mostres**.

### Procediment d'entrenament supervisat

- **Split 80/20** train/val, `ReduceLROnPlateau`, `patience=10` epochs sobre la val loss.
- **Regeneració del dataset** cada 20 epochs per evitar sobreajust a una distribució fixa.
- **Sortida**: només es guarden els pesos del cos (`model.cos.state_dict()`), no les caps (que actuen com a "bastides" i es descarten). L'arxiu resultant és `best_pesos_cos_truc.pth`.

Les caps de preentrenament mai s'usen durant el RL; únicament serveixen per donar gradient informatiu al cos durant la fase supervisada.

## Quatre règims de comparació

La Fase 3.5 compara quatre règims sobre el protocol **mans** (`TrucGymEnvMa`, 24M steps):

| Règim                    | Pesos inicials del cos     | `requires_grad(cos)` | Novetat respecte F3                      |
| :------------------------ | :------------------------- | :--------------------- | :--------------------------------------- |
| **Sense COS (MLP)** | — (CombinedExtractor pla) | —                     | Referència de Fase 3                    |
| **Scratch**         | Aleatoris (SB3)            | True                   | Línia base arquitectural (igual que F3) |
| **Frozen**          | Preentrenats supervisats   | **False**        | ✓ Nou                                   |
| **Finetune**        | Preentrenats supervisats   | True                   | ✓ Nou                                   |

La variant **sense COS** és la mateixa execució que la Fase 3 (`ppo_sb3_mans_mlp` / `dqn_sb3_mans_mlp`) i es reutilitza com a referència absoluta. La variant **scratch** és la mateixa execució que la Fase 3 (protocol mans amb COS aleatori) i serveix com a línia base arquitectural per aïllar l'efecte del preentrenament. Les variants **frozen** i **finetune** són les noves execucions d'aquesta fase.

### Discussió teòrica de cada règim

**Sense COS (MLP)**: el `CombinedExtractor` de SB3 aplana les 240 dims d'observació i les passa directament al MLP `[256, 256]`. No hi ha cap inducció estructural sobre la forma de les cartes. Estableix el sostre que un model "sense cap coneixement estructural" pot assolir.

**Scratch**: l'arquitectura CNN és la correcta, però els pesos són aleatoris. La CNN ha d'aprendre la representació de les cartes simultàniament amb la política RL, sense cap senyal supervisat. La comparació scratch vs sense COS mesura l'**efecte arquitectural pur**; la comparació scratch vs frozen/finetune mesura l'**efecte del preentrenament**.

**Frozen**: el cos ja sap extreure la representació (envit, accions legals, forces) però no es pot actualitzar durant el RL. Tota l'adaptació recau sobre les capes de política (`net_arch`). El risc és un **sostre rígid**: si el preentrenament no captura alguna característica clau per al RL, no hi ha manera de compensar-ho. El benefici és **estabilitat**: el cos no pot ser destruït pels gradients del RL.

**Finetune**: el cos s'inicialitza amb els pesos supervisats i es continua entrenant durant el RL. Teòricament és el millor dels dos móns, però introdueix el risc de **catastrophic forgetting**: el gradient on-policy (especialment el de PPO, que fa múltiples passos per rollout) pot desplaçar el cos lluny del punt preentrenat abans que la política hagi aprofitat la representació inicial.

Per a **DQN**, el risc de forgetting és moderat: el *replay buffer* i la *target network* amortiguen els gradients. Per a **PPO**, el risc és alt pel nombre d'`n_epochs` per actualització; cal monitoritzar la `wr_regles` a les primeres avaluacions per detectar col·lapsos precoços.

## Connexió amb fases anteriors i posteriors

La Fase 3.5 tanca el cicle d'estudi del COS en entorn de mans. Els resultats de la millor combinació (agent × règim) sobre el protocol mans alimenten les fases següents, on s'explorarà si el model pot transferir el coneixement de mans a partides senceres.

## Enllaços creuats

- [[10_Fase3_MarcTeoric]] — Q1: valor arquitectural del COS vs MLP.
- [[11_Fase3_Implementacio]] — resultats que motiven la Q2 i la reutilització de sèries.
- [[9_Checkpoint1]] — selecció dels dos models i pla de les fases 3–5.
- [[4_Estructura_Models]] — descripció completa de `CosMultiInput`.
- [[13_Fase35_Implementacio]] — detalls tècnics i resultats numèrics de la Fase 3.5.
