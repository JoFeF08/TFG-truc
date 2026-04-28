# 14. Fase 4: Marc Teòric (Memòria d'Oponent amb LSTM)

Aquest document introdueix els fonaments teòrics de la Fase 4. Partint del millor model de la Fase 3.5 ([[13_Fase35_Implementacio]]), la Fase 4 vol respondre:

> **Q3:** Pot un agent amb memòria (LSTM) adaptar-se al comportament d'un oponent desconegut al llarg d'una sessió de partides consecutives, i aquest avantatge es tradueix en millor rendiment quan s'enfronta a oponents amb estils variats?

## Motivació

Totes les fases anteriors (2, 3, 3.5) han entrenat agents sense estat intern (MLP pur) contra un oponent únic i fix: 10% Random + 90% `AgentRegles` amb paràmetres hardcodats. Els millors resultats són **DQN frozen (91.2%)** i **PPO frozen (89.5%)** al protocol mans.

Quan desplegues aquests agents a `demo.py` contra un humà, cada partida comença en blanc: l'agent no pot aprofitar res del que ha passat a la partida anterior, ni del patró de joc del jugador. Un humà que sempre canta truc sense mà, o que mai aposta envits, és el mateix que un humà òptim per a la vista de l'agent — no té manera d'adaptar-se.

La Fase 4 introdueix dues variables noves respecte a Fase 3.5:

1. **Memòria seqüencial**: substituir el MLP post-COS per un **LSTM** que manté un estat `(h_t, c_t)` entre steps. Això permet a l'agent acumular informació al llarg de múltiples partides d'una sessió.
2. **Pool d'oponents divers**: entrenar contra múltiples variants d'`AgentRegles` amb estils de joc clarament diferents (conservador, agressiu, truc-bot, envit-bot, etc.). Un oponent fix no força l'agent a aprendre a adaptar-se; un pool divers sí.

La hipòtesi és:

> *Si entrenem un `RecurrentPPO` amb COS preentrenat i congelat contra un pool d'oponents variats, en sessions de N partides consecutives, l'LSTM aprendrà a detectar el patró de cada oponent als primers steps i ajustar la política. Això s'hauria de traduir en un **WR creixent al llarg de la sessió** (partida 1 < partida 5).*

## El model de memòria: LSTM

### Hidden state i propagació temporal

Una xarxa LSTM (Long Short-Term Memory) manté dos vectors d'estat `(h_t, c_t)` que evolucionen al llarg del temps:

- `c_t` (cell state): memòria a llarg termini, es modifica poc pels gates.
- `h_t` (hidden state): sortida visible a la capa següent, actua com a context condensat.

A cada step `t`, la LSTM rep l'entrada `x_t` (en el nostre cas, les 256 dims del COS) i produeix `(h_t, c_t)` a partir de `(h_{t-1}, c_{t-1}, x_t)`. El flux gradient a través del temps (BackProp Through Time, BPTT) permet que els gradients al temps `t` afectin els pesos que van generar estats anteriors — és a dir, l'LSTM pot aprendre a desar informació rellevant al principi de la seqüència i recuperar-la més tard.

### Truncated BPTT dins PPO

`RecurrentPPO` (de `sb3_contrib`) implementa PPO amb LSTM mitjançant **truncated BPTT** sobre seqüències de longitud `n_steps`. Per cada entorn vectoritzat, es recullen `n_steps` transicions; aquestes formen una seqüència amb estat LSTM propagat. L'entrenament divideix el rollout en minibatches mesurats en **seqüències** (no en transicions) — un detall crític que afecta com cal configurar `batch_size`.

El flag `episode_start` és la senyal que marca quan resetejar l'estat LSTM: quan `terminated=True` al step anterior, el proper step té `episode_start=True` i l'estat es reinicia a zeros.

### Per què RecurrentPPO i no DRQN

DRQN (Deep Recurrent Q-Network) seria l'anàleg off-policy: DQN amb LSTM. Requereix un **replay buffer seqüencial** que emmagatzemi finestres de transicions en lloc de transicions individuals. Aquest buffer no està disponible a SB3 ni a `sb3_contrib`; implementar-lo des de zero implica reescriure la lògica de mostreig i sincronització buffer↔LSTM, que queda fora de l'abast d'aquest TFG. La Fase 4 se centra en PPO on-policy.

Tot i això, **DQN frozen (F3.5, 91.2%)** continua sent la millor referència sense memòria del costat off-policy, i apareix al quadre de comparació.

## El disseny de sessions multi-partida

### Episodi RL = N partides

A la Fase 3.5 l'episodi RL era una única mà. A la Fase 4 l'episodi és una **sessió de N=5 partides consecutives** contra el mateix oponent. La raó és estructural: l'LSTM només es reseteja quan `terminated=True`, de manera que per aprofitar la memòria cross-partides cal que el senyal terminal no arribi fins al final de les 5 partides. El nou wrapper `TrucGymEnvSessio` envolta `TrucGymEnvMa` i intercepta els `terminated=True` intermedis perquè l'agent percebi la sessió com un episodi continu.

Al principi de cada sessió es samplea un oponent nou del pool, i `terminated=True` al final de la sessió marca el canvi d'oponent.

### El pool de 6 variants

L'objectiu del pool és **obligar l'LSTM a adaptar-se**. Si l'oponent fos únic (com a Fase 3.5), un agent sense memòria amb prou capacitat podria memoritzar una política òptima mitjana. Amb sis variants clarament diferenciades (truc_agressio entre 0.4 i 2.0, farol_prob entre 0.02 i 0.40), no existeix una política única òptima: cal detectar qui tens al davant i ajustar.

Els sis perfils del pool són:

| Nom         | truc_agressio | envit_agressio | farol_prob | resposta_truc |
| :---------- | ------------: | -------------: | ---------: | ------------: |
| Conservador |           0.5 |            0.5 |       0.02 |           0.6 |
| Agressiu    |           1.8 |            1.8 |       0.30 |           1.5 |
| Truc-bot    |           2.0 |            0.4 |       0.15 |           1.2 |
| Envit-bot   |           0.4 |            2.0 |       0.05 |           0.7 |
| Faroler     |           1.3 |            1.3 |       0.40 |           1.3 |
| Equilibrat  |           1.0 |            1.0 |       0.12 |           1.0 |

Aquests quatre paràmetres s'han afegit al constructor d'`AgentRegles`; els valors per defecte (1.0, 1.0, 0.12, 1.0) reprodueixen exactament el comportament previ, mantenint compatibilitat amb Fases 2–3.5.

### L'arquitectura de l'agent F4-complet

La cadena processament és:

```
obs (240) → CosMultiInputSB3 (frozen, 256) → LSTM(256) → MLP lleuger → policy/value heads
```

El COS preentrenat i congelat produeix embeddings estables sobre l'estat observable (cartes, context, rondes jugades). L'LSTM rep aquests embeddings i hi acumula el context temporal, concentrant-se exclusivament en detectar patrons de l'oponent. El MLP final (`net_arch=[256]`) és lleuger perquè l'LSTM ja fa la feina representacional pesada.

**Per què COS frozen i no COS finetune o scratch**: a Fase 3.5 es va validar que per a PPO, **frozen** (89.5%) supera o iguala finetune (~90%) i és substancialment més estable. El COS congelat proporciona embeddings fiables que l'LSTM pot usar com a base sòlida; si el COS es desestabilitzés durant l'entrenament RL, l'LSTM hauria d'adaptar-se a una representació canviant — un problema ortogonal al que volem estudiar.

## Tres comparacions, tres preguntes

| Run             | Memòria | Pool divers | Pregunta que respon                     |
| :-------------- | :------- | :---------- | :-------------------------------------- |
| F3.5 PPO frozen | No       | No          | Baseline absolut (reutilitzat)          |
| F3.5 DQN frozen | No       | No          | Millor agent off-policy sense memòria  |
| F4-ablació     | No       | Sí         | *Efecte del pool divers* (sense LSTM) |
| F4-complet      | Sí      | Sí         | *Efecte de la memòria* sobre el pool |

**F3.5 → F4-ablació**: aïlla l'efecte d'entrenar contra estils variats. Un agent sense memòria pot aprendre una política més robusta en mitjana, però no pot especialitzar-se per oponent.

**F4-ablació → F4-complet**: aïlla l'efecte de la memòria. Si l'LSTM aporta valor, es veurà com a millora de WR global *i* com a corba ascendent dins la sessió (WR_pos_5 > WR_pos_1).

## Hipòtesis i riscos

### Hipòtesis principals

- **H1**: F4-complet > F4-ablació en WR global (almenys +3 pp).
- **H2**: F4-complet mostra corba d'adaptació ascendent dins la sessió: WR a la partida 5 superior a WR a la partida 1 en almenys 5 pp.
- **H3**: F4-ablació ≈ F3.5 PPO frozen (dins ±3 pp); el pool divers no degrada ni millora significativament si no hi ha memòria.

### Riscos

- **LSTM no aprèn a adaptar-se**: si el gradient a través de 5 partides (~500 steps) és massa petit, l'LSTM pot quedar-se com a "feature extractor temporal passiu" sense comportament adaptatiu real. Detecció: si WR_pos_5 ≈ WR_pos_1, la hipòtesi H2 falla.
- **Catastrophic forgetting al pool**: entrenant contra 6 oponents simultàniament, l'agent pot oscil·lar entre polítiques especialitzades per a cada un i acabar sense aprendre cap. El COS congelat mitiga parcialment això (la representació d'entrada no es desestabilitza).
- **Cost computacional**: `RecurrentPPO` és més lent que PPO estàndard (BPTT), i les sessions de 5 partides fan els rollouts més llargs. Per això s'ha reduït el pressupost a 12M steps per run (24M acumulats), meitat que F3.5.

## Connexió amb les fases

La Fase 4 tanca l'eix d'exploració de **representació + memòria** sobre el millor setup de F3.5. Amb això es té resposta a:

- Q1 (F3): arquitectura COS vs MLP.
- Q2 (F3.5): preentrenament supervisat del COS (scratch/frozen/finetune).
- Q3 (F4): memòria cross-partides.

Els resultats de la millor combinació alimenten el desplegament final a `demo.py`, on un jugador humà pot jugar una sessió de 5 partides contra l'agent amb memòria persistent.

## Enllaços creuats

- [[13_Fase35_Implementacio]] — font del millor model (PPO frozen) i dels pesos COS preentrenats.
- [[12_Fase35_MarcTeoric]] — fonaments del preentrenament que Fase 4 reutilitza.
- [[4_Estructura_Models]] — descripció de `CosMultiInput`.
- [[15_Fase4_Implementacio]] — detalls tècnics i resultats numèrics de la Fase 4.
