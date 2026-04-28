# 15. Fase 4: Implementació i Resultats

Aquest document descriu la implementació tècnica de la Fase 4 definida a [[14_Fase4_MarcTeoric]]. L'objectiu és respondre:

> **Q3:** Pot un agent amb memòria (LSTM) adaptar-se al comportament d'un oponent desconegut al llarg d'una sessió de partides consecutives?

## Disseny experimental

**2 nous runs** de 12M steps + **2 baselines reutilitzades**:

| Run                   | Agent            | Memòria           | Pool oponents        | Sessions      | Font             |
| :-------------------- | :--------------- | :----------------- | :------------------- | :------------ | :--------------- |
| DQN frozen            | SB3 DQN          | No                 | No                   | No            | F3.5 reutilitzat |
| PPO frozen            | SB3 PPO          | No                 | No                   | No            | F3.5 reutilitzat |
| **F4-ablació** | SB3 PPO          | No                 | **6 variants** | **N=5** | **nou**    |
| **F4-complet**  | SB3 RecurrentPPO | **LSTM 256** | **6 variants** | **N=5** | **nou**    |

Tots els runs comparteixen:

- Arquitectura del cos: `CosMultiInputSB3` amb `best_pesos_cos_truc.pth` carregat i **congelat**.
- Hiperparàmetres PPO: LR 3e-4, γ 0.995, GAE 0.95, clip 0.2, ent 0.01, vf 0.5, n_epochs 7 (iguals que F2/F3.5).
- Mètrica principal: `metric = 0.25·WR_random + 0.75·WR_pool`.
- Avaluació sobre **partides senceres** (`TrucGymEnv`), cada 500 000 steps.

## El pool d'oponents

`pool_oponents.py` defineix 6 variants d'`AgentRegles` parametritzades:

```python
POOL_OPONENTS = [
    ("conservador", dict(truc_agressio=0.5, envit_agressio=0.5, farol_prob=0.02, resposta_truc=0.6)),
    ("agressiu",    dict(truc_agressio=1.8, envit_agressio=1.8, farol_prob=0.30, resposta_truc=1.5)),
    ("truc_bot",    dict(truc_agressio=2.0, envit_agressio=0.4, farol_prob=0.15, resposta_truc=1.2)),
    ("envit_bot",   dict(truc_agressio=0.4, envit_agressio=2.0, farol_prob=0.05, resposta_truc=0.7)),
    ("faroler",     dict(truc_agressio=1.3, envit_agressio=1.3, farol_prob=0.40, resposta_truc=1.3)),
    ("equilibrat",  dict(truc_agressio=1.0, envit_agressio=1.0, farol_prob=0.12, resposta_truc=1.0)),
]
```

`sample_oponent(rng)` tria uniformement una variant per cada sessió. `crear_oponent(nom)` instancia una variant específica (útil a l'avaluació per tipus).

## L'entorn `TrucGymEnvSessio`

Wrapper sobre `TrucGymEnvMa` que agrupa N **mans** consecutives en un sol episodi RL. Nota: `TrucGymEnvMa` opera a nivell de mà (cada episodi = 1 mà completa), per tant una sessió de N=5 encadena 5 mans consecutives, no 5 partides senceres:

```python
def step(self, action):
    obs, reward, terminated, truncated, info = self._inner.step(action)
    if terminated:
        self._partides_fetes += 1
        if self._partides_fetes < self.n_partides:
            obs, _ = self._inner.reset()  # nova mà, mateix oponent
            terminated = False             # LSTM NO es reseteja
    return obs, reward, terminated, truncated, info
```

El punt clau és la **intercepció dels `terminated=True` intermedis**: l'entorn interior (`TrucGymEnvMa`) retorna `terminated=True` al final de cada **mà**, però el wrapper només propaga aquest senyal quan s'han completat les N mans. Això manté l'estat LSTM viu durant tota la sessió.

L'info que retorna `step()` inclou:

- `partida_idx`: índex de la partida en curs (0..N-1).
- `partida_acabada`: `True` quan acaba una partida (per diagnòstic).
- `oponent_nom`: nom de la variant contra la qual es juga (per logging).

## L'entrenament

### F4-ablació

PPO estàndard sobre `TrucGymEnvSessio`. Idèntic a F3.5 PPO frozen, però canvia:

- Entorn: sessions de 5 partides, oponent samplejat del pool.
- Steps: 12M (en lloc de 24M).

```python
policy_kwargs = dict(
    features_extractor_class=CosMultiInputSB3,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=HIDDEN_LAYERS, vf=HIDDEN_LAYERS),
)
model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, ...)
_aplicar_frozen(model, pesos_cos, lr=PPO_LR)
```

### F4-complet

`RecurrentPPO` amb LSTM sobre els embeddings del COS congelat:

```python
from sb3_contrib import RecurrentPPO

policy_kwargs = dict(
    features_extractor_class=CosMultiInputSB3,
    features_extractor_kwargs=dict(features_dim=256),
    lstm_hidden_size=256,
    n_lstm_layers=1,
    enable_critic_lstm=True,
    net_arch=dict(pi=[256], vf=[256]),  # MLP lleuger post-LSTM
)
model = RecurrentPPO("MlpLstmPolicy", vec_env,
                    learning_rate=PPO_LR, n_steps=PPO_N_STEPS,
                    batch_size=batch_size_seqs,  # mesurat en SEQÜÈNCIES
                    ...)
_aplicar_frozen(model, pesos_cos, lr=PPO_LR)
```

**Punt tècnic — `batch_size` en seqüències**: `RecurrentPPO` interpreta `batch_size` com a nombre de **seqüències**, no de transicions individuals. Amb `n_envs=32` hi ha 32 seqüències per rollout, cadascuna de `n_steps` transicions. Un `batch_size_seqs=8` processa 8 × `n_steps` transicions per minibatch, equivalent al volum que F2/F3.5 processaven amb `batch_size=2048` de transicions.

## L'avaluació per sessions

La funció `evaluar_sessions()` (dins `entrenament_fase4.py`) reemplaça el mètode d'avaluació estàndard:

```python
def evaluar_sessions(eval_agent, n_partides_sessio, n_sessions_random, n_sessions_pool):
    def _jugar_sessio(oponent, is_random):
        if hasattr(eval_agent, 'reset'):
            eval_agent.reset()  # reset LSTM al principi de sessió
        for idx in range(n_partides_sessio):
            # ... juga 1 partida sencera, acumula resultat per posició idx ...
    ...
    return wr_random, wr_pool, metric, wr_per_posicio  # wr_per_posicio: list[N]
```

Cada sessió reseteja l'estat LSTM; dins la sessió, les N partides es juguen consecutivament sense reset. Els resultats s'acumulen per **posició** (índex de partida dins la sessió) per poder mesurar adaptació.

El log (`training_log.csv`) té columnes addicionals respecte a F3.5:

```
step, wr_random, wr_regles, metric, wr_pos_1, wr_pos_2, wr_pos_3, wr_pos_4, wr_pos_5, elapsed
```

## Resultats (run complet, 12M steps)

### Taula resum

| Run               | Memòria           | Pool |      Pic metric | Step pic | Temps | ΔWR_pos |
| :---------------- | :----------------- | :--- | --------------: | -------: | ----: | -------: |
| DQN frozen (F3.5) | No                 | No   | **91.2%** |    23.0M |    — |       — |
| PPO frozen (F3.5) | No                 | No   |           89.5% |    11.0M |    — |       — |
| F4-ablació       | No                 | Sí  |           86.0% |     9.5M | 0.33h |  +4.3 pp |
| F4-complet        | **LSTM 256** | Sí  |           82.0% |    10.0M | 15.2h |  +1.4 pp |

### Validació d'hipòtesis

| Hipòtesi                                      | Criteri                                    | Resultat | Validada? |
| :--------------------------------------------- | :----------------------------------------- | -------: | :-------- |
| **H1** — LSTM aporta valor global       | F4-complet − F4-ablació ≥ +3 pp         | −4.0 pp | ✗ Falla  |
| **H2** — Adaptació dins la sessió     | WR_pos_5 − WR_pos_1 ≥ +5 pp (F4-complet) |  +1.4 pp | ✗ Falla  |
| **H3** — Pool no degrada sense memòria | \|F4-ablació − PPO frozen F3.5\| ≤ 3 pp |   3.5 pp | ~ Parcial |

### Lectures principals

**H1 falla — l'LSTM no millora el rendiment global.** F4-complet (82.0%) queda 4 pp per sota de F4-ablació (86.0%). El cost computacional és 46× superior (15.2h vs 0.33h) sense cap guany mesurable.

**H2 falla — no hi ha adaptació clara dins la sessió.** La corba de F4-complet és sorollosa i no monòtona: wr_pos = [77.5, 73.2, 80.7, 71.1, 78.9] (Δ = +1.4 pp). El patró de pic a la partida 3 i vall a la partida 4 apareix igualment a F4-ablació (sense LSTM), indicant un artefacte de la mida de mostra de l'avaluació (~56 sessions) i no aprenentatge adaptatiu real.

**H3 parcial — el pool dificulta lleugerament sense memòria.** F4-ablació (86.0%) és 3.5 pp per sota de PPO frozen F3.5 (89.5%). La diversitat d'oponents complica la convergència sense que l'agent pugui especialitzar-se.

**Causa probable del fracàs de l'LSTM:**

- **Budget insuficient**: 12M steps (meitat que F3.5) per a `RecurrentPPO`, que té significativament més paràmetres.
- **Sessions curtes d'entrenament**: N=5 mans (~50 steps totals) — el BPTT no té finestra suficient per propagar senyals d'adaptació.

Un experiment alineat (tots dos protocols amb mans, o tots dos amb partides) i amb 24M steps o sessions N≥10 podria revelar el potencial real de la memòria.

**Model recomanat per a desplegament:** **COS + PPO entrenat per mans amb pool d'agents (F4-ablació, 86.0%)**. Entrena contra 6 variants d'`AgentRegles`, és ràpid d'entrenar (0.33h) i produeix una política robusta davant estils diversos. DQN frozen (F3.5, 91.2%) té millor WR global però va ser entrenat contra un oponent fix; una extensió natural seria DQN amb pool, però la rapidesa de PPO el fa preferible per iterar.

## Enllaços creuats

- [[14_Fase4_MarcTeoric]] — fonaments teòrics de l'LSTM i del disseny del pool.
- [[13_Fase35_Implementacio]] — font del millor model (PPO frozen) i pattern de càrrega COS frozen reutilitzat.
- [[9_Checkpoint1]] — selecció dels dos models i pla de fases.
