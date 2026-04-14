# 7. Fase 2: Marc Teòric (Entorn per Mans i Curriculum Learning)

Aquest document introdueix els components que fan possible l'enfocament de la Fase 2: l'entorn per **mans individuals** (una mà = un episodi complet) i el concepte teòric de **curriculum learning** aplicat al Truc.

## Motivació

Com es mostra als resultats de la Fase 1 ([[6_Fase1_Resultats]]), els quatre algorismes s'estanquen en una mètrica modesta (30–40%) quan entrenen directament sobre partides senceres. Les raons estructurals són:

- **Horizon llarg**: una partida fins a 24 punts pot durar desenes de mans i centenars d'steps. Això dilueix el senyal del reward final.
- **Reward sparse**: la recompensa oficial arriba només al final (`±1.0`). El *reward shaping* intermedi implementat a `TrucGame` mitiga el problema però no el resol.
- **Mescla tàctica + estratègia**: l'agent ha d'aprendre simultàniament a jugar bé una mà *i* a gestionar el marcador global. Són dos problemes acoblats.

La solució clàssica en RL quan un problema és massa llarg i sparse és **factoritzar-lo** en subproblemes més petits i densos. Al Truc això surt gratis: una *mà* és una unitat natural del joc amb regles ben definides, reward calculable al final i durada acotada (~10–20 steps).

## `TrucGameMa` — Motor per mans

La classe `TrucGameMa` hereta de `TrucGame` i redefineix el comportament de fi de partida per tal que **cada mà sigui un episodi complet**. Els canvis clau són:

- `puntuacio_final=999` (salvaguarda perquè `is_over()` mai retorni `True` pel marcador).
- Quan una mà acaba, en lloc d'executar `_reset_hand_state()` i continuar la partida, es crida `_end_ma()` que tanca l'episodi i torna `(state, None)`.
- El reward s'acumula en una variable `reward_intermedis = [r_equip_0, r_equip_1]` **només** al final de la mà, en forma de **reward net normalitzat** `pts / 24.0`.

Detalls complets a [[2_Logica_Joc]] (secció "Entorn d'Entrenament per Mans").

## TrucEnvMa — Adaptador RLCard per Mans

`TrucEnvMa` (a `joc/entorn_ma/env_ma.py`) és funcionalment idèntica a `TrucEnv` (veure [[3_Entorns_Simulacio_RL]] per als detalls d'extracció d'estats) en tot el que respecta a l'extracció d'observacions. La **única diferència** és el motor de joc subjacent:

| Aspecte | `TrucEnv` | `TrucEnvMa` |
|:--------|:----------|:-------------|
| Motor de joc | `TrucGame` | `TrucGameMa` |
| `self.name` | `'truc'` | `'truc_ma'` |
| Observació `obs_cartes` | `(6, 4, 9)` ✓ | `(6, 4, 9)` ✓ — **idèntica** |
| Observació `obs_context` | `(23,)` ✓ | `(23,)` ✓ — **idèntica** |
| `state_size` | `239` | `239` — **idèntic** |
| `_extract_state()` | Codi idèntic | Codi idèntic |

Aquesta identitat d'observacions és deliberada: permet que **el mateix model neuronal** (MLP o GRU) pugui entrenar-se amb qualsevol dels dos entorns sense cap modificació arquitectural. L'única variable que canvia és la **durada i reward de l'episodi**. És la precondició tècnica que fa possible el curriculum.

`TrucGymEnvMa` (a `joc/entorn_ma/gym_env_ma.py`) és el wrapper Gymnasium anàleg a `TrucGymEnv`, preparat per SB3. Episodi = 1 mà; el `done=True` arriba quan el motor retorna `player_id=None`.

#### Estructura de l'Observació (recordatori)

Com a referència ràpida (detalls complets a [[3_Entorns_Simulacio_RL]]):

**`obs_cartes` (6, 4, 9)** — Tensor de cartes one-hot:
- Canal 0: Mà actual del jugador
- Canals 1-4: Historial de cartes (propi, rival 1, company, rival 2)
- Canal 5: Cartes assenyalades pel company

**`obs_context` (23,)** — Vector de context:
- [0-1]: Puntuació equip propi/rival (÷24)
- [2-3]: Nivell truc/envit (÷24)
- [4]: Fase del torn
- [5]: Comptador ronda (÷cartes)
- [6-9]: One-hot qui és mà
- [10-13]: One-hot qui ha cantat truc
- [14-16]: One-hot qui ha cantat envit
- [17-18]: Rondes guanyades propi/rival (÷3)
- [19-20]: Guanyador R1/R2 (+1/−1/0)
- [21]: Envit acceptat (0/1)
- [22]: Response state (0.0/0.5/1.0)

## Curriculum Learning: el concepte

El *curriculum learning* (Bengio et al., 2009) és una tècnica inspirada en l'aprenentatge humà: presentar a l'aprenent **primer els exemples més simples i després els més complexos**. En RL, "simple" sol significar:

- Horizons més curts.
- Reward més dens o més informatiu.
- Espai d'estats/accions restringit.
- Condicions inicials més fàcils.

L'efecte teòric esperat és **accelerar la convergència** perquè l'agent pot aprendre primer les regularitats locals (què fa una bona acció dins una mà?) sense haver de lluitar amb la variància gegant del reward d'una partida sencera. Un cop ja té una política raonable per mans, fer *finetune* sobre partides senceres és un problema molt més fàcil.

### Aplicació al Truc

L'estratègia del projecte és un curriculum de **dues etapes**:

1. **Etapa 1 — Mans** (`TrucGymEnvMa`, 12M steps):
   - Cada episodi dura ~15 steps.
   - Reward calculat al final de la mà amb `(pts_truc + pts_envit) / 24`.
   - L'agent aprèn tàctica local: quan jugar forta, quan acceptar un envit, quan bluffar un truc.
   - Sense preocupar-se pel marcador global (que és constant 0-0 a cada episodi).

2. **Etapa 2 — Partides** (`TrucGymEnv`, 12M steps):
   - Pesos carregats de l'etapa 1.
   - Episodi fins a 24 punts, reward final ±1.
   - L'agent refina la política per tenir en compte el marcador: quan el partit està quasi guanyat, quan cal arriscar més perquè anem a perdre…

Els dos blocs sumats són exactament el mateix pressupost (24M steps) que el grup de control, que entrena directament en partides. Això permet comparar *amb justícia*.

### Riscos teòrics

- **Catastrophic forgetting**: quan canviem d'entorn, el gradient pot destruir la política apresa a l'etapa 1. Això és especialment perillós per a algorismes on-policy com PPO.
- **Shift de distribució**: la distribució de mans que apareixen quan el marcador és 10-18 és diferent de la que apareix a 0-0. Un agent entrenat sense marcador pot sobreoptimitzar per la situació "tot igualat".
- **Mala correspondència de reward**: el reward per mà (`pts/24`) no és el mateix que el reward per partida (`±1`). Si els magnituds no casen, el finetune pot ser inestable.

### Mitigacions aplicades

- **Sense self-play** a Fase 2: per reduir variables, s'elimina el self-play i tots els agents s'entrenen amb `10% Random + 90% AgentRegles`. Així, l'efecte observat entre control i curriculum s'atribueix al canvi d'entorn, no a l'oponent.
- **Exploration rate reduïda al finetune** (DQN-SB3): `exploration_initial_eps = 0.10` (en lloc de 1.0), perquè la política ja està inicialitzada i no cal explorar des de zero.
- **Càrrega de pesos explicita**: els agents que ho suporten (DQN-SB3, PPO-SB3) carreguen el `best_mans.zip` al principi de l'etapa 2 via `PPO.load()` o modificant `q_net.load_state_dict()` segons el cas.

La implementació i els resultats concrets es desenvolupen a [[8_Fase2_Implementacio]].
