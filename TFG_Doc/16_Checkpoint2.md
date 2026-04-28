# Checkpoint 2 — Tancament de Fase 3, Fase 3.5 i Fase 4

Aquest document resumeix les conclusions de les tres fases experimentals centrals del TFG i justifica quin model s'adopta com a agent final per a desplegament.

## Context

Les tres fases d'aquest bloc partien dels dos models seleccionats al Checkpoint 1 (DQN-SB3 i PPO-SB3, protocol mans) i exploraven si una millor representació de l'estat i/o la memòria inter-mans podien superar els seus resultats. Les tres preguntes de recerca eren:

- **Q1** (Fase 3): Aporta valor el feature extractor COS respecte al MLP estàndard?
- **Q2** (Fase 3.5): Dona valor el preentrenament supervisat del COS, i quin regim (frozen vs finetune) es preferible?
- **Q3** (Fase 4): Pot un agent amb memoria (LSTM) adaptar-se al comportament d'un oponent desconegut al llarg d'una sessio de mans consecutives?

La metrica principal en totes les fases es `metric = 0.25 * WR_random + 0.75 * WR_regles`, avaluada sempre sobre partides senceres.

## Fase 3 — Arquitectura COS vs MLP

Detall complet a [11_Fase3_Implementacio.md](11_Fase3_Implementacio.md).

Sis protocols x dues arquitectures (COS scratch, MLP) per a DQN-SB3 i PPO-SB3, 24M steps cadascun. Les sèries MLP dels protocols control i curriculum provenen de Fase 2.

**Taula resum (pic metric, 24M steps):**

| Protocol   | Agent   | COS scratch |   MLP |    Delta |
| :--------- | :------ | ----------: | ----: | -------: |
| Control    | DQN-SB3 |       71.0% | 60.5% | +10.5 pp |
| Control    | PPO-SB3 |       32.5% | 35.0% |  -2.5 pp |
| Curriculum | DQN-SB3 |       56.2% | 56.0% |  +0.2 pp |
| Curriculum | PPO-SB3 |       63.0% | 75.0% | -12.0 pp |
| Mans       | DQN-SB3 |       82.2% | 85.8% |  -3.6 pp |
| Mans       | PPO-SB3 |       89.0% | 87.2% |  +1.8 pp |

**Conclusions:**

1. El protocol **mans** es el mes productiu: totes les combinacions superen el 82%, molt per sobre del control o el curriculum.
2. El COS scratch no supera sistematicament el MLP. Al protocol mans, les diferencies son inferiors a 4 pp i no son consistents entre agents.
3. El preentrenament del COS es una pregunta separada: l'arquitectura sola (pesos aleatoris) no es suficient per desbloquejar un avantatge clar.
4. **Decisio de Fase 3**: fixar el protocol mans per a la fase seguent i provar regims d'inicialitzacio supervisada del COS.

## Fase 3.5 — Regims d'inicialitzacio del COS preentrenat

Detall complet a [13_Fase35_Implementacio.md](13_Fase35_Implementacio.md).

Protocol mans (24M steps) per a tots els runs. Quatre regims comparats: sense COS (MLP), scratch, frozen i finetune.

**Taula resum (protocol mans, pic metric, 24M steps):**

| Regim           |         DQN-SB3 |         PPO-SB3 |
| :-------------- | --------------: | --------------: |
| Sense COS (MLP) |           85.8% |           87.2% |
| Scratch         |           82.2% |           89.0% |
| Frozen          | **91.2%** | **89.5%** |
| Finetune        |           88.5% |           88.0% |

**Conclusions:**

1. **Frozen es el millor regim per a DQN** (91.2%, +5.4 pp sobre MLP, +9 pp sobre scratch). El cos preentrenat i congelat actua com a extractor estable que permet a la Q-network aprendre sense interferencies.
2. **Finetune degrada respecte a frozen**: el gradient RL deteriora parcialment la representacio preentrenada tant per a DQN (88.5% vs 91.2%) com per a PPO (88.0% vs 89.5%).
3. **Per a PPO, els quatre regims convergeixen entre 87-89%**: scratch es practicament equivalent a frozen; el preentrenament no aporta millora significativa per a l'agent on-policy.
4. **Scratch vs MLP**: diferencia inferior a 4 pp i no consistent, confirmant el resultat de Fase 3.
5. **Resposta a Q2**: el preentrenament supervisat aporta valor real, pero nomes quan el cos queda congelat. El frozen supera el MLP en tots dos agents.

## Fase 4 — Memoria LSTM i pool d'oponents

Detall complet a [15_Fase4_Implementacio.md](15_Fase4_Implementacio.md).

Dos nous runs de 12M steps sobre sessions de N=5 mans consecutives amb pool de 6 variants d'AgentRegles, mes les baselines DQN frozen i PPO frozen de Fase 3.5.

**Taula resum (pic metric dins 12M steps — pressupost igualat per comparabilitat):**

Nota: DQN frozen te el seu pic global a 23M steps (91.2%); aqui es mostra el seu maxim dins el pressupost de 12M de Fase 4.

| Run               | Memoria  | Pool | Pic metric a 12M | Step pic | Temps | Delta WR pos (*) |
| :---------------- | :------- | :--- | ---------------: | -------: | ----: | ---------------: |
| DQN frozen (F3.5) | No       | No   |           87.75% |     4.0M | 2.14h |               -- |
| PPO frozen (F3.5) | No       | No   |           89.5%  |    11.0M | 0.33h |               -- |
| F4-ablacio        | No       | Si   |           86.0%  |     9.5M | 0.33h |         +4.3 pp  |
| F4-complet        | LSTM 256 | Si   |           82.0%  |    10.0M | 15.2h |         +1.4 pp  |

(*) Delta WR pos = WR_pos_5 - WR_pos_1: diferencia de win-rate entre la 5a i la 1a ma dins la sessio. Mesura si l'agent millora a mesura que avanca la sessio (adaptacio a l'oponent). Nomes aplicable als runs F4 que usen sessions.

**Validacio d'hipotesis:**

| Hipotesi                            | Criteri                                   | Resultat | Estat   |
| :---------------------------------- | :---------------------------------------- | -------: | :------ |
| H1 -- LSTM aporta valor global      | F4-complet - F4-ablacio >= +3 pp          |  -4.0 pp | Falla   |
| H2 -- Adaptacio dins la sessio      | WR_pos_5 - WR_pos_1 >= +5 pp (F4-complet) |  +1.4 pp | Falla   |
| H3 -- Pool no degrada sense memoria | abs(F4-ablacio - PPO frozen) <= 3 pp      |   3.5 pp | Parcial |

**Conclusions:**

1. **H1 falla**: l'LSTM no millora el rendiment global. F4-complet (82.0%) queda 4 pp per sota de F4-ablacio (86.0%) amb un cost computacional 46x superior (15.2h vs 0.33h).
2. **H2 falla**: no hi ha adaptacio clara dins la sessio. La corba wr_pos de F4-complet es sorollosa ([77.5, 73.2, 80.7, 71.1, 78.9]) i el patró de variabilitat apareix tambe a F4-ablacio (sense LSTM), indicant un artefacte de mostra i no aprenentatge adaptatiu real.
3. **H3 parcial**: el pool dificulta lleugerament la convergencia sense memoria (3.5 pp de degradacio respecte a PPO frozen F3.5).
4. **Causa probable del fracas de l'LSTM**: budget insuficient (12M vs 24M de les baselines), sessions d'entrenament curtes (~50 steps de BPTT vs ~500 steps d'avaluacio), i mismatch train/eval (entrenament sobre 5 mans, avaluacio sobre 5 partides senceres).
5. **Resposta a Q3**: amb el disseny actual, la memoria LSTM no millora l'adaptacio a oponents desconeguts de manera mesurable.

## Decisio: model final per a desplegament

El model seleccionat per a desplegament es **COS + PPO entrenat per mans amb pool d'agents (F4-ablacio, 86.0% a 12M steps)**.

### Justificacio

**Robustesa davant estils diversos.** F4-ablacio s'ha entrenat contra 6 variants d'AgentRegles (conservador, agressiu, truc-bot, envit-bot, faroler, equilibrat), cosa que li dona una politica generalista que les baselines de Fase 3.5 no tenen: aquestes van entrenar sempre contra un oponent fix (10% Random + 90% AgentRegles estandard).

**Rapidesa d'entrenament.** 0.33h per run contra les 15.2h de F4-complet o les hores necessaries per a DQN. Aixo permet iterar i tornar a entrenar si cal.

**Rendiment competitiu en el pressupost igualar.** A 12M steps, F4-ablacio (86.0%) queda nomes 3.5 pp per sota de PPO frozen F3.5 (89.5%), una diferencia modesta per l'avantatge de diversitat d'oponents.

**LSTM no justifica el cost.** F4-complet te un cost 46x superior i un rendiment 4 pp inferior. Cap de les tres hipotesis de Fase 4 ha estat validada amb el disseny actual.

**DQN frozen te millor WR global (91.2% al pic complet, 87.75% a 12M) pero menys robustesa.** Va ser entrenat contra un oponent fix i no s'ha provat contra el pool divers. PPO + pool es preferible per a un context de desplegament real on l'oponent es desconegut.

## Extensio natural

Una extensio directa que podria combinar els avantatges seria **DQN frozen + pool d'oponents**: DQN te la millor WR global (91.2%) i el pool aportaria diversitat. No s'ha executat per limitacions de temps, pero es la progressio logica si es vol maximitzar el rendiment brut mantenint robustesa.

Igualment, un experiment de Fase 4 alineat (train i eval tots dos sobre mans, o tots dos sobre partides senceres) i amb 24M steps o sessions N >= 10 podria revelar si la memoria LSTM te potencial real en un disseny sense el mismatch actual.

## Enllacos creuats

- [[11_Fase3_Implementacio]] -- resultats detallats de Fase 3 (COS vs MLP).
- [[13_Fase35_Implementacio]] -- resultats detallats de Fase 3.5 (regims d'inicialitzacio).
- [[15_Fase4_Implementacio]] -- resultats detallats de Fase 4 (LSTM + pool).
- [[9_Checkpoint1]] -- seleccio dels models de Fase 1 i Fase 2, pla de les fases 3-5.
