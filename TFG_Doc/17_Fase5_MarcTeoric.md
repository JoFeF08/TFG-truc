# 17. Fase 5: Marc Teoric — Self-Play Mixt i Explotabilitat

## Motivacio

F4-ablacio (PPO + COS frozen + pool AgentRegles, 86% WR a 12M steps) es el millor model obtingut fins ara. Tanmateix, presenta un defecte estructural: ha convergit a una politica quasi-determinista especialitzada en explotar els patrons fixos de les 6 variants d'AgentRegles. Contra un oponent huma que varia el seu estil, aquesta especialitzacio es converteix en una debilitat: el model es predictible i explotable.

La metrica actual (`0.25 * WR_random + 0.75 * WR_pool_mean`) no detecta aquest problema perque promitja el rendiment entre variants i amaga la variancia. Un agent pot tenir 95% contra la variant conservadora i 60% contra el faroler i sortir amb una metrica aparentment bona.

Fase 5 ataca el problema des de dues bandes:

1. **Self-play mixt**: diversificar l'oponent d'entrenament afegint versions passades del propi agent.
2. **Metriques d'explotabilitat**: quantificar la robustesa sense necessitar un huma.

> **Q4**: Pot el self-play mixt reduir l'explotabilitat del PPO sense degradar el rendiment global?

---

## Fictitious Self-Play (FSP)

Fictitious Play (Brown, 1951) es un procediment iteratiu per a jocs 2-jugadors zero-sum: en cada iteracio, cada jugador tria la millor resposta a la politica **promig** de l'oponent en totes les iteracions anteriors. Brown va demostrar que la politica promig convergeix a un equilibri de Nash en jocs de suma zero finits.

Neural Fictitious Self-Play (Heinrich & Silver, 2016) adapta aquest principi a xarxes neuronals:

- L'agent **RL** aprèn la millor resposta contra la politica promig de l'oponent.
- Un agent **SL** aprèn la politica promig de l'RL via un reservoir buffer.
- A Nash, cap jugador pot millorar canviant unilateralment d'estrategia.

La variant d'aquesta fase, **Simplified FSP**, elimina el component SL (que es reserva per a Fase 6) i aproxima la politica promig de l'oponent mantenint una **finestra rodant de snapshots** del model. Samplejant oponents del pool de snapshots, el PPO entrena contra una distribucio d'estrategies que representa la seva propia evolucio historica.

### Per que funciona el pool de snapshots

Cada snapshot es la millor resposta del model en aquell moment. La distribucio uniforme sobre N snapshots aproxima la politica promig de les ultimes N iteracions. Si el pool es prou divers (N >= 6), el PPO no pot especialitzar-se en un unic patró: qualsevol estrategia explotable per una versio passada quedara exposada.

### Bootstrapping amb AgentRegles

Al principi de l'entrenament no hi ha snapshots. Sense oponents, el PPO no pot aprendre. La solucio es mantenir les 6 variants d'AgentRegles **sempre presents** al pool. Aixo garanteix:

- Aprenentatge basic des del primer step.
- Diversitat de base que evita col·lapsos en el periode de bootstrapping.
- Comparabilitat directa amb F4-ablacio (que usava exclusivament AgentRegles).

A mesura que s'acumulen snapshots, el pool creix: 6 AgentRegles + N_snapshots variants. El sample es uniform entre tots els candidats, de manera que la proporcio de self-play augmenta gradualment.

---

## Metriques noves

### metric_robust — robustesa inter-estil

```
metric_robust = WR_pool_mean - lambda * std(WR_per_variant)
```

On `WR_per_variant` es el vector de win-rates contra cadascuna de les 6 variants d'AgentRegles avaluades per separat, i `lambda = 0.5` penalitza la dispersio.

**Propietats:**

- Un agent Nash ideal te `std = 0` (guanya igual contra tots): `metric_robust = WR_pool_mean`.
- Un agent especialitzat te `std` alta: `metric_robust < WR_pool_mean`.
- Permet comparar agents amb el mateix WR_pool_mean pero distribucions molt diferents.

Exemple:

```
Agent A: WR = [95, 90, 88, 85, 80, 62]  mean=83.3  std=11.0  robust=83.3-5.5=77.8
Agent B: WR = [84, 83, 85, 82, 84, 81]  mean=83.2  std=1.4   robust=83.2-0.7=82.5
```

A i B tenen practicament el mateix WR_pool_mean pero B es molt mes robust.

### exploit_selfplay — auto-explotabilitat

```
exploit_selfplay = |WR_current_vs_recent_snapshots - 50|
```

On `WR_current_vs_recent_snapshots` es el WR mig del model actual contra els N snapshots mes recents del pool.

**Propietats:**

- A Nash perfecte: cap versio pot explotar sistemàticament cap altra -> WR ≈ 50% -> exploit = 0.
- Si el model actual guanya molt contra si mateix d'abans, indica que ha convergit a una estrategia nova que exploita la versio antiga: cycling potencial.
- Decreixent al llarg de l'entrenament es el senyal desitjat: el model cada cop es menys explotable per versions anteriors.

**Limitacio**: el WR natural augmenta amb l'entrenament perque el model "sap mes". Per aïllar l'efecte d'explotabilitat real del simple creixement de capacitat, nomes es compara contra els **N_RECENT_EVAL = 3 snapshots mes recents** (no contra els molt antics).-

## Enllacos creuats

- [[15_Fase4_Implementacio]] -- F4-ablacio: el model de partida i el problema de predictibilitat.
- [[16_Checkpoint2]] -- Justificacio de F4-ablacio com a model seleccionat.
- [[18_Fase5_Implementacio]] -- Detalls tecnics de la implementacio.
