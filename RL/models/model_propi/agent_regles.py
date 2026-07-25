import random
from joc.entorn.rols.judger import TrucJudger
from joc.entorn.cartes_accions import ACTION_SPACE

# Accions
PLAY_0 = ACTION_SPACE['play_card_0']
PLAY_1 = ACTION_SPACE['play_card_1']
PLAY_2 = ACTION_SPACE['play_card_2']
APOSTAR_ENVIT = ACTION_SPACE['apostar_envit']
APOSTAR_TRUC = ACTION_SPACE['apostar_truc']
VULL_ENVIT = ACTION_SPACE['vull_envit']
VULL_TRUC = ACTION_SPACE['vull_truc']
FORA_ENVIT = ACTION_SPACE['fora_envit']
FORA_TRUC = ACTION_SPACE['fora_truc']
PASSAR = ACTION_SPACE['passar']

FORCA_TOP = 90  # 3s, manilles, asos forts, B11, O10

# Estils curats en lloc de mostreig continu lliure de 4 parametres
# independents (que acabava produint sobretot variants "agressives" i cap
# d'unica dominant).
#
# Important: la variacio nomes es fa en la FREQUENCIA d'iniciar/blofejar
# (truc_agressio/envit_agressio/farol_prob). `resposta_truc` (disciplina
# per acceptar un truc/envit ja fet) es manté alta i similar a tots els
# estils: cap n'es un "pushover" que cedeixi nomes per pressio bruta. Si
# es deixa que "es rendeix facil sota pressio" fos la feblesa de mes d'un
# estil, la contraestrategia dominant acaba sent trivial ("empenyer sempre
# i no rendir-se mai"), i un agent entrenat contra aquest pool aprendria a
# blofejar fins al final en lloc de jugar be. L'explotacio de cada estil
# ha de venir de llegir-ne el patro (acceptar-li quan se sap que bloqueja
# sovint), no de pressio bruta.
#   conservador -> inicia poc -> cedeix la iniciativa, dona marge a l'altre.
#   equilibrat  -> valors de referencia.
#   agressiu    -> inicia molt -> exposa mans mediocres sovint, explotable
#                  per qui l'accepti/torni a pujar amb criteri.
#   farol       -> el mateix pero encara mes extrem amb els farols purs.
#
# Calibratge en dues fases (script fora del repo, RL/tools/calibrar_estils
# si es promou): (1) arbre de decisio/regressio (sklearn) sobre ~200
# configuracions aleatories, que va aillar una unica regio "robusta" a les
# cantonades extremes (farol_prob > 0.14, truc_agressio > 1.32,
# envit_agressio > 1.28), on cauen 'agressiu'/'farol'; (2) cerca local per
# apujar 'conservador'/'equilibrat' i estrenyer el marge de victories.
#
# Important: la primera ronda de mesures tenia un biaix metodologic -- el
# candidat avaluat sempre seia de "ma" (TrucGameMa fixa ma=0 sempre), i
# "ma" resulta ser ~8pp PITJOR posicio en aquest joc (qui respon veu la
# carta rival abans de jugar la seva). Un cop corregit alternant seients,
# 'agressiu'/'farol' guanyaven ~55-57% de mitjana contra 'conservador'/
# 'equilibrat' (~44-46%) -- massa marge -- i la cerca es va refer amb
# aquesta metodologia corregida.
#
# 'farol' es va retocar un cop mes perque no quedes com "l'opcio clarament
# mes forta" del grup (podria ensenyar a l'agent RL que blofejar sempre es
# la resposta, en lloc de llegir quan val la pena): reduir nomes els 4
# parametres no ho baixa gaire mes enlla d'un ~54-56% (sembla estructural,
# no nomes calibratge), aixi que a mes de retocar els numeros es baixa el
# pes de mostreig de 'farol'/'agressiu' a OpponentPool perque l'exposicio
# d'entrenament quedi mes equilibrada encara que la seva força relativa
# no arribi a ser identica.
ESTILS = {
    'conservador': dict(truc_agressio=1.00, envit_agressio=0.90, farol_prob=0.10, resposta_truc=0.98),
    'equilibrat':  dict(truc_agressio=1.20, envit_agressio=1.15, farol_prob=0.16, resposta_truc=0.99),
    'agressiu':    dict(truc_agressio=1.45, envit_agressio=1.35, farol_prob=0.20, resposta_truc=1.00),
    'farol':       dict(truc_agressio=1.25, envit_agressio=1.20, farol_prob=0.24, resposta_truc=1.00),
}


class AgentRegles:
    use_raw = False

    def __init__(self, num_actions=19, seed=None,
                 estil: str | None = None,
                 truc_agressio: float | None = None,
                 envit_agressio: float | None = None,
                 farol_prob: float | None = None,
                 resposta_truc: float | None = None,
                 marcador_simulat: tuple[int, int] | None = None):
        self.num_actions = num_actions
        self.rng = random.Random(seed)

        base = ESTILS[estil] if estil is not None else ESTILS['equilibrat']
        self.truc_agressio = float(truc_agressio if truc_agressio is not None else base['truc_agressio'])
        self.envit_agressio = float(envit_agressio if envit_agressio is not None else base['envit_agressio'])
        self.farol_prob = float(farol_prob if farol_prob is not None else base['farol_prob'])
        self.resposta_truc = float(resposta_truc if resposta_truc is not None else base['resposta_truc'])

        # Marcador simulat (propi, rival): com que cada mà d'entrenament es
        # un episodi que sempre reinicia el marcador real a 0-0, la lògica
        # adaptativa (agressiu si guanyant/perdent) mai s'exercita durant
        # l'entrenament. Si es dona, substitueix el marcador real només per
        # als càlculs interns d'aquest agent (útil per a l'OpponentPool).
        self.marcador_simulat = marcador_simulat

    def set_marcador_simulat(self, propi: int, rival: int) -> None:
        self.marcador_simulat = (propi, rival)

    def step(self, state):
        action, _ = self.eval_step(state)
        return action

    def eval_step(self, state):
        raw = state['raw_obs']
        legal = set(state['legal_actions'].keys())

        if raw['response_state_val'] == 2:
            action = self._respondre_envit(raw, legal)
        elif raw['response_state_val'] == 1:
            action = self._respondre_truc(raw, legal)
        else:
            action = self._torn_normal(raw, legal)

        if action not in legal:
            action = self._fallback(legal)
        return action, {}

    def _hand(self, raw):
        return raw['ma_jugador']

    def _forces(self, raw):
        return [TrucJudger.get_forca_carta(c) for c in self._hand(raw)]

    def _n_top(self, raw):
        return sum(1 for f in self._forces(raw) if f >= FORCA_TOP)

    def _best_force(self, raw):
        forces = self._forces(raw)
        return max(forces) if forces else 0

    def _hand_strength(self, raw):
        forces = self._forces(raw)
        return sum(forces) / 319.0 if forces else 0.0

    def _envit_score(self, raw):
        return TrucJudger.get_envit_ma(self._hand(raw))

    def _rondes_info(self, raw):
        pid = raw['id_jugador']
        equip = pid % 2
        guanyades = 0
        perdudes = 0
        for w in raw['ronda_winners']:
            if w is None or w == -1:
                continue
            if w % 2 == equip:
                guanyades += 1
            else:
                perdudes += 1
        return guanyades, perdudes

    def _rival_carta_taula(self, raw):
        """Retorna la forca de la carta rival a la taula, o None."""
        pid = raw['id_jugador']
        for p, carta in raw.get('cartes_taula_actual', []):
            if p % 2 != pid % 2:
                return TrucJudger.get_forca_carta(carta)
        return None

    def _som_ma(self, raw):
        return raw['ma'] == raw['id_jugador']

    def _marcador_efectiu(self, raw):
        """Retorna (propi, rival). Fa servir el marcador simulat si se n'ha
        donat un (entrenament per mans); si no, el marcador real de l'estat
        (partida persistent, p.ex. GUI)."""
        if self.marcador_simulat is not None:
            return self.marcador_simulat
        equip = raw['id_jugador'] % 2
        return raw['puntuacio'][equip], raw['puntuacio'][1 - equip]

    def _avantatge_puntuacio(self, raw):
        """Retorna diferencia de puntuacio (positiu = anem per davant)."""
        propi, rival = self._marcador_efectiu(raw)
        return propi - rival

    def _score_context(self, raw):
        """Retorna (per_guanyar_propi, per_guanyar_rival) en punts."""
        propi, rival = self._marcador_efectiu(raw)
        pf = raw.get('puntuacio_final', 24)
        per_propi = max(1, pf - propi)
        per_rival = max(1, pf - rival)
        return per_propi, per_rival

    def _fallback(self, legal):
        if PASSAR in legal:
            return PASSAR
        for a in [PLAY_0, PLAY_1, PLAY_2]:
            if a in legal:
                return a
        return min(legal)

    # respondre envit
    def _respondre_envit(self, raw, legal):
        es = self._envit_score(raw)
        per_propi, per_rival = self._score_context(raw)
        envit_level = raw['estat_envit']['level']

        # % que representa l'envit dels punts que li falten a cada equip
        perill_rival = es / per_rival   # > 1: rival guanya el joc si guanya l'envit
        necessitat   = es / per_propi   # > 1: nosaltres guanyem si guanyem l'envit

        llindar_puja    = 30 / self.envit_agressio
        llindar_accepta = 25 / self.envit_agressio

        # Rival a punt de guanyar -> ser conservadors (no regalar punts)
        if perill_rival >= 1.0:
            llindar_puja    += 8
            llindar_accepta += 8
        elif perill_rival >= 0.6:
            llindar_puja    += 4
            llindar_accepta += 4

        # Nosaltres a punt de guanyar -> ser agressius
        if necessitat >= 1.0:
            llindar_puja    -= 10
            llindar_accepta -= 10
        elif necessitat >= 0.6:
            llindar_puja    -= 5
            llindar_accepta -= 5

        # Soroll estocàstic (±2)
        llindar_puja    += self.rng.randint(-2, 2)
        llindar_accepta += self.rng.randint(-2, 2)

        # Lògica estricta per nivells alts d'envit 
        if envit_level >= 4:
            llindar_puja = max(34 - (self.envit_agressio - 1.0) * 2, 32)
            llindar_accepta = max(32 - (self.envit_agressio - 1.0) * 2, 30)

        if es >= llindar_puja and APOSTAR_ENVIT in legal:
            return APOSTAR_ENVIT
        if es >= llindar_accepta and VULL_ENVIT in legal:
            return VULL_ENVIT
        if FORA_ENVIT in legal:
            return FORA_ENVIT
        return self._fallback(legal)

    # respondre truc
    def _respondre_truc(self, raw, legal):
        guanyades, perdudes = self._rondes_info(raw)
        n_top = self._n_top(raw)
        best = self._best_force(raw)
        n_cartes = len(self._hand(raw))
        truc_level = raw['estat_truc']['level']
        per_propi, per_rival = self._score_context(raw)

        punts_en_joc = truc_level
        perill_rival = punts_en_joc / per_rival   # % dels punts rival que representa el truc
        necessitat   = punts_en_joc / per_propi   # % dels nostres punts necessaris

        # Quantes pujades han passat (0=truc inicial, 1=retruc, 2=vale9...)
        n_pujades = max(0, (truc_level - 3) // 3)

        # Ajust de força: negatiu = acceptem amb menys força; positiu = exigim més
        ajust_forca = 0

        # Cada pujada de l'oponent senyala força -> exigim més mà per acceptar
        ajust_forca += n_pujades * 10

        if perill_rival >= 1.0:
            # Foldant els donem la victòria igualment -> lluitar val la pena
            ajust_forca -= 10
        elif perill_rival >= 0.6:
            # Truc molt perillós per al nostre marcador -> més conservadors
            ajust_forca += 10

        if necessitat >= 1.0:
            # Guanyar el truc ens dona la victòria -> molt agressius
            ajust_forca -= 20
        elif necessitat >= 0.6:
            ajust_forca -= 10

        # Soroll estocàstic (±5)
        ajust_forca += self.rng.randint(-5, 5)

        # Ajust per paràmetre de comportament: >1 accepta més; <1 exigeix més
        ajust_forca -= 15 * (self.resposta_truc - 1.0)

        # Ja hem guanyat 1 ronda
        if guanyades >= 1:
            if best >= 97 and APOSTAR_TRUC in legal:
                return APOSTAR_TRUC
            if best >= 70 + ajust_forca and VULL_TRUC in legal:
                return VULL_TRUC
            if self.rng.random() < 0.20 and VULL_TRUC in legal:
                return VULL_TRUC
            if FORA_TRUC in legal:
                return FORA_TRUC

        # Ronda 0 (cap ronda jugada)
        if guanyades == 0 and perdudes == 0:
            if n_top >= 2 and APOSTAR_TRUC in legal:
                return APOSTAR_TRUC
            if n_top >= 2 and VULL_TRUC in legal:
                return VULL_TRUC
            if n_top == 1 and best >= 97 and VULL_TRUC in legal:
                return VULL_TRUC
            if n_top == 1 and self.rng.random() < 0.20 and VULL_TRUC in legal:
                return VULL_TRUC
            if FORA_TRUC in legal:
                return FORA_TRUC

        # Ja hem perdut 1 ronda -> necessitem guanyar les 2 restants
        if perdudes >= 1:
            # Si estem desesperats (necessitat alta), abaixem requisit de top cards
            n_top_needed = 1 if necessitat >= 0.5 else 2
            if n_top >= n_top_needed and n_cartes >= 2 and VULL_TRUC in legal:
                return VULL_TRUC
            if FORA_TRUC in legal:
                return FORA_TRUC

        return self._fallback(legal)

    # torn normal
    def _torn_normal(self, raw, legal):
        ronda = raw['comptador_ronda']
        guanyades, perdudes = self._rondes_info(raw)

        # Considerar envit
        if ronda == 0 and APOSTAR_ENVIT in legal:
            es = self._envit_score(raw)
            envit_level = raw['estat_envit']['level']
            if envit_level == 0:
                # Llindars variables per fer-lo menys predictible
                llindar_alt = self.rng.randint(28, 30)
                llindar_baix = self.rng.randint(24, 26)

                if es >= llindar_alt:
                    return APOSTAR_ENVIT
                if es >= llindar_baix and self._som_ma(raw) and self.rng.random() < 0.50:
                    return APOSTAR_ENVIT
                # Zona grisa: aleatorietat
                if llindar_baix <= es < llindar_alt and self.rng.random() < 0.35:
                    return APOSTAR_ENVIT

        # Considerar truc
        if APOSTAR_TRUC in legal:
            action_truc = self._considerar_truc(raw, legal, guanyades, perdudes)
            if action_truc is not None:
                return action_truc

        # Jugar carta
        return self._escollir_carta(raw, legal, ronda, guanyades, perdudes)

    # considerar truc
    def _considerar_truc(self, raw, legal, guanyades, perdudes):
        n_top = self._n_top(raw)
        n_cartes = len(self._hand(raw))
        rival_visible = self._rival_carta_taula(raw) is not None
        avantatge = self._avantatge_puntuacio(raw)

        if avantatge > 6:
            return None

        if guanyades >= 1 and n_top >= 1:
            if self.rng.random() < self.truc_agressio:
                return APOSTAR_TRUC

        # Bluff (parametritzat)
        if guanyades >= 1 and self.rng.random() < self.farol_prob:
            return APOSTAR_TRUC

        # Altres Casos
        if guanyades == 0 and perdudes == 0 and not rival_visible and n_top == 3:
            return APOSTAR_TRUC

        if guanyades == 0 and perdudes == 0 and rival_visible and n_top >= 2:
            if self.rng.random() < self.truc_agressio:
                return APOSTAR_TRUC

        if perdudes >= 1 and n_top >= 2 and n_cartes >= 2 and self.rng.random() < 0.40 * self.truc_agressio:
            return APOSTAR_TRUC

        # Mes agressiu si anem per darrere
        if avantatge < -6 and n_top >= 1 and guanyades >= 1:
            return APOSTAR_TRUC

        # Farol extra quan truc_agressio > 1
        if self.truc_agressio > 1.0 and guanyades == 0 and perdudes == 0:
            if self.rng.random() < self.farol_prob * (self.truc_agressio - 1.0):
                return APOSTAR_TRUC

        return None

    # escollir carta
    def _escollir_carta(self, raw, legal, ronda, guanyades, perdudes):
        hand = self._hand(raw)
        forces = self._forces(raw)
        n_cartes = len(hand)

        # Nomes una carta -> jugar-la
        if n_cartes == 1 or (PLAY_1 not in legal and PLAY_2 not in legal):
            return PLAY_0

        rival_forca = self._rival_carta_taula(raw)

        # Rival ha jugat -> intentar guanyar barat o sacrificar
        if rival_forca is not None:
            return self._guanyar_barat(forces, rival_forca, legal)

        # Juguem primers
        if ronda == 0:
            n_top = self._n_top(raw)
            # Aleatorietat: de vegades jugar forta a ronda 0
            if self.rng.random() < 0.15:
                return PLAY_0 if PLAY_0 in legal else PLAY_1
            # 2+ top -> forta primer
            if n_top >= 2:
                return PLAY_0
            # Si no -> del mig
            if PLAY_1 in legal:
                return PLAY_1
            return PLAY_0

        if ronda == 1:
            if guanyades >= 1:
                return self._carta_mes_feble(legal)
            else:
                return PLAY_0

        # Ronda 2
        return PLAY_0

    def _guanyar_barat(self, forces, rival_forca, legal):
        """Juga la carta mes feble que guanya, o la mes feble si no pot guanyar."""
        n = len(forces)
        play_actions = [PLAY_0, PLAY_1, PLAY_2][:n]

        guanyadores = []
        for i in range(n - 1, -1, -1):
            if play_actions[i] in legal and forces[i] > rival_forca:
                guanyadores.append(play_actions[i])

        if guanyadores:
            return guanyadores[0]  # La mes feble que guanya

        # No pot guanyar
        return self._carta_mes_feble(legal)

    def _carta_mes_feble(self, legal):
        for a in [PLAY_2, PLAY_1, PLAY_0]:
            if a in legal:
                return a
        return PLAY_0
