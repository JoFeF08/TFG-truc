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


class AgentRegles:
    use_raw = False

    def __init__(self, num_actions=19, seed=None):
        self.num_actions = num_actions
        self.rng = random.Random(seed)

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

    def _avantatge_puntuacio(self, raw):
        """Retorna diferencia de puntuacio (positiu = anem per davant)."""
        equip = raw['id_jugador'] % 2
        return raw['puntuacio'][equip] - raw['puntuacio'][1 - equip]

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

        if es >= 30 and APOSTAR_ENVIT in legal:
            return APOSTAR_ENVIT
        if es >= 25 and VULL_ENVIT in legal:
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

        # Penalitzacio per nivells alts
        extra = max(0, (truc_level - 3)) // 3  # 0 per level<=3, 1 per 6, 2 per 9+

        # Ja hem guanyat 1 ronda
        if guanyades >= 1:
            if best >= 97 and APOSTAR_TRUC in legal:
                return APOSTAR_TRUC
            if best >= 70 - extra * 15 and VULL_TRUC in legal:
                return VULL_TRUC
            # Acceptar arriscat amb un poc d'aleatorietat
            if self.rng.random() < 0.20 and VULL_TRUC in legal:
                return VULL_TRUC
            if FORA_TRUC in legal:
                return FORA_TRUC

        # Ronda 0 (cap ronda jugada)
        if guanyades == 0 and perdudes == 0:
            if n_top >= 2 + extra and APOSTAR_TRUC in legal:
                return APOSTAR_TRUC
            if n_top >= 2 and VULL_TRUC in legal:
                return VULL_TRUC
            if n_top == 1 and best >= 97 and VULL_TRUC in legal:
                return VULL_TRUC
            # Acceptar arriscat
            if n_top == 1 and self.rng.random() < 0.20 and VULL_TRUC in legal:
                return VULL_TRUC
            if FORA_TRUC in legal:
                return FORA_TRUC

        # Ja hem perdut 1 ronda -> necessitem guanyar les 2 restants
        if perdudes >= 1:
            if n_top >= 2 and n_cartes >= 2 and VULL_TRUC in legal:
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
                if es >= 28:
                    return APOSTAR_ENVIT
                if es >= 24 and self._som_ma(raw) and self.rng.random() < 0.50:
                    return APOSTAR_ENVIT
                # Zona grisa: aleatorietat
                if 24 <= es < 28 and self.rng.random() < 0.35:
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
        truc_level = raw['estat_truc']['level']
        avantatge = self._avantatge_puntuacio(raw)

        if avantatge > 6:
            return None

        if guanyades >= 1 and n_top >= 1:
            return APOSTAR_TRUC

        # Bluff
        if guanyades >= 1 and self.rng.random() < 0.12:
            return APOSTAR_TRUC

        # Altres Casos
        if guanyades == 0 and perdudes == 0 and not rival_visible and n_top == 3:
            return APOSTAR_TRUC

        if guanyades == 0 and perdudes == 0 and rival_visible and n_top >= 2:
            return APOSTAR_TRUC

        if perdudes >= 1 and n_top >= 2 and n_cartes >= 2 and self.rng.random() < 0.40:
            return APOSTAR_TRUC

        # Mes agressiu si anem per darrere
        if avantatge < -6 and n_top >= 1 and guanyades >= 1:
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
