import sys
import os
import numpy as np

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

from joc.entorn.rols.player import TrucPlayer
from joc.entorn.rols.dealer import TrucDealer
from joc.entorn.rols.judger import TrucJudger
from joc.entorn.cartes_accions import ACTION_SPACE, ACTION_LIST
from enum import Enum


class ResponseState(Enum):
    NO_PENDING = 0
    TRUC_PENDING = 1
    ENVIT_PENDING = 2


class TrucGame:
    def __init__(self, num_jugadors=2, cartes_jugador=3, senyes=False, puntuacio_final=24, player_class=TrucPlayer, verbose=False):
        self.num_jugadors = num_jugadors
        self.cartes_jugador = cartes_jugador
        self.senyes = senyes
        self.puntuacio_final = puntuacio_final
        self.player_class = player_class
        self.verbose = verbose

        self.np_random = np.random.RandomState()
        self.payoffs = [0] * self.num_jugadors
        

    def debug_print(self, *args, **kwargs):
        """Imprimeix missatges només si self.verbose està activat"""
        if self.verbose:
            print(*args, **kwargs)
        

    def init_game(self):
        self.payoffs = [0] * self.num_jugadors
        
        self.players = []
        for i in range(self.num_jugadors):
            if isinstance(self.player_class, dict):
                p_class = self.player_class.get(i, TrucPlayer)
            else:
                p_class = self.player_class
                
            self.players.append(p_class(i, self.np_random))

        self.dealer = TrucDealer(self.np_random, n_cartes=self.cartes_jugador)
        self.judger = TrucJudger(self.np_random, n_cartes=self.cartes_jugador)

        #repartir
        self.dealer.shuffle()
        for player in self.players:
            self.dealer.deal_cards(player)
            player.hand.sort(key=lambda c: TrucJudger.get_forca_carta(c), reverse=True)
            player.initial_hand = list(player.hand)

        # Estat del Joc
        self.ma = 0
        self.current_player = self.ma
        self.turn_player = self.ma

        self.turn_phase = 0 if self.senyes else 1  # 0 (Senyals), 1 (Joc/Apostes)
        self.response_state = ResponseState.NO_PENDING # Estat de resposta actual

        self.comptador_ma = 1
        self.hist_cartes = []
        self.hist_cartes_ant = []
        self.hist_senyes = []
        
        # Guanyadors recents
        self.ultim_guanyador_envit = None
        self.ultim_guanyador_truc = None

        self.score = [0, 0] # Puntuació global
        
        # Estat del Truc
        self.truc_level = 1 
        self.truc_owner = -1 
        self.previous_truc_level = 1 
        
        # Estat de l'Envit
        self.envit_level = 0 
        self.envit_owner = -1
        
        self.envit_accepted = False 
        self.previous_envit_level = 1 
        self.envit_is_falta = False
        

        self.ronda_winners = [] # -1 per empat
        self.reward_intermedis = [0.0, 0.0]
        self.round_counter = 0
        self.cartes_ronda = []

        self.punts_envit_pendents = None 
        
        return self._get_return_state()

    def step(self, action):
        self.ultim_guanyador_envit = None
        self.ultim_guanyador_truc = None
        
        # Recompenses per accio
        self.reward_intermedis = [0.0, 0.0]
        
        action_str = ACTION_LIST[action] if isinstance(action, int) else action
        player = self.players[self.current_player]
        
        # --- LÒGICA DE RESPOSTA ---
        if self.response_state != ResponseState.NO_PENDING:
            
            # ENVIT
            if self.response_state == ResponseState.ENVIT_PENDING:
                
                if action_str == 'vull_envit':
                    self.envit_accepted = True
                    self.response_state = ResponseState.NO_PENDING
                    
                    # Reward intermedi de l'envit acceptat
                    mans_p = [p.initial_hand for p in self.players]
                    winner_team = self.judger.guanyador_envits(mans_p, self.ma)
                    pts_env = self.envit_level
                    self.reward_intermedis[winner_team] += (pts_env / 24.0)
                    self.reward_intermedis[1-winner_team] -= (pts_env / 24.0)

                    # Retornar el torn
                    self.current_player = self.turn_player
                    return self._get_return_state() 

                elif action_str == 'fora_envit':
                    points_won = self.previous_envit_level
                    winner_team = self.judger.get_equip((self.current_player + 1) % 2)
                    
                    self.reward_intermedis[winner_team] += 1.5 * (points_won / 24.0)   # guanyador: 1.5x (incentiu ofensiu)
                    self.reward_intermedis[1-winner_team] -= 1.0 * (points_won / 24.0)  # perdedor: 1.0x (retirada estratègica)

                    self.punts_envit_pendents = (winner_team, points_won, None)
                    self.ultim_guanyador_envit = (winner_team, points_won, None)

                    self.envit_accepted = False 
                    self.envit_owner = -1 
                    
                    # Retornar el torn
                    self.response_state = ResponseState.NO_PENDING
                    self.current_player = self.turn_player
                    self.turn_phase = 1 
                    return self._get_return_state()


            # TRUC
            elif self.response_state == ResponseState.TRUC_PENDING:
                if action_str == 'vull_truc':
                    self.response_state = ResponseState.NO_PENDING
                    
                    # Sense bonus per acceptar Truc — el reward ve del resultat

                    # Retornar el torn
                    self.current_player = self.turn_player
                    self.turn_phase = 1 
                    return self._get_return_state()
                
                elif action_str == 'fora_truc':
                    # Resoldre envit
                    self._resoldre_envit_si_pendent()

                    # Sumar punts pendents d'envit al tancar el truc
                    if self.punts_envit_pendents:
                        eq, pts, _ = self.punts_envit_pendents
                        self.score[eq] += pts
                        self.punts_envit_pendents = None

                    winner_team = self.judger.get_equip((self.current_player + 1) % 2)
                    pts_truc = self.previous_truc_level
                    self.score[winner_team] += pts_truc
                    self.ultim_guanyador_truc = (winner_team, pts_truc)
                    
                    # Reward intermedi de truc (fora - RESPOSTA)
                    self.reward_intermedis[winner_team] += 1.5 * (pts_truc / 24.0)   # qui va forçar el fora: 1.5x
                    self.reward_intermedis[1-winner_team] -= 1.0 * (pts_truc / 24.0)  # qui fa fora: 1.0x (retirada estratègica)

                    if max(self.score) >= self.puntuacio_final:
                        return self.get_state(self.current_player), None

                    self.response_state = ResponseState.NO_PENDING
                    self.hist_cartes_ant = list(self.hist_cartes)
                    self._reset_hand_state()
                    return self._get_return_state()

        
        # --- LÒGICA DE TORN NORMAL ---
        if action_str.startswith('senya_'):
            self.hist_senyes.append((self.current_player, action_str))
            self.turn_phase = 1 # canviar de fase
            return self._get_return_state()
            
        elif action_str == 'passar':
            if self.turn_phase == 0:
                self.turn_phase = 1 # Senyal -> Joc i Apostes
            
            return self._get_return_state()
            
        elif action_str == 'apostar_envit':
            # Guardar anterior per si diuen NO
            if self.envit_level == 0:
                self.previous_envit_level = 1
            else:
                self.previous_envit_level = self.envit_level

            # Seqüencia: 2 -> 4 -> 6 -> Tots (Falta)
            next_level = 0
            
            if self.envit_level == 0:
                 next_level = 2 # Envit
            elif self.envit_level == 2:
                 next_level = 4 # Un més
            elif self.envit_level == 4:
                 next_level = 6 # Dos més
            elif self.envit_level >= 6:
                 # Tots (Falta Envit)
                 self.envit_is_falta = True
                 # Regla: Si cap parella > 12 -> Guanya Partida.
                 # Si alguna > 12 -> Punts que falten al que va guanyant per arribar a 24.
                 leader_score = max(self.score)
                 if leader_score > 12:
                     next_level = 24 - leader_score
                 else:
                     next_level = 24 
            
            self.envit_level = next_level     
            self.envit_owner = self.current_player
            self.response_state = ResponseState.ENVIT_PENDING
            
            # donar torn per respondre
            self.current_player = (self.current_player + 1) % self.num_jugadors
            return self._get_return_state()

        elif action_str == 'apostar_truc':
            # Seqüencia: 1 -> 3 (Truc) -> 6 (Retruc) -> 9 (Val Nou) -> 24 (Joc Fora)

            next_val = 3
            if self.truc_level == 1: next_val = 3
            elif self.truc_level == 3: next_val = 6
            elif self.truc_level == 6: next_val = 9 
            elif self.truc_level >= 9: next_val = 24 
            
            self.previous_truc_level = self.truc_level
            self.truc_level = next_val
            self.truc_owner = self.current_player
            self.response_state = ResponseState.TRUC_PENDING
            self.current_player = (self.current_player + 1) % self.num_jugadors
            return self._get_return_state()
        
        elif action_str == 'fora_truc':
            self._resoldre_envit_si_pendent()
        
            winner = (self.current_player + 1) % 2
            
            # Sumar envits pendents
            if self.punts_envit_pendents:
                eq, pts, _ = self.punts_envit_pendents
                self.score[eq] += pts
                self.punts_envit_pendents = None

            pts_truc = self.truc_level
            self.score[winner] += pts_truc
            self.ultim_guanyador_truc = (winner, pts_truc)
            
            # Reward intermedi de truc (fora - VOLUNTARI)
            self.reward_intermedis[winner] += 1.0 * (pts_truc / 24.0)    # oponent: benefici normal
            self.reward_intermedis[1-winner] -= 1.2 * (pts_truc / 24.0)  # penalització moderada (permet retirada estratègica)

            if max(self.score) >= self.puntuacio_final:
                return self.get_state(self.current_player), None
            
            self.hist_cartes_ant = list(self.hist_cartes)
            self._reset_hand_state()
            return self._get_return_state()

            
        elif action_str.startswith('play_card'):
            idx = int(action_str[-1])
            card_played = player.hand.pop(idx)
            self.cartes_ronda.append((self.current_player, card_played))
            self.hist_cartes.append((self.current_player, card_played))
            
            # Comprovar fi de ronda
            if len(self.cartes_ronda) == self.num_jugadors:
                winner = self.judger.guanyador_ronda(self.cartes_ronda)
                
                # Reward intermedi de ronda
                pes = self._pes_ronda(self.round_counter, self.ronda_winners)
                if winner is not None:
                    eq_w = winner % 2
                    self.reward_intermedis[eq_w] += 0.40 * pes   # guanyar ronda: fort
                    self.reward_intermedis[1-eq_w] -= 0.20 * pes  # perdre ronda: moderat

                if winner is not None:
                    self.turn_player = winner
                    self.ronda_winners.append(winner)
                    self.debug_print(f"=====>DEBUG: Ronda {self.round_counter + 1} acabada. Guanyador: Jugador {winner}")
                else:
                    # Empat
                    self.turn_player = self.ma
                    self.ronda_winners.append(-1)  # -1 indica empat
                    self.debug_print(f"=====>DEBUG: Ronda {self.round_counter + 1} acabada. EMPAT")

                self.cartes_ronda = []
                self.round_counter += 1

                # Comprovar aquesta mà tanca ronda
                winner_ma = self.judger.guanyador_ma(self.ronda_winners, self.ma)
                if winner_ma != -1:
                    # Resoldre envit
                    self._resoldre_envit_si_pendent()

                    if self.punts_envit_pendents:
                        eq_env, pts_env, _ = self.punts_envit_pendents
                        self.score[eq_env] += pts_env
                        self.punts_envit_pendents = None

                    self.debug_print(f"=====>DEBUG: Mà acabada. Guanyador equip: {winner_ma}. Punts guanyats: {self.truc_level}. Score abans: {self.score}")
                    self.score[winner_ma] += self.truc_level
                    self.ultim_guanyador_truc = (winner_ma, self.truc_level)
                    
                    # Reward intermedi de truc (guanyador per mà)
                    self.reward_intermedis[winner_ma] += 1.0 * (self.truc_level / 24.0)
                    self.reward_intermedis[1-winner_ma] -= 1.0 * (self.truc_level / 24.0)

                    self.debug_print(f"=====>DEBUG: Score després: {self.score}")

                    if max(self.score) >= self.puntuacio_final:
                        return self.get_state(self.current_player), None

                    self.hist_cartes_ant = list(self.hist_cartes)
                    self._reset_hand_state()
                    return self.get_state(self.current_player), self.current_player
            else:
                self.turn_player = (self.turn_player + 1) % self.num_jugadors

            self.current_player = self.turn_player
            self.turn_phase = 0 if self.senyes else 1

        return self._get_return_state()

    def _get_return_state(self):
        legal_actions = self.get_legal_actions()
        if len(legal_actions) == 1 and legal_actions[0] == ACTION_SPACE['passar']:
             return self.step('passar')
        
        return self.get_state(self.current_player), self.current_player

    def _resoldre_envit_si_pendent(self):
        if self.envit_accepted and self.punts_envit_pendents is None:
            all_hands = [p.initial_hand for p in self.players]
            
            punts_envit = []
            for hand in all_hands:
                punts = self.judger.get_envit_ma(hand)
                punts_envit.append(punts)
            
            winner_team = self.judger.guanyador_envits(all_hands, self.ma)
            self.debug_print(f"=====>DEBUG: Resolent Envit acceptat al final. Punts: J0={punts_envit[0]}, J1={punts_envit[1]}. Guanyador: Equip {winner_team}. Punts guanyats: {self.envit_level}.")
            self.punts_envit_pendents = (winner_team, self.envit_level, punts_envit)
            self.ultim_guanyador_envit = (winner_team, self.envit_level, punts_envit)

    def get_state(self, player_id):
        """
        Retorna l'estat complet del joc des de la perspectiva del jugador_id.
        Inclou mà, cartes públiques, puntuacions, estats d'apostes, etc.
        """
        player = self.players[player_id]
        state = {}

        # --- CONTEXT GENERAL ---
        state['id_jugador'] = player_id
        state['ma'] = self.ma
        state['comptador_ma'] = self.comptador_ma
        state['puntuacio'] = self.score

        # --- ESTAT DEL TORN ---
        state['comptador_ronda'] = self.round_counter
        state['fase_torn'] = self.turn_phase
        state['estat_resposta'] = self.response_state.value

        # --- APOSTES ---
        state['estat_truc'] = {
            'level': self.truc_level,
            'owner': self.truc_owner,
            'active': (self.truc_level > 1)
        }
        state['estat_envit'] = {
            'level': self.envit_level,
            'owner': self.envit_owner,
            'active': (self.envit_level > 0),
            'accepted': self.envit_accepted
        }

        # --- INFORMACIÓ DEL JUGADOR ---
        state['ma_jugador'] = [c for c in player.hand]
        state['mans_rivals'] = {p.player_id: list(p.hand) for p in self.players if p.player_id != player_id}
        state['accions_legals'] = self.get_legal_actions()

        state['ronda_winners'] = list(self.ronda_winners)
        state['envit_accepted'] = self.envit_accepted
        state['response_state_val'] = self.response_state.value
        
        # Reward intermedi pendents de llegir
        state['reward_intermedis'] = list(self.reward_intermedis)

        # --- HISTORIAL FILTRAT PER VISIBILITAT ---
        # Cartes visibles: cartes pròpies (totes) + cartes jugades dels rivals (totes)
        state['hist_cartes'] = list(self.hist_cartes)
        state['hist_cartes_ant'] = list(self.hist_cartes_ant)
        state['cartes_taula_actual'] = list(self.cartes_ronda)
        state['hist_senyes'] = self.hist_senyes

        return state

    def get_legal_actions(self):
        actions = []
        player = self.players[self.current_player]
        
        # --- MODE RESPOSTA ---
        if self.response_state != ResponseState.NO_PENDING:
            # --- RESPOSTA ENVIT ---
            if self.response_state == ResponseState.ENVIT_PENDING:
                actions.append(ACTION_SPACE['vull_envit'])
                actions.append(ACTION_SPACE['fora_envit'])
                
                if self.envit_level <= 6 and not self.envit_is_falta:
                    actions.append(ACTION_SPACE['apostar_envit']) # Re-apostar (Torna-hi)
                
                return actions

            # --- RESPOSTA TRUC ---
            if self.response_state == ResponseState.TRUC_PENDING:
                actions.append(ACTION_SPACE['vull_truc'])
                actions.append(ACTION_SPACE['fora_truc'])
                
                # Només pots pujar si no has arribat al màxim (Joc Fora / 24)
                if self.truc_level < 24:
                    actions.append(ACTION_SPACE['apostar_truc']) # Re-apostar (Retruc, etc)


                return actions
            
            return actions

        # --- TORN NORMAL ---
        # Fase 0: Senyals
        if self.turn_phase == 0:
            actions.append(ACTION_SPACE['passar']) 
            for act_str in ACTION_LIST:
                 if act_str.startswith('senya_'):
                     actions.append(ACTION_SPACE[act_str])
            return actions

        # Fase 1: Joc i Apostes
        if self.turn_phase == 1:            
            # Envit: Només si ningú ha cantat res encara i primera ronda
            if self.envit_level == 0 and self.round_counter == 0:
                 actions.append(ACTION_SPACE['apostar_envit'])
            
            # Truc: Si no som propietaris de l'aposta actual
            if self.truc_owner != self.current_player:    
                 actions.append(ACTION_SPACE['apostar_truc'])
            
            actions.append(ACTION_SPACE['fora_truc'])
            
            num_cards = len(player.hand)
            for i in range(num_cards):
                actions.append(ACTION_SPACE[f'play_card_{i}'])

            return actions


    def get_num_jugadors(self):
        return self.num_jugadors

    def get_num_players(self):
        return self.num_jugadors

    def get_num_actions(self):
        return len(ACTION_LIST)
    
    def get_player_id(self):
        return self.current_player

    def is_ma_over(self):
        return self.judger.guanyador_ma(self.ronda_winners, self.ma) != -1

    def is_over(self):
        return max(self.score) >= self.puntuacio_final

    def get_payoffs(self):
        """
        Retorna els payoffs (recompenses) finals: +10 per guanyar, -10 per perdre.
        """
        score = self.score
        payoffs = []

        for pid in range(self.num_jugadors):
            oponent = (pid + 1) % 2
            delta = score[pid] - score[oponent]
            
            if delta > 0:
                payoffs.append(1.0)
            elif delta < 0:
                payoffs.append(-1.0)
            else:
                payoffs.append(0.0)

        return payoffs


    def _reset_hand_state(self):
        # Avançar mà
        self.ma = (self.ma + 1) % self.num_jugadors
        self.comptador_ma += 1
        self.current_player = self.ma
        self.turn_player = self.ma
        
        # Repartir de nou
        self.dealer.shuffle()
        for player in self.players:
            player.hand = [] # Netejar mà vella
            self.dealer.deal_cards(player)
            player.hand.sort(key=lambda c: TrucJudger.get_forca_carta(c), reverse=True)
            player.initial_hand = list(player.hand)
            
        # Reset variables
        self.truc_level = 1
        self.truc_owner = -1
        self.previous_truc_level = 1
        self.envit_level = 0
        self.envit_owner = -1
        self.envit_accepted = False
        self.previous_envit_level = 1
        self.envit_is_falta = False
        
        self.round_counter = 0
        self.cartes_ronda = []
        self.ronda_winners = []
        self.hist_cartes = []
        self.hist_senyes = []
        self.punts_envit_pendents = None
        
        self.turn_phase = 0 if self.senyes else 1
        self.response_state = ResponseState.NO_PENDING

    def _pes_ronda(self, ronda_num, ronda_winners):
        """
        Dona un pes dinàmic de la ronda actual
        """
        n_cartes = self.cartes_jugador

        # Distribucions per 3 i 5 cartes
        if n_cartes == 3:
            pesos = [2.0, 1.0, 0.5]
        elif n_cartes == 5:
            pesos = [2.0, 1.0, 0.5, 0.3, 0.1]
        else:
            raise ValueError(f"_pes_ronda només suporta 3 o 5 cartes, rebut {n_cartes}")

        pes = pesos[ronda_num]

        rondes_equip_0 = 0
        rondes_equip_1 = 0

        for winner in ronda_winners:
            if winner != -1:  # Si no és empat
                equip = winner % 2
                if equip == 0:
                    rondes_equip_0 += 1
                else:
                    rondes_equip_1 += 1

        # Calcular rondes necessàries per guanyar
        rondes_necessaries = (n_cartes // 2)
        estas_obligat = rondes_equip_1 >= (rondes_necessaries)

        if estas_obligat:
            pes = min(1.0, pes * 1.5)

        return pes