import numpy as np
from collections import OrderedDict
from rlcard.envs import Env
from entorn.game import TrucGame
from entorn.cartes_accions import ACTION_SPACE, ACTION_LIST, ACTIONS_SIGNAL, init_joc_cartes

class TrucEnv(Env):
    """
    Entorn del joc del Truc per a RLCard.
    Aquesta classe connecta la lògica del joc (TrucGame) amb la interfície estàndard d'entorns de RLCard.
    """
    def __init__(self, config):
        self.name = 'truc'
        self.num_jugadors = config.get('num_jugadors', 2)
        self.cartes_jugador = config.get('cartes_jugador', 3)
        self.puntuacio_final = config.get('puntuacio_final', 24)
        senyes = config.get('senyes', False)
        player_class = config.get('player_class', None)

        if player_class:
            self.game = TrucGame(num_jugadors=self.num_jugadors, 
                                 cartes_jugador=self.cartes_jugador, 
                                 senyes=senyes,
                                 puntuacio_final=self.puntuacio_final,
                                 player_class=player_class)
        else:
             self.game = TrucGame(num_jugadors=self.num_jugadors, 
                                 cartes_jugador=self.cartes_jugador, 
                                 senyes=senyes,
                                 puntuacio_final=self.puntuacio_final)
        
        config.setdefault('allow_step_back', False)
        config.setdefault('seed', None)
        super().__init__(config)
        
        self.cartes = init_joc_cartes()
        self.carta_map = {carta: i for i, carta in enumerate(self.cartes)}
        self.signal_map = {signal: i for i, signal in enumerate(ACTIONS_SIGNAL)}

        # espai d'estat
        self.num_cartes = len(self.cartes)
        self.espai_joc_cartes = self.num_cartes + 1 # carta buida
        self.espai_hist_cartes = self.num_jugadors * self.cartes_jugador 
        
        self.espai_senya = len(ACTIONS_SIGNAL) + 1
        if senyes:
            self.espai_hist_senyes = self.num_jugadors * self.cartes_jugador
        else:
            self.espai_hist_senyes = 0


        self.espai_info_publica = 10
        self.state_size = (self.cartes_jugador * self.espai_joc_cartes) + \
                          (self.espai_hist_cartes * self.espai_joc_cartes) + \
                          (self.espai_hist_senyes * self.espai_senya) + \
                          self.espai_info_publica
                          
        
        self.state_shape = [[self.state_size] for _ in range(self.num_jugadors)]
        self.action_shape = [[len(ACTION_LIST)] for _ in range(self.num_jugadors)]

    def _extract_state(self, state):
        obs = np.zeros(self.state_size, dtype=np.int8)
        idx = 0
        
        #mà
        ma_jugador = state['ma_jugador']

        for i in range(self.cartes_jugador):
            if i < len(ma_jugador):
                card_str = ma_jugador[i]
                if card_str in self.carta_map:
                    card_idx = self.carta_map[card_str]
                    obs[idx + card_idx] = 1
            else:
                obs[idx + self.num_cartes] = 1 # simbol per buit
            idx += self.espai_joc_cartes

        #Historials
        hist_cartes = state['hist_cartes']
        for i in range(self.espai_hist_cartes):
            if i < len(hist_cartes):
                _, card_str = hist_cartes[i] # (pid, card)
                if card_str in self.carta_map:
                    card_idx = self.carta_map[card_str]
                    obs[idx + card_idx] = 1
            else:
                obs[idx + self.num_cartes] = 1 # Slot buit
            idx += self.espai_joc_cartes
            
        if self.espai_hist_senyes > 0:
            signals = state.get('hist_senyes', [])
            for i in range(self.espai_hist_senyes):
                if i < len(signals):
                    _, signal_str = signals[i]
                    if signal_str in self.signal_map:
                        sig_idx = self.signal_map[signal_str]
                        obs[idx + sig_idx] = 1
                else:
                    obs[idx + len(ACTIONS_SIGNAL)] = 1 # Slot buit
                idx += self.espai_senya

        #Info pública
        obs[idx] = state['puntuacio'][0]; idx += 1
        obs[idx] = state['puntuacio'][1]; idx += 1
        
        #apostes
        obs[idx] = state['estat_truc']['level']; idx += 1
        obs[idx] = state['estat_truc']['owner'] + 1; idx += 1 # 0=Ningú, 1=P0, 2=P1
        
        obs[idx] = state['estat_envit']['level']; idx += 1
        obs[idx] = state['estat_envit']['owner'] + 1; idx += 1
        
        #situacio
        obs[idx] = state['fase_torn']; idx += 1
        obs[idx] = state['ma']; idx += 1
        obs[idx] = state['comptador_ronda']; idx += 1
        
        obs[idx] = state['id_jugador']; idx += 1

        #accions legals
        legal_actions_list = state['accions_legals']
        legal_actions = OrderedDict({a: None for a in legal_actions_list})
        
        # Construir l'objecte d'estat final per a l'agent
        extracted_state = {
            'obs': obs,
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': [ACTION_LIST[a] for a in legal_actions_list],
            'action_record': self.action_recorder
        }
        return extracted_state

    # Factor de les micro-recompenses intermèdies.
    MICRO_FACTOR = 0.3

    def get_payoffs(self):
        score    = self.game.score
        objectiu = self.game.puntuacio_final
        payoffs  = []

        for pid in range(self.num_jugadors):
            ha_guanyat   = score[pid] >= objectiu
            reward_final = 1.0 if ha_guanyat else -1.0
            reward_inter = (score[pid] / objectiu) * self.MICRO_FACTOR
            payoffs.append(reward_final + reward_inter)

        return payoffs

    def get_estat_taula(self, player_id):
        """
        Retorna l'estat brut del joc per mostrar la taula
        """
        return self.game.get_state(player_id)

    def _decode_action(self, action_id):
        return ACTION_LIST[action_id]

    def _get_legal_actions(self):
        return self.game.get_legal_actions()
