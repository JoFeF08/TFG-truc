import os
import numpy as np

# Definicions de Cartes
# Pals: S=Espases, C=Copes, O=Ors, B=Bastos
# Rangs: 1, 3, 4, 5, 6, 7, 10, 11, 12 (S'eliminen els 8, 9 i els 2).
# Mapeig de Pals:
# S -> Espases
# C -> Copes
# O -> Ors
# B -> Bastos

PALS = ['S', 'C', 'O', 'B']
NUMS = ['1', '3', '4', '5', '6', '7', '10', '11', '12']

JOC_CARTES = {
    'S': ['1', '3', '4', '5', '6', '7', '10', '11', '12'],
    'O': ['1', '3', '4', '5', '6', '7', '10', '11', '12'],
    'C': ['1', '3', '4', '5', '6', '7', '10', '11', '12'],
    'B': ['1', '3', '4', '5', '6', '7', '10', '11', '12']
}


# Accions del Joc
# 0-2: Jugar carta (índex de la mà)
# 3: Apostar Envit (Envit, Torna-hi)
# 4: Apostar Truc (Truc, Retruc, Val Nou, Joc Fora)
# 5: Vull Envit (Acceptar aposta envit)
# 6: Vull Truc (Acceptar aposta truc)
# 7: Fora Envit (Rebutjar aposta envit - continua el joc)
# 8: Fora Truc (Rebutjar aposta truc / Retirar-se - fi ronda)
# 9: Passar (Saltar fase d'aposta o senyals)
# 10-17: Senyes

# Subconjunts d'accions
ACTIONS_PLAY = [
    'play_card_0',
    'play_card_1',
    'play_card_2'
]

ACTIONS_BET = [
    'apostar_envit',
    'apostar_truc',
    'vull_envit',
    'vull_truc',
    'fora_envit',
    'fora_truc',
    'passar'
]

ACTIONS_SIGNAL = [
    'senya_onze_bastos',
    'senya_deu_ors',
    'senya_as_espases',
    'senya_as_bastos',
    'senya_manilla_espases',
    'senya_manilla_ors',
    'senya_tres',
    'senya_as_bord',
    'senya_cegas'
]

ACTION_LIST = ACTIONS_PLAY + ACTIONS_BET + ACTIONS_SIGNAL

ACTION_SPACE = {action: i for i, action in enumerate(ACTION_LIST)}

def init_joc_cartes():
    deck = []
    for suit, ranks in JOC_CARTES.items():
        for rank in ranks:
            deck.append(suit + rank)
    return deck



