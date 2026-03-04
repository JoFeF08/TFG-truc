#%%
import sys
import os
import csv
from pathlib import Path
from datetime import datetime
from tqdm import trange
# Afegim l'arrel de TFG-truc al sys.path perquè Jupyter pugui trobar els mòduls d' 'entorn' i 'models'
try:
    if '__file__' in globals():
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    else:
        # A Jupyter de la UIB, l'arrel del teu usuari ja és on hi ha el codi penjat
        # o dins d'alguna carpeta visible. Posem el directori actual al path.
        sys.path.insert(0, os.getcwd())
except Exception:
    pass

import torch
import torch.nn as nn
import numpy as np
import random

from entorn.cartes_accions import PALS, NUMS, ACTION_LIST
from entorn.env import TrucEnv
from entorn.rols.judger import TrucJudger
from models.xarxa_truc import ModelPreEntrenament

BATCH_SIZE      = 128
N_ITERACIONS    = 10_000
LR              = 0.001
PRINT_CADA      = 500
N_CARTES_MA     = 3       
ENVIT_MAX       = 24
MAX_FORCA_TRUC  = 330
N_ACCIONS       = len(ACTION_LIST)

# Carpetes de sortida
if '__file__' in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    # A Jupyter, agafem el directori relatiu des de l'arrel on sol iniciar-se
    BASE_DIR = Path.cwd() / "entrenament" / "entrenamentEstatTruc"

TIMESTAMP = datetime.now().strftime("%d_%m_%y_a_les_%H%M")
RUN_DIR   = BASE_DIR / "registres" / TIMESTAMP
MODEL_DIR = RUN_DIR / "models"
LOG_DIR   = RUN_DIR / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generador de Dades per Pre-entrenament
def generar_batch_envido(batch_size: int = BATCH_SIZE, env_base=None):
    """
    Genera un batch de dades totalment coherents fent servir instàncies reals
    de l'entorn de joc (TrucEnv). Això garanteix que les rondes, nombres de cartes,
    fases de torn i les accions legals tenen sentit estricte al 100%.
    
    Returns:
        tensor_cartes        : Tensor (batch, 6, 4, 9), float32
        tensor_context       : Tensor (batch, 17),      float32
        tensor_labels_envido : Tensor (batch, 1),       float32
        tensor_labels_truc   : Tensor (batch, 1),       float32
        tensor_labels_accions: Tensor (batch, 19),      float32
    """
    cartes_np  = np.zeros((batch_size, 6, 4, 9), dtype=np.float32)
    context_np = np.zeros((batch_size, 17), dtype=np.float32)
    labels_envido_np = np.zeros((batch_size, 1), dtype=np.float32)
    labels_truc_np   = np.zeros((batch_size, 1), dtype=np.float32)
    labels_accions_np= np.zeros((batch_size, 19), dtype=np.float32)

    # Creem un entorn de llançar partides si no existeix
    if env_base is None:
        env_base = TrucEnv(config={'num_jugadors': 2, 'cartes_jugador': 3, 'puntuacio_final': 12, 'senyes': True})

    for i in range(batch_size):
        state, current_player = env_base.reset()
        
        passos_random = random.randint(0, 20)
        for _ in range(passos_random):
            accions_possibles = list(state['legal_actions'].keys())
            if not accions_possibles or env_base.is_over():
                break
            
            # Agafem una acció aleatòria i avancem
            acció_escollida = random.choice(accions_possibles)
            last_player = current_player
            state, current_player = env_base.step(acció_escollida)
            if env_base.is_over():
                break
                
        # Extreure les dades de l'estat en el que s'ha quedat detingut
        obs_cartes   = state['obs']['obs_cartes']
        obs_context  = state['obs']['obs_context']
        
        cartes_np[i] = obs_cartes
        context_np[i] = obs_context
        
        # A) Etiquetes d'Accions Legals
        for acc_id in state['legal_actions'].keys():
            labels_accions_np[i, acc_id] = 1.0

        # B) Etiquetes d'Envit
        # Obtenim la llista real de cartes de la mà inicial guardada a game pel jugador actual
        # (Si el joc acaba, usem l'últim que va moure o ens quedem amb l'id relatiu a la simulació)
        pid = current_player if current_player is not None else (last_player if 'last_player' in locals() else 0)
        player_obj = env_base.game.players[pid]
        ma_inicial_strings = [c for c in player_obj.initial_hand]
        punts_envido = TrucJudger.get_envit_ma(ma_inicial_strings)
        labels_envido_np[i, 0] = punts_envido / ENVIT_MAX
        
        # C) Etiquetes de Força_Truc
        forca = sum(TrucJudger.get_forca_carta(c) for c in ma_inicial_strings)
        labels_truc_np[i, 0] = forca / MAX_FORCA_TRUC

    return (torch.tensor(cartes_np), 
            torch.tensor(context_np), 
            torch.tensor(labels_envido_np), 
            torch.tensor(labels_truc_np),
            torch.tensor(labels_accions_np))

# Bucle d'entrenament
def entrenar():
    """Entrena el ModelPreEntrenament durant N_ITERACIONS iteracions."""

    print(f"Dispositiu: {DEVICE}")
    print(f"Iniciant pre-entrenament: {N_ITERACIONS} iteracions, "
          f"batch={BATCH_SIZE}, lr={LR}\n")

    model     = ModelPreEntrenament().to(DEVICE)
    criteri_mse = nn.MSELoss()
    criteri_bce = nn.BCEWithLogitsLoss()
    optimitzador = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()

    log_path = LOG_DIR / "preentrenament_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["iteracio", "loss_total", "loss_env", "loss_truc", "loss_acc"])

    env_generador = TrucEnv(config={'num_jugadors': 2, 'cartes_jugador': 3, 'puntuacio_final': 12, 'senyes': True})

    bucle = trange(1, N_ITERACIONS + 1, desc="Pre-entrenant models")
    for iteracio in bucle:
        cartes, context, labels_env, labels_truc, labels_acc = generar_batch_envido(BATCH_SIZE, env_generador)
        cartes  = cartes.to(DEVICE)
        context = context.to(DEVICE)
        labels_env  = labels_env.to(DEVICE)
        labels_truc = labels_truc.to(DEVICE)
        labels_acc  = labels_acc.to(DEVICE)

        optimitzador.zero_grad()
        prediccions_env, prediccions_truc, prediccions_acc = model(cartes, context)
        
        # Pèrdues independents
        loss_env  = criteri_mse(prediccions_env, labels_env)
        loss_truc = criteri_mse(prediccions_truc, labels_truc)
        loss_acc  = criteri_bce(prediccions_acc, labels_acc)
        
        # Suma per al backward conjunt
        loss_total= loss_env + loss_truc + loss_acc
        
        loss_total.backward()
        optimitzador.step()

        if iteracio % PRINT_CADA == 0 or iteracio == 1:
            lv_tot  = loss_total.item()
            lv_env  = loss_env.item()
            lv_truc = loss_truc.item()
            lv_acc  = loss_acc.item()
            
            bucle.set_postfix(
                Total=f"{lv_tot:.3f}", 
                Env=f"{lv_env:.3f}", 
                Truc=f"{lv_truc:.3f}", 
                Acc=f"{lv_acc:.3f}"
            )
            
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([iteracio, lv_tot, lv_env, lv_truc, lv_acc])

    # Desar els pesos del cos
    output_path = MODEL_DIR / "pesos_cos_truc.pth"
    torch.save(model.cos.state_dict(), output_path)
    print(f"\nPesos del cos desats a: {output_path}")

if __name__ == "__main__":
    entrenar()
# %%
