import sys
import os
import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm, trange
try:
    if '__file__' in globals():
        # Arribar a l'arrel del projecte (TFG-truc) que està 4 nivells amunt
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        sys.path.insert(0, root_path)
    else:
        sys.path.insert(0, os.getcwd())
except Exception:
    pass

import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader

from joc.entorn.cartes_accions import PALS, NUMS, ACTION_LIST
from joc.entorn.env import TrucEnv
from joc.entorn.rols.judger import TrucJudger
from RL.models.xarxa_truc import ModelPreEntrenament

# Constants
MIDA_DATASET    = 200_000
VAL_SPLIT       = 0.2
BATCH_SIZE      = 256
NUM_EPOCHS      = 100
LR              = 0.001
WEIGHT_DECAY    = 1e-4  # Regularització L2
PATIENCE        = 10    # Early Stopping
N_CARTES_MA     = 3       
ENVIT_MAX       = 24
MAX_FORCA_TRUC  = 330
N_ACCIONS       = len(ACTION_LIST)

# Carpetes de sortida
if '__file__' in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd() / "entrenament" / "entrenamentEstatTruc"

TIMESTAMP = datetime.now().strftime("%d_%m_%y_a_les_%H%M")
RUN_DIR   = BASE_DIR / "registres" / TIMESTAMP
MODEL_DIR = RUN_DIR / "models"
LOG_DIR   = RUN_DIR / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generar_dataset_estatic(num_mostres: int, env_base=None):
    """
    Genera un dataset de dades totalment coherents fent servir instàncies reals
    de l'entorn de joc (TrucEnv).
    """
    cartes_np  = np.zeros((num_mostres, 6, 4, 9), dtype=np.float32)
    context_np = np.zeros((num_mostres, 23), dtype=np.float32)
    labels_envit_np = np.zeros((num_mostres, 1), dtype=np.float32)
    labels_truc_np   = np.zeros((num_mostres, 1), dtype=np.float32)
    labels_accions_np= np.zeros((num_mostres, 19), dtype=np.float32)

    if env_base is None:
        env_base = TrucEnv(config={'num_jugadors': 2, 'cartes_jugador': 3, 'puntuacio_final': 12, 'senyes': True})

    mostres_recollides = 0
    pbar = tqdm(total=num_mostres, desc="Generant dataset sintètic (Augment complexitat)")
    
    while mostres_recollides < num_mostres:
        state, current_player = env_base.reset()
        
        # Juguem l'episodi i guardem TOTS els estats generats
        while not env_base.is_over():
            accions_possibles = list(state['legal_actions'].keys())
            if not accions_possibles:
                break
                
            obs_cartes   = state['obs']['obs_cartes']
            obs_context  = state['obs']['obs_context']
            
            # Etiquetes d'Accions Legals
            labels_acc = np.zeros(19, dtype=np.float32)
            for acc_id in accions_possibles:
                labels_acc[acc_id] = 1.0

            # Etiquetes Envido i Truc
            pid = current_player if current_player is not None else 0
            player_obj = env_base.game.players[pid]
            ma_inicial_strings = [c for c in player_obj.initial_hand]
            punts_envido = TrucJudger.get_envit_ma(ma_inicial_strings)
            forca = sum(TrucJudger.get_forca_carta(c) for c in ma_inicial_strings)
            
            if mostres_recollides < num_mostres:
                cartes_np[mostres_recollides] = obs_cartes
                context_np[mostres_recollides] = obs_context
                labels_envit_np[mostres_recollides, 0] = punts_envido / ENVIT_MAX
                labels_truc_np[mostres_recollides, 0]   = forca / MAX_FORCA_TRUC
                labels_accions_np[mostres_recollides]   = labels_acc
                
                mostres_recollides += 1
                pbar.update(1)
            else:
                break
            
            # Fer un pas amb una acció pseudo-aleatòria
            accions_no_cartes = [a for a in accions_possibles if a > 5]
            if random.random() < 0.2 and len(accions_no_cartes) > 0:
                acció_escollida = random.choice(accions_no_cartes)
            else:
                acció_escollida = random.choice(accions_possibles)
                
            state, current_player = env_base.step(acció_escollida)
            
    pbar.close()
    return (torch.tensor(cartes_np), 
            torch.tensor(context_np), 
            torch.tensor(labels_envit_np), 
            torch.tensor(labels_truc_np),
            torch.tensor(labels_accions_np))

# Bucle d'entrenament principal
def entrenar():
    """Entrena el ModelPreEntrenament amb Validació (80/20), Early Stopping i L2."""

    print(f"Dispositiu: {DEVICE}")
    print(f"Iniciant pre-entrenament: {NUM_EPOCHS} epocs, "
          f"dataset={MIDA_DATASET}, batch={BATCH_SIZE}, lr={LR}, wd={WEIGHT_DECAY}\n")

    # Validation Split
    env_generador = TrucEnv(config={'num_jugadors': 2, 'cartes_jugador': 3, 'puntuacio_final': 12, 'senyes': True})
    t_cartes, t_context, t_lenv, t_ltruc, t_lacc = generar_dataset_estatic(MIDA_DATASET, env_generador)
    
    dataset_complet = TensorDataset(t_cartes, t_context, t_lenv, t_ltruc, t_lacc)
    
    n_val = int(MIDA_DATASET * VAL_SPLIT)
    n_train = MIDA_DATASET - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_complet, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset split completat: Train={n_train}, Val={n_val}")

    model = ModelPreEntrenament().to(DEVICE)
    optimitzador = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimitzador, mode='min', factor=0.5, patience=5, verbose=True)

    # Pesos per la Loss (Balanç de caps)
    W_ENVIT = 1.0
    W_TRUC  = 1.0
    W_ACC   = 2.0  # Li donem més pes a aprendre les accions legals

    log_path = LOG_DIR / "preentrenament_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoca", "train_loss", "val_loss", "val_env", "val_truc", "val_acc"])

    # Early Stopping state
    best_val_loss = float('inf')
    best_model_path = MODEL_DIR / "best_pesos_cos_truc.pth"
    patience_counter = 0

    # Regeneration parameters
    REGENERATE_EVERY_N_EPOCHS = 20

    for epoca in range(1, NUM_EPOCHS + 1):
        # Regenerar el dataset si toca
        if epoca > 1 and (epoca - 1) % REGENERATE_EVERY_N_EPOCHS == 0:
            print(f"\n[INFO] Època {epoca}: Regenerant dades de preentrenament de forma dinàmica...")
            t_cartes, t_context, t_lenv, t_ltruc, t_lacc = generar_dataset_estatic(MIDA_DATASET, env_generador)
            dataset_complet = TensorDataset(t_cartes, t_context, t_lenv, t_ltruc, t_lacc)
            train_dataset, val_dataset = torch.utils.data.random_split(dataset_complet, [n_train, n_val])
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            print("[INFO] Noves dades carregades amb èxit!\n")

        model.train()
        train_loss_total = 0.0
        
        pbar_train = tqdm(train_loader, desc=f"Epoca {epoca}/{NUM_EPOCHS} [Train]", leave=False)
        for cartes, context, labels_env, labels_truc, labels_acc in pbar_train:
            cartes, context = cartes.to(DEVICE), context.to(DEVICE)
            labels_env, labels_truc, labels_acc = labels_env.to(DEVICE), labels_truc.to(DEVICE), labels_acc.to(DEVICE)
            
            optimitzador.zero_grad()
            pred_env, pred_truc, pred_acc = model(cartes, context)
            
            loss_env  = criteri_mse(pred_env, labels_env)
            loss_truc = criteri_mse(pred_truc, labels_truc)
            loss_acc  = criteri_bce(pred_acc, labels_acc)
            loss_total = (W_ENVIT * loss_env) + (W_TRUC * loss_truc) + (W_ACC * loss_acc)
            
            loss_total.backward()
            optimitzador.step()
            
            train_loss_total += loss_total.item() * cartes.size(0)
            pbar_train.set_postfix(loss=f"{loss_total.item():.4f}")
            
        train_loss_avg = train_loss_total / n_train
        
        # Validació
        model.eval()
        val_loss_total = 0.0
        v_env_tot, v_truc_tot, v_acc_tot = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for cartes, context, labels_env, labels_truc, labels_acc in val_loader:
                cartes, context = cartes.to(DEVICE), context.to(DEVICE)
                labels_env, labels_truc, labels_acc = labels_env.to(DEVICE), labels_truc.to(DEVICE), labels_acc.to(DEVICE)
                
                pred_env, pred_truc, pred_acc = model(cartes, context)
                
                loss_env  = criteri_mse(pred_env, labels_env)
                loss_truc = criteri_mse(pred_truc, labels_truc)
                loss_acc  = criteri_bce(pred_acc, labels_acc)
                loss_total = (W_ENVIT * loss_env) + (W_TRUC * loss_truc) + (W_ACC * loss_acc)
                
                val_loss_total += loss_total.item() * cartes.size(0)
                v_env_tot += loss_env.item() * cartes.size(0)
                v_truc_tot += loss_truc.item() * cartes.size(0)
                v_acc_tot += loss_acc.item() * cartes.size(0)
                
        val_loss_avg = val_loss_total / n_val
        val_env_avg  = v_env_tot / n_val
        val_truc_avg = v_truc_tot / n_val
        val_acc_avg  = v_acc_tot / n_val
        
        # Actualització del Scheduler
        scheduler.step(val_loss_avg)
        lr_actual = optimitzador.param_groups[0]['lr']

        # Guardem els logs
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoca, train_loss_avg, val_loss_avg, val_env_avg, val_truc_avg, val_acc_avg, lr_actual])
        
        PRINT_CADA = NUM_EPOCHS // 20
        
        if epoca % PRINT_CADA == 0 or epoca == 1 or epoca == NUM_EPOCHS:
            print(f"Epoca {epoca:03d}/{NUM_EPOCHS} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} "
                  f"(Env: {val_env_avg:.4f}, Truc: {val_truc_avg:.4f}, Acc: {val_acc_avg:.4f})")
              
        # Early Stopping logic
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.cos.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"  -> No millora. Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"\nEarly Stopping actvat a l'epoca {epoca}. Millor Val Loss: {best_val_loss:.4f}.")
            print(f"Els millors pesos estan a: {best_model_path}")
            break

    if patience_counter < PATIENCE:
        print(f"\nEntrenament finalitzat ({NUM_EPOCHS} epocs). Millor Val Loss: {best_val_loss:.4f}")
        print(f"Els millors pesos estan a: {best_model_path}")


if __name__ == "__main__":
    entrenar()
