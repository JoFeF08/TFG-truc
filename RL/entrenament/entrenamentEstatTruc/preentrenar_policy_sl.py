"""Warm-start supervisat (SL) del cap de cartes + tronc + cap de valor d'una
política MaskablePPO, per distil·lació d'`AgentProbabilistic`.

CLAU DE DISSENY (vegeu el pla): en comptes d'entrenar un model separat i
transferir només el tronc `.pth`, entrenem l'objectiu SL DIRECTAMENT sobre una
instància real de `MaskablePPO(MultiHeadMaskableACPolicy)`, tocant només:
  - `policy.features_extractor`   (tronc CNN+dens, CosMultiInputSB3)
  - `policy.mlp_extractor`        (MLP política + valor)
  - `policy.action_net.head_cartes` (els 3 logits play_card_i)
  - `policy.value_net`            (el crític)
i desem amb `model.save(zip)`. Així RL hi continua amb `--resume_from` SENSE cap
cirurgia de state_dict (les formes queden garantides pel policy_kwargs desat).

Els caps `head_truc`/`head_envit`/`head_residual` NO es toquen (queden a la
seva init ortogonal), a punt per aprendre les apostes en fases posteriors.

S'usa un optimitzador SEPARAT (no `policy.optimizer`) perquè els moments d'Adam
del SL no es filtrin al resume d'RL: `MaskablePPO.load` restaura `policy.optimizer`,
que aquí queda net (mai s'ha fet servir).

Etiquetes (del dataset de generar_dataset_cartes.py):
  - `slot_values` (N,3): valor quasi-òptim de cada slot (NaN si il·legal).
  - `slot_mask`  (N,3): slots legals.
  - política: soft = softmax(slot_values / τ) sobre slots legals (destil·lació).
  - valor:   value_target = max sobre slots legals (valor de l'estat sota
             continuació òptima) -> warm-start del crític.
"""
from __future__ import annotations

import sys
import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F

if '__file__' in globals():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from RL.entrenament.entrenament_sb3 import crear_entorn, STAGE_FLAGS, OPPONENT_PESOS
from RL.models.sb3.sb3_features_extractor import CosMultiInputSB3
from RL.models.sb3.multi_head_policy import MultiHeadMaskableACPolicy

NEG = -1e9


def construir_model(seed: int) -> MaskablePPO:
    """Crea una MaskablePPO(MultiHeadMaskableACPolicy) offline, amb el MATEIX
    policy_kwargs que entrenament_sb3.py (etapa cartes), per garantir formes
    compatibles amb el resume d'RL. No s'entrena via .learn(); l'entorn només
    proporciona observation_space/action_space."""
    env_config = {"num_jugadors": 2, "cartes_jugador": 3, "senyes": False, **STAGE_FLAGS["cartes"]}
    # pool_dir irrellevant: amb pesos "random" l'OpponentPool no llegeix cap
    # checkpoint. L'entorn només serveix per fixar observation/action_space.
    opponent_kwargs = {"pool_dir": "._sl_dummy_pool", "pesos": OPPONENT_PESOS["random"]}
    env = DummyVecEnv([crear_entorn(env_config, opponent_kwargs, seed=seed)])

    policy_kwargs = dict(
        features_extractor_class=CosMultiInputSB3,
        features_extractor_kwargs=dict(features_dim=256, in_channels=3, context_size=17),
        net_arch=[256, 256],
    )
    model = MaskablePPO(
        MultiHeadMaskableACPolicy, env,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )
    return model


def forward_sl(policy, obs_t):
    """Forward SL manual: features compartides -> latents pi/vf -> els 3 logits
    de cartes (crida directa a head_cartes, sense scatter) + el valor escalar."""
    features = policy.features_extractor(obs_t)            # (B,256)  (share_features_extractor=True)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    latent_vf = policy.mlp_extractor.forward_critic(features)
    cartes_logits = policy.action_net.head_cartes(latent_pi)   # (B,3)
    value = policy.value_net(latent_vf).squeeze(-1)            # (B,)
    return cartes_logits, value


def calcular_targets(slot_values_t, slot_mask_t, tau: float):
    """De (slot_values, slot_mask) -> (soft_label, value_target). soft = softmax
    dels valors/τ sobre slots legals; value_target = màxim valor legal."""
    vals = torch.where(slot_mask_t, slot_values_t, torch.full_like(slot_values_t, NEG))
    soft = torch.softmax(vals / tau, dim=1)
    soft = soft * slot_mask_t                    # zero exacte als il·legals
    soft = soft / soft.sum(dim=1, keepdim=True)
    value_target = vals.max(dim=1).values        # màxim sobre slots legals
    return soft, value_target


def perdua_i_metriques(cartes_logits, value, soft, value_target, slot_mask_t, lambda_v):
    masked = cartes_logits.masked_fill(~slot_mask_t, NEG)
    logp = F.log_softmax(masked, dim=1)
    loss_pi = -(soft * logp).sum(dim=1).mean()
    loss_v = F.mse_loss(value, value_target)
    loss = loss_pi + lambda_v * loss_v

    with torch.no_grad():
        pred = masked.argmax(dim=1)
        millor = soft.argmax(dim=1)
        top1 = (pred == millor).float().mean()
        vmae = (value - value_target).abs().mean()
    return loss, loss_pi.detach(), loss_v.detach(), top1, vmae


def main():
    ap = argparse.ArgumentParser(description="Warm-start SL del cap de cartes/valor sobre MaskablePPO")
    ap.add_argument('--dataset', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'dades_sl', 'dataset_cartes.npz'))
    ap.add_argument('--out', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'models_sl', 'sl_cartes'))
    ap.add_argument('--tau', type=float, default=0.2, help="temperatura de la softmax de l'etiqueta tova")
    ap.add_argument('--lambda_v', type=float, default=0.5, help="pes de la pèrdua de valor (MSE)")
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--val_split', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.dataset)
    obs = torch.as_tensor(d['obs'], dtype=torch.float32)
    slot_values = torch.as_tensor(np.nan_to_num(d['slot_values'], nan=0.0), dtype=torch.float32)
    slot_mask = torch.as_tensor(d['slot_mask'], dtype=torch.bool)
    n = obs.shape[0]
    print(f"Dataset: {n} estats | obs={tuple(obs.shape)} | tau={args.tau} lambda_v={args.lambda_v}")

    # Split train/val
    perm = torch.randperm(n)
    n_val = int(n * args.val_split)
    idx_val, idx_tr = perm[:n_val], perm[n_val:]

    model = construir_model(args.seed)
    policy = model.policy
    device = policy.device
    policy.set_training_mode(True)

    obs, slot_values, slot_mask = obs.to(device), slot_values.to(device), slot_mask.to(device)

    # Optimitzador SEPARAT sobre els 4 submòduls (no policy.optimizer).
    sl_params = (list(policy.features_extractor.parameters())
                 + list(policy.mlp_extractor.parameters())
                 + list(policy.action_net.head_cartes.parameters())
                 + list(policy.value_net.parameters()))
    opt = torch.optim.Adam(sl_params, lr=args.lr, weight_decay=args.weight_decay)

    def avaluar(idx):
        policy.set_training_mode(False)
        with torch.no_grad():
            logits, value = forward_sl(policy, obs[idx])
            soft, vt = calcular_targets(slot_values[idx], slot_mask[idx], args.tau)
            loss, lpi, lv, top1, vmae = perdua_i_metriques(
                logits, value, soft, vt, slot_mask[idx], args.lambda_v)
        policy.set_training_mode(True)
        return loss.item(), lpi.item(), lv.item(), top1.item(), vmae.item()

    millor_val = float('inf')
    millor_estat = None
    sense_millora = 0

    for epoch in range(1, args.epochs + 1):
        policy.set_training_mode(True)
        ordre = idx_tr[torch.randperm(idx_tr.shape[0])]
        for i in range(0, ordre.shape[0], args.batch_size):
            b = ordre[i:i + args.batch_size]
            logits, value = forward_sl(policy, obs[b])
            soft, vt = calcular_targets(slot_values[b], slot_mask[b], args.tau)
            loss, *_ = perdua_i_metriques(logits, value, soft, vt, slot_mask[b], args.lambda_v)
            opt.zero_grad()
            loss.backward()
            opt.step()

        vloss, vlpi, vlv, vtop1, vvmae = avaluar(idx_val)
        if epoch == 1 or epoch % 5 == 0 or vloss < millor_val:
            print(f"epoch {epoch:3d} | val_loss={vloss:.4f} (pi={vlpi:.4f} v={vlv:.4f}) "
                  f"| top1={vtop1:.3f} | value_MAE={vvmae:.3f}")

        if vloss < millor_val - 1e-5:
            millor_val = vloss
            millor_estat = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            sense_millora = 0
        else:
            sense_millora += 1
            if sense_millora >= args.patience:
                print(f"Early stopping a l'època {epoch} (millor val_loss={millor_val:.4f})")
                break

    if millor_estat is not None:
        policy.load_state_dict(millor_estat)

    _, _, _, top1_f, vmae_f = avaluar(idx_val)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print("=" * 70)
    print(f"Millor val_loss={millor_val:.4f} | top1 final={top1_f:.3f} | value_MAE final={vmae_f:.3f}")
    print(f"Model MaskablePPO desat a: {args.out}.zip")
    print("Continua amb: python -m RL.entrenament.entrenament_sb3 --multi_head "
          f"--resume_from {args.out}.zip --stage envit ...")


if __name__ == '__main__':
    main()
