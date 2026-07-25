"""Exporta el COS congelat (CosMultiInput) d'un model SB3 (.zip) a un .pth.

Motiu: el cos del llinatge F4→F5→F6 (congelat) no és cap fitxer de
`RL/entrenament/entrenamentEstatTruc/registres/` (el `15_04` té l'arquitectura
correcta però els pesos difereixen). El cos exacte només és recuperable del propi
zip d'un model de la cadena (F5 best_robust o F6 best.zip, que són idèntics).

El .pth resultant és compatible amb
`CosMultiInputSB3.carregar_pesos_preentrenats` (fa `self.cos.load_state_dict(state)`),
de manera que es pot passar com a `--pesos_cos` a entrenament_fase6.py i la SL.

Ús:
    python RL/tools/exportar_cos_de_zip.py <model.zip> <sortida.pth>
"""
from __future__ import annotations

import argparse
import os
import sys

# L'arrel del repo ha de ser al path: el zip SB3 desa policy_kwargs amb una
# referència (via cloudpickle) a CosMultiInputSB3, que viu sota el paquet `RL`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from stable_baselines3.common.save_util import load_from_zip_file


_PREFIX = "features_extractor.cos."


def exportar_cos(zip_path: str, out_path: str) -> int:
    """Extreu els tensors `features_extractor.cos.*` del policy state_dict del zip
    i els desa com a state_dict de `CosMultiInput` (sense el prefix). Retorna el
    nombre de tensors exportats."""
    _data, params, _vars = load_from_zip_file(zip_path, device="cpu")
    policy_sd = params["policy"]

    cos_sd: dict[str, torch.Tensor] = {
        k[len(_PREFIX):]: v.clone()
        for k, v in policy_sd.items()
        if k.startswith(_PREFIX)
    }
    if not cos_sd:
        raise ValueError(
            f"No s'han trobat tensors amb prefix '{_PREFIX}' a {zip_path}. "
            "El model no usa CosMultiInputSB3?"
        )

    torch.save(cos_sd, out_path)
    return len(cos_sd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta el COS d'un model SB3 a .pth")
    parser.add_argument("zip_path", type=str, help="Ruta al model SB3 (.zip)")
    parser.add_argument("out_path", type=str, help="Ruta de sortida (.pth)")
    args = parser.parse_args()

    n = exportar_cos(args.zip_path, args.out_path)
    print(f"COS exportat: {n} tensors  {args.zip_path} -> {args.out_path}")


if __name__ == "__main__":
    main()
