import torch
import torch.onnx
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.xarxa_unificada import XarxaUnificada

def export_best_to_onnx(pt_path, onnx_path):
    print(f"Carregant pesos de: {pt_path}")
    
    hidden_layers = [256, 256]
    # Creem el model en CPU per exportar
    model = XarxaUnificada(n_actions=19, mlp_layers=hidden_layers, mode="scratch", device=torch.device("cpu"), output="q")
    
    # Carregar pesos
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=True)
    q_sd = checkpoint.get("q_net", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(q_sd)
    model.eval()


    dummy_input = torch.randn(1, 233)
    print(f"Exportant a {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Exportació completada amb èxit.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exporta un model .pt a .onnx")
    parser.add_argument("--input", type=str, required=True, help="Ruta al fitxer .pt")
    parser.add_argument("--output", type=str, default="model_truc.onnx", help="Ruta de sortida .onnx")
    
    args = parser.parse_args()
    
    export_best_to_onnx(args.input, args.output)
