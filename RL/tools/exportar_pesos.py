import os
import sys
import torch
import numpy as np

def exportar_model(ruta_pt, ruta_npz):
    print(f"[{os.path.basename(__file__)}] Carregant els pesos de: {ruta_pt}")
    checkpoint = torch.load(ruta_pt, map_location='cpu', weights_only=True)
    
    # Extraient estats
    q_sd = checkpoint.get("q_net", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    
    np_dict = {}
    for k, v in q_sd.items():
        v_np = v.numpy()www
        if 'weight' in k and len(v_np.shape) == 2:
            v_np = v_np.T
            
        np_dict[k] = v_np
        print(f"Exportat {k} amb forma {v_np.shape}")
        
    np.savez(ruta_npz, **np_dict)
    print(f"[{os.path.basename(__file__)}] Model exportat a: {ruta_npz} correctament!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Ús: python exportar_pesos.py <ruta_entrada.pt> <ruta_sortida.npz>")
        sys.exit(1)
        
    ruta_in = sys.argv[1]
    ruta_out = sys.argv[2]
    exportar_model(ruta_in, ruta_out)
