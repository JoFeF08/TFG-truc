import numpy as np
import torch
from joc.entorn.game import TrucGame
from RL.models.rlcard_legacy.loader import crear_model

def run_test():
    # Rutes
    ruta_pt = r"C:\Users\ferri\Documents\ProjectesCodi\TFG-truc\entrenament\entrenamentsUnificats\registres\resultats_comparativa\dqn_finetune_0803_0024\models\best.pt"
    ruta_np = r"C:\Users\ferri\Documents\ProjectesCodi\TFG-truc\RL\models\best.npz"

    env_config = {
        "num_jugadors": 2,
        "cartes_jugador": 3,
        "senyes": False
    }

    # Carregar models
    print("Carregant model PyTorch...")
    model_pt = crear_model({"tipus": "dqn", "ruta": ruta_pt}, env_config)
    
    print("Carregant model NumPy...")
    model_np = crear_model({"tipus": "numpy_dqn", "ruta": ruta_np}, env_config)

    print("Models carregats.")

    print("\nGenerant partides i comparant sortides...")
    
    num_partides = 5
    diferencia_maxima = 0.0

    for i in range(num_partides):
        game = TrucGame(num_jugadors=2)
        game.init_game()

        while not game.is_over():
            pid = game.current_player
            state = game.get_state(pid)
            
            # Forward PyTorch
            obs = model_pt._extract(state)['obs']
            obs_array = np.array([obs], dtype=np.float32)
            obs_tensor = torch.tensor(obs_array, device=model_pt._agent.device)
            with torch.no_grad():
                q_pt = model_pt._agent.q_estimator.qnet(obs_tensor).cpu().numpy()[0]
                
            # Forward NumPy
            q_np = model_np.forward(obs_array)[0]
            
            # Càlcul d'error numèric
            diff = np.max(np.abs(q_pt - q_np))
            diferencia_maxima = max(diferencia_maxima, diff)
            
            # Accions triades (el mètode triar_accio s'encarrega d'emmascarar i agafar l'argminmax)
            accio_pt = model_pt.triar_accio(state)
            accio_np = model_np.triar_accio(state)
            
            assert accio_pt == accio_np, f"Mismatch d'acció! PT ha triat {accio_pt}, NP ha triat {accio_np}"
            
            # Avancem el joc
            game.step(accio_pt)

    print(f"\nTest superat! Número de partides provades: {num_partides}")
    print("Totes les accions coincideixen entre models.")
    print(f"Diferència màxima absoluta entre Q-values (PT vs NP): {diferencia_maxima:.8e}")

if __name__ == '__main__':
    run_test()
