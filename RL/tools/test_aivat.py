import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from joc.entorn.env import TrucEnv
from RL.tools.aivat_evaluator import AIVATEvaluator
from rlcard.agents import DQNAgent, RandomAgent

def test_aivat():
    print("--- Test AIVAT Implementation ---")
    
    config = {
        'num_jugadors': 2,
        'cartes_jugador': 3,
        'puntuacio_final': 12,
        'seed': 42
    }
    env = TrucEnv(config)
    
    # Crear un agent DQN (amb pesos aleatoris) per als càlculs de V
    # AIVAT necessita una funció de valor aproximada (Q-net)
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[256, 256],
        device=torch.device("cpu")
    )
    
    # Oponent aleatori per a la prova
    opponent = RandomAgent(num_actions=env.num_actions)
    
    # Evaluador amb pocs samples per anar ràpid al test
    evaluator = AIVATEvaluator(agent, env, num_samples=5)
    
    try:
        print("Corrent avaluació de 5 episodis de prova...")
        results = evaluator.run_evaluation(opponent, num_episodes=5)
        
        print("\nResultats de la prova:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        print("\n[OK] L'avaluador AIVAT ha funcionat sense errors.")
        
    except Exception as e:
        print(f"\n[ERROR] Ha fallat l'avaluació: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_aivat()
