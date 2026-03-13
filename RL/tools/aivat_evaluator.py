import numpy as np
import torch
from tqdm import tqdm
import copy as _copy

class AIVATEvaluator:
    """
    Evaluador AIVAT per reduir la variància en les mètriques de rendiment.
    Abordatge: Reward_AIVAT = Reward_Real - Luck_Initial_Deal - Luck_Opponent_Actions
    """
    def __init__(self, agent, env, num_samples=10):
        self.agent = agent
        self.env = env
        self.num_samples = num_samples
        # Obtenir el device des de l'agent RLCard
        if hasattr(agent, 'device'):
            self.device = agent.device
        else:
            self.device = torch.device("cpu")

    def _get_v_value(self, game, player_id, agent=None):
        """Estima el valor V(s). Per a estats terminals és el payoff.
        Per a estats no terminals és max Q(s, a).
        """
        # Si el joc ha acabat, el valor és el payoff real
        if game.is_over():
            return float(game.get_payoffs()[player_id])

        if agent is None:
            agent = self.agent

        # Si l'agent no té funció de valor, la correcció és nul·la
        q_estimator = (
            agent.q_estimator if hasattr(agent, 'q_estimator')
            else getattr(getattr(agent, 'rl_agent', None), 'q_estimator', None)
        )
        if q_estimator is None:
            return 0.0

        state = game.get_state(player_id)
        extracted = self.env._extract_state(state)
        obs = extracted['obs']

        if isinstance(obs, dict):
            flat = np.concatenate([
                obs['obs_cartes'].flatten(),
                obs['obs_context'],
            ], axis=0).astype(np.float32)
        else:
            flat = obs

        obs_tensor = torch.from_numpy(np.ascontiguousarray([flat])).to(self.device)

        with torch.no_grad():
            q_values = q_estimator.qnet(obs_tensor).cpu().numpy()[0]

        legal_actions = extracted['legal_actions']
        if not legal_actions:
            return 0.0

        # Filtrem accions legals
        mask = np.full(self.env.num_actions, -1e9, dtype=np.float32)
        for act in legal_actions:
            mask[act] = q_values[act]

        return float(np.max(mask))

    def _prepare_state_for_agent(self, extracted_state):
        obs = extracted_state['obs']
        if isinstance(obs, dict):
            flat = np.concatenate([
                obs['obs_cartes'].flatten(),
                obs['obs_context'],
            ], axis=0).astype(np.float32)
            flat_state = dict(extracted_state)
            flat_state['obs'] = flat
            return flat_state
        return extracted_state

    def _get_expected_v(self, game, observer_id, agent=None, resample_observer=False):
        """Estima E[V] mitjançant resampling."""
        v_samples = []
        for _ in range(self.num_samples):
            resampled_game = game.clone_and_resample(observer_id, resample_observer=resample_observer)
            v = self._get_v_value(resampled_game, observer_id, agent=agent)
            v_samples.append(v)
        return np.mean(v_samples)

    def evaluate_episode(self, opponent_agent):
        """
        Executa una partida i calcula la correcció AIVAT de forma robusta (zero-centered).
        Abordatge: Correction = sum(V_outcome - E[V_outcome | info_before])
        """
        game = self.env.game
        self.env.action_recorder = []
        game.init_game()
        
        total_correction = 0.0
        observer_id = 0 
        
        # 1. Sort Inicial (Initial Deal Luck)
        v_start_real = self._get_v_value(game, observer_id, agent=self.agent)
        v_start_expected = self._get_expected_v(game, observer_id, agent=self.agent, resample_observer=True)
        total_correction += (v_start_real - v_start_expected)
        
        while not game.is_over():
            curr_pid = game.current_player
            
            # Triar acció
            current_state = self._prepare_state_for_agent(
                self.env._extract_state(game.get_state(curr_pid))
            )
            
            if curr_pid == observer_id:
                # Acció de l'agent: l'executem normalment
                action, _ = self.agent.eval_step(current_state)
                game.step(int(action))
            else:
                # Acció de l'oponent: calculem la seva "sort" per a nosaltres
                legal_actions = game.get_legal_actions()
                v_after_samples = []
                
                # Estimem el valor esperat després de qualsevol acció de l'oponent
                for act in legal_actions:
                    g_clone = game.clone()
                    g_clone.step(act)
                    v_after_samples.append(self._get_v_value(g_clone, observer_id, agent=self.agent))
                
                v_expected_after = np.mean(v_after_samples) if v_after_samples else 0.0
                
                # Executem l'acció real
                action, _ = opponent_agent.eval_step(current_state)
                game.step(int(action))
                
                # V real després de l'acció triada
                v_real_after = self._get_v_value(game, observer_id, agent=self.agent)
                
                # La correcció és la "sort" de l'acció triada
                total_correction += (v_real_after - v_expected_after)
            
        real_reward = game.get_payoffs()[observer_id]
        aivat_reward = real_reward - total_correction
        
        return float(real_reward), float(aivat_reward), float(total_correction)

    def run_evaluation(self, opponent_agent, num_episodes=50):
        """Sessió d'avaluació amb comparativa de variància."""
        real_history = []
        aivat_history = []
        
        for ep in tqdm(range(num_episodes), desc="AIVAT Eval"):
            r, a, c = self.evaluate_episode(opponent_agent)
            real_history.append(r)
            aivat_history.append(a)
            
        real_mean = np.mean(real_history)
        real_std = np.std(real_history)
        aivat_mean = np.mean(aivat_history)
        aivat_std = np.std(aivat_history)
        
        return {
            'episodes': num_episodes,
            'real_reward_mean': real_mean,
            'real_reward_std': real_std,
            'aivat_reward_mean': aivat_mean,
            'aivat_reward_std': aivat_std,
            'variance_reduction_pct': (1.0 - (aivat_std / (real_std + 1e-9))) * 100
        }
