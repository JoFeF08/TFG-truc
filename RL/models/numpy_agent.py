import os
import numpy as np
from typing import Any, Callable
from RL.models.loader import TrucModel

def _crear_env_temp(env_config: dict[str, Any]):
    from joc.entorn.env import TrucEnv
    from RL.models.adapters.feature_extractor import wrap_env_aplanat
    env = TrucEnv(
        config={
            "num_jugadors": env_config.get("num_jugadors", 2),
            "cartes_jugador": env_config.get("cartes_jugador", 3),
            "senyes": env_config.get("senyes", False),
        }
    )
    return wrap_env_aplanat(env)

class ModelNumpy:
    """Implementació de NumPy de la XarxaUnificada per evitar dependències de PyTorch."""
    def __init__(self, npz_path: str, extract_state_fn: Callable[[dict[str, Any]], dict[str, Any]]):
        self._extract = extract_state_fn
        self.weights = np.load(npz_path)

        self.split = 6 * 4 * 9
        self.obs_cartes_shape = (6, 4, 9)

    def _conv2d(self, x, w, b):
        # x: (N, in_c, H, W)
        # w: (out_c, in_c, kH, kW)
        N, in_c, H, W = x.shape
        out_c, _, kH, kW = w.shape
        
        out_H = H - kH + 1
        out_W = W - kW + 1
        
        out = np.zeros((N, out_c, out_H, out_W), dtype=np.float32)
        for i in range(out_H):
            for j in range(out_W):
                region = x[:, :, i:i+kH, j:j+kW]
                region_flat = region.reshape(N, -1)
                w_flat = w.reshape(out_c, -1).T
                out[:, :, i, j] = np.dot(region_flat, w_flat)
                
        out += b[np.newaxis, :, np.newaxis, np.newaxis]
        return out

    def forward(self, obs: np.ndarray) -> np.ndarray:
        # obs: (N, 233)
        cartes_f = obs[:, :self.split]
        context = obs[:, self.split:]
        cartes = cartes_f.reshape(-1, *self.obs_cartes_shape)
        
        # cos.branca_cnn
        x_cnn = self._conv2d(cartes, self.weights['cos.branca_cnn.0.weight'], self.weights['cos.branca_cnn.0.bias'])
        x_cnn = np.maximum(0, x_cnn) # ReLU
        x_cnn = self._conv2d(x_cnn, self.weights['cos.branca_cnn.2.weight'], self.weights['cos.branca_cnn.2.bias'])
        x_cnn = np.maximum(0, x_cnn) # ReLU
        x_cnn = x_cnn.reshape(obs.shape[0], -1) # Flatten
        
        # cos.branca_densa
        w_densa = self.weights['cos.branca_densa.0.weight']
        b_densa = self.weights['cos.branca_densa.0.bias']
        x_context = np.dot(context, w_densa) + b_densa
        x_context = np.maximum(0, x_context) # ReLU
        
        # concatena
        x_fusio = np.concatenate([x_cnn, x_context], axis=1)
        
        # cos.fusio
        w_fusio_0 = self.weights['cos.fusio.0.weight']
        b_fusio_0 = self.weights['cos.fusio.0.bias']
        x_fusio = np.dot(x_fusio, w_fusio_0) + b_fusio_0
        x_fusio = np.maximum(0, x_fusio)
        
        w_fusio_2 = self.weights['cos.fusio.2.weight']
        b_fusio_2 = self.weights['cos.fusio.2.bias']
        x_fusio = np.dot(x_fusio, w_fusio_2) + b_fusio_2
        z = np.maximum(0, x_fusio)
        
        # mlp
        w_mlp_0 = self.weights['mlp.0.weight']
        b_mlp_0 = self.weights['mlp.0.bias']
        x_mlp = np.dot(z, w_mlp_0) + b_mlp_0
        x_mlp = np.maximum(0, x_mlp)
        
        w_mlp_2 = self.weights['mlp.2.weight']
        b_mlp_2 = self.weights['mlp.2.bias']
        x_mlp = np.dot(x_mlp, w_mlp_2) + b_mlp_2
        x_mlp = np.maximum(0, x_mlp)
        
        w_mlp_4 = self.weights['mlp.4.weight']
        b_mlp_4 = self.weights['mlp.4.bias']
        out = np.dot(x_mlp, w_mlp_4) + b_mlp_4
        
        return out

    def triar_accio(self, estat: dict[str, Any]) -> int:
        rlcard_state = self._extract(estat)
        obs = rlcard_state['obs']
        legal_actions = list(rlcard_state['legal_actions'].keys())
        
        # Transformem a batch de 1 per numpy
        obs_array = np.array([obs], dtype=np.float32)
        q_values = self.forward(obs_array)[0]
        
        best_action = -1
        best_q = -float('inf')
        for action in legal_actions:
            if q_values[action] > best_q:
                best_q = q_values[action]
                best_action = action
                
        return int(best_action)

def _crear_numpy_dqn(spec: dict[str, Any], env_config: dict[str, Any]) -> TrucModel:
    ruta = spec["ruta"]
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No s'ha trobat el model numpy a: {ruta}")
        
    env_wrapped = _crear_env_temp(env_config)
    
    print(f"Model Numpy Lleuger carregat des de: {ruta}")
    return ModelNumpy(ruta, env_wrapped._extract_state)
