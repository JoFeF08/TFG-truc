import onnxruntime as ort
import numpy as np
import os
from typing import Any

class ONNXModelAdapter:
    """
    Adaptador que utilitza ONNX Runtime per fer inferència.
    NO depèn de torch, el que evita el hang en l'executable.
    """
    def __init__(self, onnx_path: str, state_extractor: Any):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"No s'ha trobat el model ONNX a: {onnx_path}")
        
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self._extract = state_extractor

    def triar_accio(self, estat: dict[str, Any]) -> int:
        """
        Calcula l'acció òptima segons l'estat del joc.
        Assumeix que estat és el diccionari complet del joc.
        """
        extracted = self._extract(estat)
        obs = extracted['obs']
        
        input_val = np.array(obs, dtype=np.float32)
        if input_val.ndim == 1:
            input_val = np.expand_dims(input_val, axis=0)
            
        # Inferència ONNX
        result = self.session.run(None, {self.input_name: input_val})
        q_values = result[0]
        
        if 'accions_legals' in estat:
            mask = np.full(q_values.shape, -np.inf)
            for act in estat['accions_legals']:
                mask[0, act] = 0
            q_values += mask
        elif 'legal_actions' in extracted:
            mask = np.full(q_values.shape, -np.inf)
            for act in extracted['legal_actions']:
                mask[0, act] = 0
            q_values += mask
            
        return int(np.argmax(q_values[0]))
