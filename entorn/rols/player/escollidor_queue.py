"""
EscollidorAccio que rep l'acció via una queue (emplenada pel servidor WebSocket).
Permet reutilitzar HumanPlayer per al mode online.
"""
import queue


class EscollidorQueue:
    """EscollidorAccio que rep l'acció via una queue (emplenada pel servidor WebSocket)."""

    def __init__(self, action_queue: queue.Queue, timeout: int = 300):
        self._queue = action_queue
        self._timeout = timeout

    def escollir_accio(self, accions_legals: list, state: dict) -> int:
        try:
            return self._queue.get(timeout=self._timeout)
        except queue.Empty:
            return 9 if 9 in accions_legals else (accions_legals[0] if accions_legals else 9)
