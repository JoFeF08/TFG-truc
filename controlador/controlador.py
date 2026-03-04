from __future__ import annotations

from entorn.cartes_accions import ACTION_LIST
from controlador.interficie_model import Model
from vista.interficie_vista import Vista


class Controlador:
    """Controlador únic que funciona amb qualsevol Vista i Model"""

    def __init__(self, vista: Vista, model: Model) -> None:
        self.vista = vista
        self.model = model

    def executar_partida(self, override_config: dict = None) -> None:
        config = override_config if override_config is not None else self.vista.demanar_config()
        if override_config is not None and hasattr(self.vista, '_config'):
            self.vista._config = config
        self.model.iniciar(config)

        while not self.model.es_final():
            pid = self.model.get_jugador_actual()
            if self.model.es_huma(pid):
                estat = self.model.get_estat(pid)
                self.vista.mostrar_estat(estat)
                accio = self.vista.escollir_accio(estat["accions_legals"], estat)
                if accio is None:
                    return  # han tancat la finestra
                self.model.aplicar_accio(accio)
                self.vista.mostrar_accio(pid, ACTION_LIST[accio], es_bot=False)
            else:
                # Mostrar l'estat des de perspectiva humana
                huma_pid = self._trobar_huma()
                if huma_pid is not None:
                    estat_huma = self.model.get_estat(huma_pid)
                    self.vista.mostrar_estat(estat_huma)

                accio, nom = self.model.get_accio_bot(pid)
                self.model.aplicar_accio(accio)
                self.vista.mostrar_accio(pid, nom, es_bot=True)

        resultat = self.model.get_resultat()
        self.vista.mostrar_fi_partida(resultat["score"], resultat["payoffs"])

    def bucle_principal(self) -> None:
        while True:
            self.executar_partida()
            if not self.vista.demanar_repetir():
                self.vista.mostrar_sortint()
                break

    def _trobar_huma(self) -> int | None:
        """Retorna l'ID del primer jugador humà, o None si no n'hi ha."""
        if hasattr(self.model, '_humans') and self.model._humans:
            return next(iter(self.model._humans))
        return None
