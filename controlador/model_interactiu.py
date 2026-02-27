from __future__ import annotations

from entorn.game import TrucGame
from entorn.cartes_accions import ACTION_LIST
from entorn.rols.player.player import TrucPlayer
from entorn.rols.player.player_default import DefaultPlayer


class _SlotHuma(TrucPlayer):
    """Placeholder per jugadors humans. L'acció arriba del controlador, no del model."""

    def triar_accio(self, estat):
        raise RuntimeError("L'acció humana ha d'arribar del controlador, no del model")


class ModelInteractiu:
    """Wrapper de TrucGame que implementa el Protocol Model."""

    def __init__(self) -> None:
        self._game: TrucGame | None = None
        self._humans: set[int] = set()
        self._num_jugadors: int = 2

    def iniciar(self, config: dict) -> None:
        self._num_jugadors = config.get("num_jugadors", 2)
        tipus = config.get("tipus_jugadors", {0: 0, 1: 1})
        self._humans = {int(i) for i, t in tipus.items() if t == 0}

        player_classes: dict = {}
        for i in range(self._num_jugadors):
            if i in self._humans:
                player_classes[i] = _SlotHuma
            else:
                player_classes[i] = DefaultPlayer

        self._game = TrucGame(
            num_jugadors=self._num_jugadors,
            cartes_jugador=config.get("cartes_jugador", 3),
            senyes=config.get("senyes", False),
            puntuacio_final=config.get("puntuacio_final", 24),
            player_class=player_classes,
            verbose=config.get("verbose", False),
        )
        self._game.init_game()

    def get_estat(self, jugador_id: int) -> dict:
        assert self._game is not None
        estat = self._game.get_state(jugador_id)
        estat["num_jugadors"] = self._num_jugadors
        return estat

    def get_jugador_actual(self) -> int:
        assert self._game is not None
        return self._game.current_player

    def es_huma(self, jugador_id: int) -> bool:
        return jugador_id in self._humans

    def get_accio_bot(self, jugador_id: int) -> tuple[int, str]:
        assert self._game is not None
        state = self._game.get_state(jugador_id)
        player = self._game.players[jugador_id]
        accio = int(player.triar_accio(state))
        return accio, ACTION_LIST[accio]

    def aplicar_accio(self, accio: int) -> None:
        assert self._game is not None
        self._game.step(int(accio))

    def es_final(self) -> bool:
        if self._game is None:
            return True
        return self._game.is_over()

    def get_resultat(self) -> dict:
        assert self._game is not None
        return {
            "score": list(self._game.score),
            "payoffs": self._game.get_payoffs(),
        }
