"""
Servidor FastAPI + WebSocket per jugar al Truc online.
Jugador 0 = humà (navegador), Jugador 1 = RandomPlayer.
Recull les transicions i les desa a dades_partides/.
"""
import asyncio
import queue
import random
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from entorn import TrucEnv, RandomPlayer, HumanPlayer, ACTION_LIST
from entorn.rols.player.escollidor_queue import EscollidorQueue
from entorn.data_collector import DataCollector

# Directori dels estàtics (vista_web)
VISTA_WEB = Path(__file__).resolve().parent / "vista" / "vista_web"

app = FastAPI(title="Truc Online")


def _to_json_serializable(obj):
    """Converteix objectes (numpy, enum, etc.) a tipus JSON-serialitzables."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


def run_partida(action_queue: queue.Queue, to_ws_queue: queue.Queue, collector: DataCollector):
    """Executa una partida en un fil; envia estats per to_ws_queue i rep accions per action_queue."""
    try:
        escollidor_web = EscollidorQueue(action_queue)
        player_classes = {
            0: lambda pid, rng: HumanPlayer(pid, rng, escollidor_web),
            1: lambda pid, rng: RandomPlayer(pid, rng, time_ms=500),
        }
        env_config = {
            "allow_step_back": False,
            "seed": random.randint(0, 100000),
            "player_class": player_classes,
            "num_jugadors": 2,
            "cartes_jugador": 3,
            "senyes": False,
            "puntuacio_final": 24,
        }
        env = TrucEnv(env_config)
        game = env.game
        collector.inici_partida()
        action_log = []

        state, player_id = env.reset()
        done = False

        while not done:
            raw = state.get("raw_obs", state)
            accions_legals = raw.get("accions_legals", [])
            id_jugador = raw.get("id_jugador", 0)
            nom_jugador = "Humà" if player_id == 0 else "Rival"

            # Sempre enviar l'estat des de la perspectiva del jugador humà (J0)
            estat_jugador_0 = game.get_state(0)
            state_per_client = _to_json_serializable({
                **estat_jugador_0,
                "action_log": [
                    {"player_id": e["player_id"], "nom_jugador": e["nom_jugador"], "action": int(e["action"]), "action_name": e["action_name"]}
                    for e in action_log
                ],
            })
            to_ws_queue.put({
                "type": "state",
                "state": state_per_client,
                "accions_legals": [int(a) for a in (accions_legals if player_id == 0 else [])],
            })

            if player_id == 0:
                # Esperar acció del client (ja es posarà a action_queue des del WebSocket)
                try:
                    action = action_queue.get(timeout=300)
                except queue.Empty:
                    action = 9 if 9 in accions_legals else accions_legals[0]
            else:
                # Jugador 1 (bot) tria
                try:
                    action = game.players[player_id].triar_accio(raw)
                except Exception as e:
                    action = accions_legals[0] if accions_legals else 9
                    import traceback
                    traceback.print_exc()
            action = int(action)

            # Registre per la UI web
            action_name = ACTION_LIST[action] if 0 <= action < len(ACTION_LIST) else str(action)
            action_log.append({
                "player_id": player_id,
                "nom_jugador": nom_jugador,
                "action": action,
                "action_name": action_name,
            })

            # Registrar transició per entrenament
            collector.registrar(
                state=raw,
                action=action,
                player_id=player_id,
                legal_actions=accions_legals,
                reward=0,
                done=False,
            )

            state, next_player_id = env.step(action)
            if game.is_over():
                done = True
                payoffs = env.get_payoffs()
                collector.finalitzar_partida(payoffs)
                to_ws_queue.put({
                    "type": "game_over",
                    "score": [int(x) for x in game.score],
                    "payoffs": [int(x) for x in payoffs],
                })
                break
            player_id = next_player_id

    except Exception as e:
        to_ws_queue.put({"type": "error", "message": str(e)})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    action_queue = queue.Queue()
    to_ws_queue = queue.Queue()
    collector = DataCollector()

    pendent = []
    game_thread = threading.Thread(
        target=run_partida,
        args=(action_queue, to_ws_queue, collector),
        daemon=True,
    )
    game_thread.start()

    try:
        while True:
            # Recollir tot el que el fil ha posat a to_ws_queue
            try:
                while True:
                    pendent.append(to_ws_queue.get_nowait())
            except queue.Empty:
                pass
            # Enviar tot el pendent (assegurar serialització JSON)
            for msg in pendent:
                try:
                    await websocket.send_json(_to_json_serializable(msg))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return
                if msg.get("type") in ("game_over", "error"):
                    return
            pendent.clear()
            # Esperar acció del client (timeout curt per tornar a comprovar pendent)
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.4)
                if data.get("action") is not None:
                    action_queue.put(int(data["action"]))
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# Muntar estàtics al final perquè /ws tingui prioritat
if VISTA_WEB.is_dir():
    app.mount("/", StaticFiles(directory=str(VISTA_WEB), html=True), name="static")


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
