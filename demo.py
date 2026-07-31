from joc.controlador import Controlador, ModelInteractiu
from joc.vista.vista_desktop.vista_desktop import VistaDesktop

import sys
import os

def resource_path(relative_path):
    """
    Get absolute path to resource.
    - PyInstaller onefile: usa sys._MEIPASS (directori d'extracció temporal)
    - Nuitka onefile:      usa __file__ (apunta al directori d'extracció de Nuitka)
    - Desenvolupament:     usa el directori del propi script
    """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller onefile
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# Pesos: entrenament --stage cartes --opponent fort (mans sense apostes).
# NOTA: el checkpoint anterior (25_07_26_a_les_1725) es va esborrar per
# accident amb un rm -rf massa ampli -- aquest és el run local que el
# substitueix (encara en curs). Checkpoint de 500k triat per avaluació amb
# repartiments duplicats (RL/tools/avaluacio_duplicada.py) contra random +
# 4 estils de regles + probabilistic -- és el "menys dolent" dels provats
# (500k/5.5M/6.5M), però encara perd net contra el conjunt (només bat
# random i, per poc, conservador). L'entrenament té un problema real
# d'inestabilitat de KL sense resoldre -- no esperis un bon nivell de joc.
MODEL_PATH = resource_path("RL\\entrenament\\registres_sb3\\26_07_26_a_les_2311\\pool\\ppo_truc_500000_steps.zip")
# "cfr" = equilibri CFR per a les apostes (truc/envit) + solver quasi-òptim per
# a les cartes -> joc COMPLET amb apostes. "probabilistic" = només cartes.
TIPUS_AGENT = "cfr"  # "cfr" | "sb3" | "regles" | "probabilistic"
ALGORISME    = "maskable_ppo"  # només per "sb3": "maskable_ppo" | "ppo" | "dqn" | "ppo_lstm"
VARIANT_REGLES = "conservador"  # només per "regles": conservador|equilibrat|agressiu|farol

# El CFR es va resoldre per al context de marcador 0-0 (falta/joc-fora = 24);
# és exacte a l'inici de partida i aproximat a prop del final (límit documentat).
PERMETRE_APOSTES = (TIPUS_AGENT == "cfr")  # joc complet amb truc/envit contra el CFR

PARTIDES_SESSIO = 1

# Verificació doble (repartiments duplicats, com RL/tools/avaluacio_duplicada.py
# però jugant-hi tu mateix): juga UNA mà amb un repartiment, i després la
# MATEIXA mà exacta amb els seients intercanviats (tu reps les cartes que
# abans tenia la IA, i viceversa). Així es cancel·la la sort del
# repartiment i es compara només la qualitat de les decisions. No toca el
# flux normal (PARTIDES_SESSIO) si es deixa a False.
VERIFICACIO_DOBLE = False

# Spec del jugador IA segons el tipus
if TIPUS_AGENT == "sb3":
    _spec_ia = {"tipus": "sb3", "algorisme": ALGORISME, "ruta": MODEL_PATH}
elif TIPUS_AGENT == "regles":
    _spec_ia = {"tipus": "regles", "variant": VARIANT_REGLES}
elif TIPUS_AGENT in ("probabilistic", "cfr"):
    _spec_ia = {"tipus": TIPUS_AGENT}
else:
    _spec_ia = {"tipus": TIPUS_AGENT, "ruta": MODEL_PATH}

config = {
    "num_jugadors": 2,
    "cartes_jugador": 3,
    "senyes": False,
    "puntuacio_final": 24,
    "permetre_apostes": PERMETRE_APOSTES,
    "permetre_truc": PERMETRE_APOSTES,
    "tipus_jugadors": {
        0: {"tipus": "huma"},
        1: _spec_ia,
    },
}


def _config_una_ma(huma_pid: int, deal_seed: int) -> dict:
    """Configuració base, amb un sol seient humà/IA (segons huma_pid) i un
    deal_seed fixat perquè la SEQÜÈNCIA sencera de repartiments (mà 1, mà 2,
    mà 3...) sigui reproduïble d'una partida a l'altra -- es fixa el
    barrejador un sol cop a l'inici, i cada mà nova en consumeix la
    continuació, així que reproduir el mateix seed reprodueix la mateixa
    seqüència de mans sencera, no només la primera.
    puntuacio_final molt alt (no s'hi arriba mai en la pràctica) perquè la
    partida no s'aturi per marcador -- dura tantes mans com es vulgui, fins
    que es tanca la finestra (el bucle de Controlador.executar_partida ja
    surt net quan es tanca)."""
    ia_pid = 1 - huma_pid
    return {
        "num_jugadors": 2,
        "cartes_jugador": 3,
        "senyes": False,
        "puntuacio_final": 999,
        "permetre_apostes": False,
        "permetre_truc": False,
        "deal_seed": deal_seed,
        "tipus_jugadors": {
            huma_pid: {"tipus": "huma"},
            ia_pid: _spec_ia,
        },
    }


def _executar_verificacio_doble(controlador) -> None:
    import random
    deal_seed = random.randrange(1 << 30)
    print(f"\n=== Verificació doble (deal_seed={deal_seed}) ===")
    print("Cada partida dura tantes mans com vulguis -- tanca la finestra quan vulguis acabar-la.")

    print("\nPartida 1/2: tu ets el jugador 0. Tanca la finestra quan vulguis passar a la 2...")
    controlador.executar_partida(override_config=_config_una_ma(huma_pid=0, deal_seed=deal_seed))
    resultat_a = controlador.model.get_resultat()
    mans_a = controlador.model._game.comptador_ma - 1

    print(f"\nPartida 2/2 ({mans_a} mans jugades a la 1a): mateix repartiment, ara tu ets el jugador 1 "
          f"(les cartes que abans tenia la IA). Per una comparació justa, tanca-la al mateix nombre de mans...")
    controlador.executar_partida(override_config=_config_una_ma(huma_pid=1, deal_seed=deal_seed))
    resultat_b = controlador.model.get_resultat()
    mans_b = controlador.model._game.comptador_ma - 1

    huma_total = resultat_a["score"][0] + resultat_b["score"][1]
    ia_total = resultat_a["score"][1] + resultat_b["score"][0]
    marge = huma_total - ia_total

    print("\n--- Resultat de la verificació doble ---")
    print(f"  Partida 1 (tu=J0, {mans_a} mans): {resultat_a['score']}")
    print(f"  Partida 2 (tu=J1, {mans_b} mans): {resultat_b['score']}")
    if mans_a != mans_b:
        print(f"  ⚠️ Nombre de mans diferent ({mans_a} vs {mans_b}) -- la comparació de totals no és del tot justa,"
              " nomes les primeres min(mans_a, mans_b) mans venen del mateix repartiment exacte.")
    print(f"  Total tu: {huma_total}   Total IA: {ia_total}   Marge: {marge:+d}")
    if marge > 0:
        print("  Has jugat millor que la IA amb les mateixes cartes.")
    elif marge < 0:
        print("  La IA ha jugat millor que tu amb les mateixes cartes.")
    else:
        print("  Empat -- mateixa qualitat de decisions amb aquest repartiment.")


def _reset_memoria_agents(controlador):
    """Crida reset_memoria() a tots els agents IA del controlador (si el suporten)."""
    try:
        models = getattr(controlador, '_models', None) or getattr(controlador.model, '_models', None)
        if models:
            for m in models.values():
                if m is not None and hasattr(m, 'reset_memoria'):
                    m.reset_memoria()
    except Exception:
        pass


if __name__ == "__main__":
    print("Iniciant demo.py...")
    vista = VistaDesktop()
    model = ModelInteractiu()
    controlador = Controlador(vista, model)
    try:
        _reset_memoria_agents(controlador)  # inici de sessió
        if VERIFICACIO_DOBLE:
            _executar_verificacio_doble(controlador)
        else:
            for partida_idx in range(PARTIDES_SESSIO):
                print(f"Executant partida {partida_idx + 1}/{PARTIDES_SESSIO}...")
                controlador.executar_partida(override_config=config)
    except KeyboardInterrupt:
        print("Sortint...")
        vista.mostrar_sortint()

