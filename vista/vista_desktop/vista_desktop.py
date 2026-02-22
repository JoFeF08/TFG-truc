"""
Vista d'escriptori (Tkinter) per al Truc.
Implementa la mateixa interfície que VistaConsola; zero dependències extra.
El bucle Tkinter (mainloop) corre en un fil separat; els mètodes bloquejants
(demanar_config, escollir_accio, demanar_repetir) esperen amb threading.Event.
"""
import json
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from entorn.cartes_accions import ACTION_LIST

_IMG_DIR = Path(__file__).resolve().parent / "img_iu" / "cartesSeparades"

ACCIONS_CAT = [
    "Jugar carta 1", "Jugar carta 2", "Jugar carta 3",
    "Apostar envit", "Truc",
    "Vull envit", "Vull truc", "Fora envit", "Fora truc",
    "Passar",
    "Senya 11 bastos", "Senya 10 ors", "Senya as espases",
    "Senya as bastos", "Senya manilla espases", "Senya manilla ors",
    "Senya tres", "Senya as bord", "Senya cegas",
]

ACCIO_TECLA = {
    0: "1", 1: "2", 2: "3",
    3: "e", 4: "t", 5: "v", 6: "v", 7: "f", 8: "f", 9: "p",
}

# Mapeig codi carta -> fitxer imatge
_carta_map = None

def _load_carta_map():
    global _carta_map
    if _carta_map is not None:
        return _carta_map
    json_path = _IMG_DIR / "carta_to_image.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            _carta_map = json.load(f)
    else:
        _carta_map = {}
    return _carta_map


def _img_path(codi: str) -> Path:
    m = _load_carta_map()
    fname = m.get(codi, f"{codi}.png")
    return _IMG_DIR / fname


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BG = "#1a1d23"
BG_TABLE = "#1a3d1c"
BG_PANEL = "#2d3139"
BG_BTN = "#5c4d7a"
BG_BTN_CARD = "#4a3d5c"
FG = "#e0e0e0"
FG_DIM = "#8ab88a"
FG_ACCENT = "#b0a0c0"
CARD_W, CARD_H = 72, 100


class VistaDesktop:
    """Vista Tkinter per al Truc."""

    def __init__(self):
        self._root = None
        self._tk_thread = None
        self._result = None
        self._event = threading.Event()
        self._card_images = {}
        self._action_log = []
        self._config = {}  # s'omple des de mostrar_configuracio (config del joc)
        self._hist_cartes_visible = []  # màxim de cartes vist aquesta mà (per no perdre-les entre torns)
        self._event_taula_done = threading.Event()

    # ------------------------------------------------------------------
    # Tk lifecycle
    # ------------------------------------------------------------------
    def _ensure_tk(self):
        if self._root is not None:
            return
        ready = threading.Event()

        def run():
            self._root = tk.Tk()
            self._root.title("Truc")
            self._root.configure(bg=BG)
            # Mida inicial i límits perquè la taula i el panell d'accions es vegin bé
            self._root.geometry("960x720")
            self._root.minsize(800, 600)
            self._root.protocol("WM_DELETE_WINDOW", self._on_close)
            ready.set()
            self._root.mainloop()

        self._tk_thread = threading.Thread(target=run, daemon=True)
        self._tk_thread.start()
        ready.wait()

    def _on_close(self):
        self._result = None
        self._event.set()
        if self._root:
            self._root.destroy()
            self._root = None

    def _clear(self):
        if self._root is None:
            return
        for w in self._root.winfo_children():
            w.destroy()

    def _wait(self):
        """Bloqueja fins que el fil Tk posa un resultat."""
        self._event.clear()
        self._event.wait()
        return self._result

    def _schedule(self, fn, *args):
        if self._root:
            self._root.after(0, fn, *args)

    # ------------------------------------------------------------------
    # Imatges
    # ------------------------------------------------------------------
    def _get_card_image(self, codi: str):
        if codi in self._card_images:
            return self._card_images[codi]
        p = _img_path(codi)
        if p.exists():
            try:
                img = tk.PhotoImage(file=str(p))
                w, h = img.width(), img.height()
                if w > CARD_W or h > CARD_H:
                    sx = max(1, w // CARD_W)
                    sy = max(1, h // CARD_H)
                    img = img.subsample(sx, sy)
                self._card_images[codi] = img
                return img
            except tk.TclError:
                pass
        self._card_images[codi] = None
        return None

    def _get_dors_image(self):
        """Imatge del revers de la carta (per la mà del rival)."""
        return self._get_card_image("dors")

    # ------------------------------------------------------------------
    # demanar_config
    # ------------------------------------------------------------------
    def demanar_config(self) -> dict:
        self._ensure_tk()
        self._schedule(self._build_config_ui)
        return self._wait()

    def _build_config_ui(self):
        self._clear()
        f = tk.Frame(self._root, bg=BG_PANEL, padx=20, pady=20)
        f.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(f, text="Configuració del joc", font=("", 16, "bold"),
                 bg=BG_PANEL, fg=FG).pack(pady=(0, 12))

        row = tk.Frame(f, bg=BG_PANEL)
        row.pack(fill="x", pady=4)
        tk.Label(row, text="Nombre de jugadors", bg=BG_PANEL, fg=FG).pack(side="left")
        num_var = tk.StringVar(value="2")
        cb = ttk.Combobox(row, textvariable=num_var, values=["2", "4"],
                          width=4, state="readonly")
        cb.pack(side="right")

        tipus_frame = tk.Frame(f, bg=BG_PANEL)
        tipus_frame.pack(fill="x", pady=4)
        tipus_vars = {}

        def refresh_tipus(*_):
            for w in tipus_frame.winfo_children():
                w.destroy()
            n = int(num_var.get())
            for i in range(n):
                r = tk.Frame(tipus_frame, bg=BG_PANEL)
                r.pack(fill="x", pady=1)
                tk.Label(r, text=f"J{i}:", bg=BG_PANEL, fg=FG, width=4).pack(side="left")
                v = tk.StringVar(value="0")
                tipus_vars[i] = v
                ttk.Combobox(r, textvariable=v, values=["0 - Humà", "1 - Aleatori"],
                             width=14, state="readonly").pack(side="right")
                v.set("0 - Humà")

        num_var.trace_add("write", refresh_tipus)
        refresh_tipus()

        row2 = tk.Frame(f, bg=BG_PANEL)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Cartes per jugador", bg=BG_PANEL, fg=FG).pack(side="left")
        cartes_var = tk.StringVar(value="3")
        tk.Spinbox(row2, from_=1, to=12, textvariable=cartes_var, width=4,
                   bg=BG, fg=FG, buttonbackground=BG_PANEL).pack(side="right")

        senyes_var = tk.BooleanVar(value=False)
        senyes_chk = tk.Checkbutton(f, text="Activar senyes", variable=senyes_var,
                                     bg=BG_PANEL, fg=FG, selectcolor=BG,
                                     activebackground=BG_PANEL, activeforeground=FG)
        senyes_chk.pack(pady=4)

        row3 = tk.Frame(f, bg=BG_PANEL)
        row3.pack(fill="x", pady=4)
        tk.Label(row3, text="Puntuació final", bg=BG_PANEL, fg=FG).pack(side="left")
        punt_var = tk.StringVar(value="24")
        tk.Spinbox(row3, from_=1, to=100, textvariable=punt_var, width=4,
                   bg=BG, fg=FG, buttonbackground=BG_PANEL).pack(side="right")

        def submit():
            n = int(num_var.get())
            tp = {}
            for i in range(n):
                val = tipus_vars.get(i)
                tp[i] = int(val.get()[0]) if val else 0
            self._result = {
                "num_jugadors": n,
                "cartes_jugador": int(cartes_var.get() or 3),
                "senyes": senyes_var.get() if n > 2 else False,
                "puntuacio_final": int(punt_var.get() or 24),
                "tipus_jugadors": tp,
            }
            self._event.set()

        tk.Button(f, text="Iniciar partida", command=submit,
                  bg=BG_BTN, fg="white", relief="flat", padx=16, pady=6,
                  font=("", 11, "bold")).pack(pady=(12, 0))

    # ------------------------------------------------------------------
    # escollir_accio
    # ------------------------------------------------------------------
    def escollir_accio(self, accions_legals: list, state: dict) -> int:
        self._schedule(self._build_game_ui, accions_legals, state, False)
        return self._wait()

    def actualitzar_taula(self, state: dict) -> None:
        """Actualitza la taula amb l'estat sense demanar acció (per veure la carta jugada)."""
        state_copy = {**state, "hist_cartes": list(state.get("hist_cartes", []))}
        self._event_taula_done.clear()

        def _fer_actualitzacio():
            self._build_game_ui([], state_copy, True)
            self._event_taula_done.set()

        self._schedule(_fer_actualitzacio)
        self._event_taula_done.wait(timeout=2.0)

    def _build_game_ui(self, accions_legals, state, readonly=False):
        self._clear()
        root = self._root

        fase_torn = state.get("fase_torn", 2)
        id_jugador = state.get("id_jugador", 0)
        ma_jugador = state.get("ma_jugador", [])
        puntuacio = state.get("puntuacio", [0, 0])
        comptador_ronda = state.get("comptador_ronda", 0)
        ma = state.get("ma", 0)
        hist_cartes = state.get("hist_cartes", [])
        hist_senyes = state.get("hist_senyes", [])
        estat_truc = state.get("estat_truc", {})
        estat_envit = state.get("estat_envit", {})

        num_jugadors = self._config.get("num_jugadors", 2)
        dors_img = self._get_dors_image()
        is_fase_cartes = fase_torn == 2

        # Panell d'accions (o "Esperant..." en mode només lectura)
        BOTTOM_BAR_H = 64
        bottom_bar = tk.Frame(root, bg=BG_PANEL, padx=10, pady=8, height=BOTTOM_BAR_H)
        bottom_bar.pack(side="bottom", fill="x")
        bottom_bar.pack_propagate(False)
        key_to_action = {}
        if readonly:
            tk.Label(bottom_bar, text="Esperant torn del rival...", bg=BG_PANEL, fg="#888",
                     font=("", 11)).pack(pady=12)
        else:
            btn_frame = tk.Frame(bottom_bar, bg=BG_PANEL)
            btn_frame.pack(fill="x")
            for action_id in accions_legals:
                label = ACCIONS_CAT[action_id] if action_id < len(ACCIONS_CAT) else f"Acció {action_id}"
                tecla = ACCIO_TECLA.get(action_id, "")
                is_play = action_id <= 2
                if is_play and action_id < len(ma_jugador):
                    label = f"Carta {action_id + 1} ({ma_jugador[action_id]})"
                    tecla = str(action_id + 1)
                b = tk.Button(
                    btn_frame, text=label + (f"  [{tecla.upper()}]" if tecla else ""),
                    bg=BG_BTN_CARD if is_play else BG_BTN, fg="white",
                    relief="flat", padx=12, pady=8, font=("", 10, "bold"),
                    cursor="hand2", command=lambda a=action_id: self._submit_action(a),
                )
                b.pack(side="left", padx=4)
                if tecla:
                    key_to_action[tecla] = action_id
            if is_fase_cartes:
                tk.Label(bottom_bar, text="Clica una carta o prem 1, 2, 3",
                         bg=BG_PANEL, fg="#888", font=("", 9)).pack(pady=(4, 0))

        # Contingut principal (taula + registre) omple la resta
        content = tk.Frame(root, bg=BG)
        content.pack(side="top", fill="both", expand=True)

        table_frame = tk.Frame(content, bg=BG_TABLE, padx=12, pady=12)
        table_frame.pack(side="left", fill="both", expand=True)

        log_panel = tk.Frame(content, bg=BG_PANEL, width=220, padx=8, pady=10)
        log_panel.pack(side="right", fill="y")
        log_panel.pack_propagate(False)

        # --- Taula: info, columnes d'altres jugadors, cartes jugades, puntuació, jugador actual ---
        info_row = tk.Frame(table_frame, bg=BG_TABLE)
        info_row.pack(pady=(0, 6))
        tk.Label(info_row, text=f"Ronda: {comptador_ronda + 1}  ·  ", bg=BG_TABLE, fg=FG_DIM,
                 font=("", 11)).pack(side="left")
        tk.Label(info_row, text=f"Mà: ", bg=BG_TABLE, fg=FG_DIM, font=("", 11)).pack(side="left")
        tk.Label(info_row, text=f"Jugador {ma}", bg=BG_TABLE, fg="#e06060", font=("", 11, "bold")).pack(side="left")
        truc_lvl = estat_truc.get("level", 1)
        envit_lvl = estat_envit.get("level", 0)
        if truc_lvl > 1:
            tk.Label(info_row, text=f"  ·  Truc: {truc_lvl}", bg=BG_TABLE, fg=FG_DIM,
                     font=("", 11)).pack(side="left")
        if envit_lvl > 0:
            tk.Label(info_row, text=f"  ·  Envit: {envit_lvl}", bg=BG_TABLE, fg=FG_DIM,
                     font=("", 11)).pack(side="left")

        # Altres jugadors en columnes: 1v1 = una columna (P1); 2v2 = tres columnes (P3 | company P2 | P1)
        def cards_count(pid):
            n = sum(1 for (p, _) in hist_cartes if p == pid)
            return max(0, 3 - n)

        def columna_altre(parent, pid, etiqueta, pack_side="left"):
            f = tk.Frame(parent, bg=BG_TABLE, padx=8, pady=4)
            f.pack(side=pack_side, fill="y")
            tk.Label(f, text=etiqueta, bg=BG_TABLE, fg=FG_DIM, font=("", 10)).pack()
            hand_f = tk.Frame(f, bg=BG_TABLE)
            hand_f.pack(pady=2)
            for _ in range(cards_count(pid)):
                if dors_img:
                    tk.Label(hand_f, image=dors_img, bg=BG_TABLE).pack(side="left", padx=2)
                else:
                    tk.Label(hand_f, text="?", bg="#2d5a30", fg=FG_DIM,
                             width=6, height=4, relief="ridge").pack(side="left", padx=2)
            return f

        # 1v1: una fila amb el rival. 2v2: company amunt (centrat), rivals avall (esq i dreta)
        if num_jugadors == 2:
            altres_row = tk.Frame(table_frame, bg=BG_TABLE)
            altres_row.pack(pady=(0, 6))
            columna_altre(altres_row, 1 - id_jugador, f"Rival (J{1 - id_jugador})")
        else:
            p_esq = (id_jugador + 3) % 4
            p_centre = (id_jugador + 2) % 4
            p_dreta = (id_jugador + 1) % 4
            fila_company = tk.Frame(table_frame, bg=BG_TABLE)
            fila_company.pack(pady=(0, 2))
            col_company = tk.Frame(fila_company, bg=BG_TABLE)
            col_company.pack()
            columna_altre(col_company, p_centre, f"Company (J{p_centre})")
            fila_rivals = tk.Frame(table_frame, bg=BG_TABLE)
            fila_rivals.pack(pady=(0, 6))
            columna_altre(fila_rivals, p_esq, f"Rival (J{p_esq})", pack_side="left")
            columna_altre(fila_rivals, p_dreta, f"Rival (J{p_dreta})", pack_side="right")

        # Centre: cartes jugades — una columna per jugador, apilades
        if comptador_ronda == 0 and len(hist_cartes) == 0:
            self._hist_cartes_visible = []
        elif len(hist_cartes) >= len(self._hist_cartes_visible):
            self._hist_cartes_visible = list(hist_cartes)
        cartes_per_mostrar = list(hist_cartes) if readonly else self._hist_cartes_visible

        piles = {}
        for pid, card in cartes_per_mostrar:
            piles.setdefault(pid, []).append(card)

        center_box = tk.Frame(table_frame, bg="#0d2a0e", relief="ridge", bd=2, padx=8, pady=6)
        center_box.pack(pady=6, fill="x")
        tk.Label(center_box, text="Cartes jugades", bg="#0d2a0e", fg="#7ab87a",
                 font=("", 10, "bold")).pack(pady=(0, 4))

        LBL_H = 18
        ordered_pids = [id_jugador] + [p for p in range(num_jugadors) if p != id_jugador]

        if not cartes_per_mostrar:
            row_labels = tk.Frame(center_box, bg="#0d2a0e")
            row_labels.pack(fill="x")
            for pid in ordered_pids:
                tk.Label(row_labels, text=f"J{pid}", bg="#0d2a0e", fg="#9aca9a",
                         font=("", 9, "bold")).pack(side="left", expand=True)
            tk.Label(center_box, text="—", bg="#0d2a0e", fg=FG_DIM, font=("", 11)).pack()
        else:
            STEP_Y = 25
            max_cards = max(len(v) for v in piles.values()) if piles else 1
            canvas_h = LBL_H + (max_cards - 1) * STEP_Y + CARD_H + 4
            canvas = tk.Canvas(center_box, bg="#0d2a0e",
                               highlightthickness=0, height=canvas_h)
            canvas.pack(fill="x")

            def _place_piles(event=None):
                w = canvas.winfo_width()
                if w < 10:
                    w = 400
                slot_w = w // num_jugadors
                for wid in canvas.find_all():
                    canvas.delete(wid)
                for col, pid in enumerate(ordered_pids):
                    cx = slot_w * col + slot_w // 2
                    cards = piles.get(pid, [])
                    lbl = tk.Label(canvas, text=f"J{pid}", bg="#0d2a0e",
                                   fg="#9aca9a", font=("", 9, "bold"))
                    canvas.create_window(cx, 0, window=lbl, anchor="n")
                    for j, card in enumerate(cards):
                        cf = tk.Frame(canvas, bg="#0d2a0e")
                        img = self._get_card_image(card)
                        if img:
                            tk.Label(cf, image=img, bg="#0d2a0e").pack()
                        else:
                            tk.Label(cf, text=card, bg="#2d5a30", fg=FG,
                                     width=8, height=5, relief="ridge").pack()
                        canvas.create_window(cx, LBL_H + j * STEP_Y, window=cf, anchor="n")

            canvas.bind("<Configure>", _place_piles)
            canvas.after(10, _place_piles)

        # Senyes d'aquesta mà (qui ha fet quina senya)
        if self._config.get("senyes", False) and hist_senyes:
            senyes_box = tk.Frame(table_frame, bg="#1a2d1a", relief="ridge", bd=2, padx=8, pady=4)
            senyes_box.pack(pady=4, fill="x")
            tk.Label(senyes_box, text="Senyes d'aquesta mà", bg="#1a2d1a", fg="#7ab87a",
                     font=("", 10, "bold")).pack(pady=(0, 4))
            for pid, action_str in hist_senyes:
                try:
                    idx = ACTION_LIST.index(action_str)
                    nom_senya = ACCIONS_CAT[idx] if idx < len(ACCIONS_CAT) else action_str
                except (ValueError, IndexError):
                    nom_senya = action_str
                tk.Label(senyes_box, text=f"Jugador {pid}: {nom_senya}", bg="#1a2d1a", fg=FG,
                         font=("", 10)).pack(anchor="w")

        # Puntuació
        tk.Label(table_frame, text=f"E0: {puntuacio[0]}    E1: {puntuacio[1]}",
                 bg=BG_TABLE, fg=FG, font=("", 14, "bold")).pack(pady=8)

        # Jugador actual (baix): la seva mà a la vista
        tk.Label(table_frame, text=f"Jugador actual (J{id_jugador})",
                 bg=BG_TABLE, fg=FG_DIM, font=("", 11)).pack(pady=(6, 4))

        hand_frame = tk.Frame(table_frame, bg=BG_TABLE)
        hand_frame.pack(pady=4)

        for i, codi in enumerate(ma_jugador):
            card_clickable = is_fase_cartes and i in accions_legals
            img = self._get_card_image(codi)
            if img:
                btn = tk.Button(
                    hand_frame, image=img, bg=BG_TABLE,
                    activebackground="#2d5a30", relief="flat", bd=0,
                    state="normal" if card_clickable else "disabled",
                    command=lambda idx=i: self._submit_action(idx),
                )
            else:
                btn = tk.Button(
                    hand_frame, text=codi, width=8, height=5,
                    bg="#2d5a30" if card_clickable else "#1e3a1e",
                    fg=FG, relief="ridge",
                    state="normal" if card_clickable else "disabled",
                    command=lambda idx=i: self._submit_action(idx),
                )
            btn.pack(side="left", padx=4)

        # --- Columna dreta: Registre d'accions (ben visible) ---
        tk.Label(log_panel, text="Registre d'accions", bg=BG_PANEL, fg=FG,
                 font=("", 12, "bold")).pack(anchor="w", pady=(0, 8))
        log_list = tk.Frame(log_panel, bg=BG_PANEL)
        log_list.pack(fill="both", expand=True)
        log_scroll = tk.Scrollbar(log_list, bg=BG_PANEL)
        log_scroll.pack(side="right", fill="y")
        log_text = tk.Text(log_list, height=14, width=26, wrap="word", bg="#252a32", fg="#d0e0d0",
                           font=("", 10), state="disabled", relief="flat", padx=6, pady=6,
                           yscrollcommand=log_scroll.set)
        log_text.pack(side="left", fill="both", expand=True)
        log_scroll.config(command=log_text.yview)
        if self._action_log:
            for pid, nom, act in self._action_log:
                log_text.config(state="normal")
                try:
                    idx = ACTION_LIST.index(act)
                    act_visible = ACCIONS_CAT[idx] if idx < len(ACCIONS_CAT) else act
                except (ValueError, IndexError):
                    act_visible = act
                log_text.insert("end", f"Jugador {pid} ({nom}): {act_visible}\n")
                log_text.config(state="disabled")
            log_text.see("end")
        else:
            log_text.config(state="normal")
            log_text.insert("end", "— Cap acció encara —")
            log_text.config(state="disabled")

        def on_key(e):
            k = e.char.lower()
            if k in key_to_action:
                self._submit_action(key_to_action[k])

        root.bind("<Key>", on_key)

    def _submit_action(self, action_id):
        self._root.unbind("<Key>")
        self._result = action_id
        self._event.set()

    # ------------------------------------------------------------------
    # mostrar_*
    # ------------------------------------------------------------------
    def mostrar_inici(self) -> None:
        self._schedule(self._show_message, "Iniciant partida...")

    def mostrar_configuracio(self, num_jugadors, tipus_jugadors, senyes,
                             cartes_jugador, puntuacio_final) -> None:
        self._config = {
            "num_jugadors": num_jugadors,
            "tipus_jugadors": tipus_jugadors,
            "senyes": senyes,
            "cartes_jugador": cartes_jugador,
            "puntuacio_final": puntuacio_final,
        }

    def mostrar_barrejant(self) -> None:
        self._action_log = []
        self._schedule(self._show_message, "Barrejant cartes...")

    def mostrar_taula(self, hist_cartes: list, state: dict | None = None) -> None:
        if state is not None:
            self.actualitzar_taula(state)

    def mostrar_accio_executada(self, player_id, nom_jugador, action_name) -> None:
        self._action_log.append((player_id, nom_jugador, action_name))

    def mostrar_fi_partida(self, score: list, payoffs: list) -> None:
        self._schedule(self._build_game_over, score, payoffs)

    def _build_game_over(self, score, payoffs):
        self._clear()
        f = tk.Frame(self._root, bg=BG_PANEL, padx=30, pady=30)
        f.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(f, text="Joc acabat!", font=("", 18, "bold"),
                 bg=BG_PANEL, fg=FG).pack(pady=(0, 12))
        tk.Label(f, text=f"E0: {score[0]}  —  E1: {score[1]}",
                 font=("", 15), bg=BG_PANEL, fg=FG).pack()
        tk.Label(f, text=f"Payoffs: {payoffs}",
                 font=("", 11), bg=BG_PANEL, fg="#aaa").pack(pady=(4, 0))

    def demanar_repetir(self) -> bool:
        self._schedule(self._build_repeat_ui)
        return self._wait()

    def _build_repeat_ui(self):
        self._clear()
        f = tk.Frame(self._root, bg=BG_PANEL, padx=30, pady=30)
        f.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(f, text="Vols tornar a jugar?", font=("", 14, "bold"),
                 bg=BG_PANEL, fg=FG).pack(pady=(0, 16))
        bf = tk.Frame(f, bg=BG_PANEL)
        bf.pack()
        tk.Button(bf, text="Tornar a jugar", bg=BG_BTN, fg="white",
                  relief="flat", padx=14, pady=6, font=("", 11, "bold"),
                  command=lambda: self._set_repeat(True)).pack(side="left", padx=6)
        tk.Button(bf, text="Sortir", bg="#3d424a", fg=FG,
                  relief="flat", padx=14, pady=6,
                  command=lambda: self._set_repeat(False)).pack(side="left", padx=6)

    def _set_repeat(self, val):
        self._result = val
        self._event.set()

    def mostrar_sortint(self) -> None:
        if self._root:
            self._schedule(self._root.destroy)

    def _show_message(self, text):
        self._clear()
        tk.Label(self._root, text=text, bg=BG, fg=FG,
                 font=("", 13)).place(relx=0.5, rely=0.5, anchor="center")
        self._root.update_idletasks()
