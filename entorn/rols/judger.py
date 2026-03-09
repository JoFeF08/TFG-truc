# Valors de Força de les Cartes (Més alt guanya)
# Jerarquia:
# 1. 11 bastos (B11)
# 2. 10 oros (O10)
# 3. As d'Espases (S1) - La major
# 4. As de Bastos (B1)
# 5. 7 d'Espases (S7) - Manilla
# 6. 7 d'Ors (O7) - Manilla
# 7. Tresos (3)
# 8. Asos "Bords" (Ors/Copes) (O1, C1)
# 9. Figures (10, 11, 12)
# 10. 7 de Bastos/Copes (B7, C7)
# 11. Sisos (6)
# 12. Cinc (5)
# 13. Quatres (4)

FORCA_ONZE_BASTOS = 110
FORCA_DEU_ORS = 109
FORCA_AS_ESPASES = 100
FORCA_AS_BASTOS = 99
FORCA_SET_ESPASES = 98
FORCA_SET_ORS = 97
FORCA_TRES = 90
FORCA_AS_BORD = 70
FORCA_FIGURES = 60 
FORCA_SET_BORD = 55 # 7 B, 7 C
FORCA_SIS = 50
FORCA_CINC = 40
FORCA_QUATRE = 30

class TrucJudger:
    """
    Classe Jutge per al Truc.
    S'encarrega de determinar el guanyador d'envits o de rondes.
    """
    def __init__(self, np_random, n_players=2, n_cartes=3):
        self.np_random = np_random
        self.n_players = n_players
        self.n_cartes = n_cartes
            

    @staticmethod
    def get_equip(player_id):
        return player_id % 2

    def guanyador_ronda(self, cartes):
        """
        Determina el guanyador d'una ronda de cartes.
        Suporta 1vs1 i 2vs2 (equips parells vs senars).
        
        Args:
            cartes: Llista de tuples (player_id, card_str).
            
        Returns:
            int: ID del jugador guanyador.
            None: Si hi ha empat entre equips.
        """
        max_forca = -1
        guanyadors = [] #(player_id, index_in_cartes)

        for i, (pid, carta) in enumerate(cartes):
            forca = self.get_forca_carta(carta)
            if forca > max_forca:
                max_forca = forca
                guanyadors = [(pid, i)]
            elif forca == max_forca:
                guanyadors.append((pid, i))
        
        if len(guanyadors) == 1:
            return guanyadors[0][0]
        
        #gestió d'empats 
        equip_guanyador = self.get_equip(guanyadors[0][0])
        mateix_equip = True

        for pid, _ in guanyadors[1:]:
            if self.get_equip(pid) != equip_guanyador:
                mateix_equip = False
                break
        
        if mateix_equip:
            guanyadors.sort(key=lambda x: x[1])
            return guanyadors[0][0]
        else:
            return None

    @staticmethod
    def get_forca_carta(carta):
        palo = carta[0]
        num = carta[1:]
        
        if palo == 'B' and num == '11': return FORCA_ONZE_BASTOS
        if palo == 'O' and num == '10': return FORCA_DEU_ORS
        
        if palo == 'S' and num == '1': return FORCA_AS_ESPASES
        if palo == 'B' and num == '1': return FORCA_AS_BASTOS
        if palo == 'S' and num == '7': return FORCA_SET_ESPASES
        if palo == 'O' and num == '7': return FORCA_SET_ORS
        
        if num == '3': return FORCA_TRES
        
        if num == '1': return FORCA_AS_BORD # As Ors o Copes
        
        if num in ['10', '11', '12']: return FORCA_FIGURES
        
        if num == '7': return FORCA_SET_BORD # 7 Bastos o Copes
        
        if num == '6': return FORCA_SIS
        if num == '5': return FORCA_CINC
        if num == '4': return FORCA_QUATRE
        
        return 0

    def guanyador_envits(self, mans, p_ma):
        """
        Jutja l'envit per a N jugadors (1vs1 o 2vs2).
        Calcula la puntuació màxima de cada equip i compara.
        
        Args:
            mans: Llista de mans [h0, h1, ..., hn]
            p_ma: ID del jugador que és mà.
            
        Returns:
            int: ID de l'EQUIP guanyador (0 o 1).
        """
        best_score = -1
        equip_guanyador = -1
        
        for i in range(len(mans)):
            pid = (p_ma + i) % len(mans)

            ma = mans[pid]
            score = self.get_envit_ma(ma)
            
            if score > best_score:
                best_score = score
                equip_guanyador = self.get_equip(pid)
                
        return equip_guanyador

    @staticmethod
    def get_envit_carta(carta):
        palo = carta[0]
        num = int(carta[1:])
        
        if palo == 'B' and num == 11: return 8
        if palo == 'O' and num == 10: return 7
        
        if num >= 10: return 0
        return num

    @staticmethod
    def get_envit_ma(ma):
        values = []
        info_cartes = []
        
        has_b11 = False
        has_o10 = False
        
        for carta in ma:
            palo = carta[0]
            num = carta[1:]

            val = TrucJudger.get_envit_carta(carta)
            
            values.append(val)
            info_cartes.append({'val': val, 'palo': palo, 'num': num})
            
            if palo == 'B' and num == '11': has_b11 = True
            if palo == 'O' and num == '10': has_o10 = True
        

        max_score = 0
        
        #casos l'homo i sa madona
        if has_b11:
            other_vals = [c['val'] for c in info_cartes if not (c['palo'] == 'B' and c['num'] == '11')]
            score = 20 + 8 + max(other_vals) if other_vals else 20 + 8
            if score > max_score: max_score = score
                 
        if has_o10:
            other_vals = [c['val'] for c in info_cartes if not (c['palo'] == 'O' and c['num'] == '10')]
            score = 20 + 7 + max(other_vals) if other_vals else 20 + 7
            if score > max_score: max_score = score

        #resta de casos
        palos = {}
        for c in info_cartes:
            if c['palo'] not in palos: palos[c['palo']] = []
            palos[c['palo']].append(c['val'])
            
        for s, vals in palos.items():
            if len(vals) >= 2:
                vals.sort(reverse=True)
                score = 20 + vals[0] + vals[1]
                if score > max_score: max_score = score
        
        #carta alta
        for val in values:
            if val > max_score: max_score = val
            
        return max_score


    def guanyador_ma(self, guanyadors_rondes, ma):
        """
        Determina el guanyador de la mà garantint que funcioni per 3, 5 o N rondes.
        
        Args:
            guanyadors_rondes: Llista de guanyadors de cada ronda jugada (-1 = empat).
            ma: ID del jugador que és mà (per desempats).
            
        Returns:
            int: 0 o 1 = equip guanyador; -1 = mà encara no acabada.
        """
        rw = guanyadors_rondes
        n_jugades = len(rw)
        
        wins = [0, 0]
        for w in rw:
            if w != -1:
                wins[self.get_equip(w)] += 1
            else:
                # Un empat compta com a victòria per a tots dos
                wins[0] += 1
                wins[1] += 1
                
        # majoria absoluta de rondes
        majoria = (self.n_cartes // 2) + 1
        
        if wins[0] >= majoria and wins[1] < majoria: return 0
        if wins[1] >= majoria and wins[0] < majoria: return 1
        
        # si tots dos arriben a la majoria a la vegada
        if (wins[0] >= majoria and wins[1] >= majoria) or n_jugades == self.n_cartes:
            if wins[0] > wins[1]: return 0
            if wins[1] > wins[0]: return 1
            
            #primera ronda no empatada
            for w in rw:
                if w != -1:
                    return self.get_equip(w)
            
            #sino guanya el ma
            return self.get_equip(ma)
            
        return -1

    def guanyador_canto(self, marcador_actual, target_score=24):
        if marcador_actual[0] >= target_score: return 0
        if marcador_actual[1] >= target_score: return 1
        return -1

