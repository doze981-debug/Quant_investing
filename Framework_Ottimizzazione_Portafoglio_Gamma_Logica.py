"""
FRAMEWORK DI OTTIMIZZAZIONE E ALLOCAZIONE ASSET (LOGICA QUANTITATIVA)
====================================================================

QUANDO UTILIZZARLO:
-------------------
1. In fase di PRE-ESECUZIONE: Dopo aver generato un segnale di trading, per 
   determinare l'esatta size (w*) in base alla volatilità corrente.
2. In fase di BACKTESTING: Per valutare se una strategia è "razionale" dal punto
   di vista dell'utilità (Equivalente Certo) e non solo del profitto lordo.
3. In fase di MONITORAGGIO: Per effettuare il reverse engineering dell'avversione
   al rischio del mercato e identificare fasi di compiacenza (Gamma basso).

COSA FA:
--------
1. Calcola il peso ottimale (Optimal Weight) di un asset rischioso minimizzando
   la funzione di penalità del rischio rispetto alla propensione dell'investitore.
2. Calcola l'Equivalente Certo (CE) utilizzando due metodologie:
   - Media-Varianza: Veloce, assume distribuzioni normali (Gaussiane).
   - CRRA (Power Utility): Avanzata, penalizza asimmetria negativa e tail risk.
3. Quantifica il "Premio per l'Illiquidità" o il "Costo della non-diversificazione"
   tramite il differenziale di utilità.

FASE NEL FLUSSO DI LAVORO:
--------------------------
Si colloca nella fase di "RISK MANAGEMENT & POSITION SIZING".
Flusso: [Segnale Alpha] -> [STIMA VOLATILITÀ (Sigma)] -> [QUESTO MODULO (w*)] -> [ESECUZIONE]

LOGICA PROFESSIONALE:
---------------------
Il modello agisce come un "freno dinamico". Se la volatilità raddoppia, il peso 
diminuisce del quadrato della volatilità (secondo l'equazione 2.10), garantendo 
che il budget di rischio del portafoglio rimanga costante indipendentemente dal 
regime di mercato.
"""

import numpy as np

class AssetAllocationEngine:
    def __init__(self, gamma: float, rf: float = 0.01):
        """
        :param gamma: Coefficiente di avversione al rischio (tipicamente 2.0 - 5.0)
        :param rf: Tasso Risk-Free annuo (default 1%)
        """
        self.gamma = gamma
        self.rf = rf

    def get_optimal_weight(self, expected_return: float, sigma: float) -> float:
        """
        Implementazione dell'Equazione 2.10: w* = (1/gamma) * (Excess Return / Variance)
        """
        excess_return = expected_return - self.rf
        variance = sigma ** 2
        if variance == 0:
            return 0.0
        return (1 / self.gamma) * (excess_return / variance)

    def certainty_equivalent_mv(self, expected_return: float, sigma: float) -> float:
        """
        Equivalente Certo in approssimazione Media-Varianza.
        """
        return expected_return - 0.5 * self.gamma * (sigma ** 2)

    def certainty_equivalent_crra(self, returns_series: np.ndarray) -> float:
        """
        Equivalente Certo con utilità CRRA (Constant Relative Risk Aversion).
        Cruciale per strategie con asimmetria negativa (es. Short Volatility).
        """
        # Ricchezza relativa (1 + r)
        w = 1 + returns_series
        # Preveniamo valori negativi estremi per il log/potenza
        w = np.clip(w, 1e-6, None)
        
        if self.gamma == 1:
            # Utilità Logaritmica
            expected_utility = np.mean(np.log(w))
            return np.exp(expected_utility) - 1
        else:
            # Utilità di Potenza
            u_series = (w ** (1 - self.gamma)) / (1 - self.gamma)
            expected_utility = np.mean(u_series)
            return ((expected_utility * (1 - self.gamma)) ** (1 / (1 - self.gamma))) - 1

    def implied_gamma(self, observed_weight: float, expected_return: float, sigma: float) -> float:
        """
        Reverse engineering del Gamma (Preferenza Rivelata).
        """
        excess_return = expected_return - self.rf
        variance = sigma ** 2
        if observed_weight == 0 or variance == 0:
            return 0.0
        return (1 / observed_weight) * (excess_return / variance)

# --- ESEMPIO OPERATIVO ---
if __name__ == "__main__":
    # Parametri: Gamma=3 (Moderato), Risk-Free=1%
    engine = AssetAllocationEngine(gamma=3.0, rf=0.01)

    # Dati ipotetici Asset (es. Azionario)
    mu = 0.08    # Rendimento atteso 8%
    std = 0.16   # Volatilità 16%

    # 1. Calcolo del Peso Ottimale
    w_star = engine.get_optimal_weight(mu, std)
    print(f"Allocazione Ottimale (w*): {w_star:.2%}")

    # 2. Verifica dell'impatto di un aumento di Volatilità (es. da 16% a 25%)
    w_star_high_vol = engine.get_optimal_weight(mu, 0.25)
    print(f"Allocazione con Volatilità al 25%: {w_star_high_vol:.2%}")

    # 3. Analisi Utilità (Equivalente Certo)
    ce_val = engine.certainty_equivalent_mv(mu, std)
    print(f"Equivalente Certo (MV): {ce_val:.2%}")