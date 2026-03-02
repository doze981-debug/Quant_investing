import numpy as np
import scipy.optimize as sco
import pandas as pd

"""
TITOLO: Confronto Ottimizzazione Mean-Variance vs Safety First (Roy's Criterion)

DESCRIZIONE:
Questo script simula un portafoglio di 3 asset e calcola l'allocazione ottimale 
secondo due diverse funzioni di utilità descritte nella teoria economica:

1. Mean-Variance (Sharpe Ratio):
   - Obiettivo: Massimizzare il rendimento aggiustato per la volatilità totale.
   - Formula: (Rendimento_Portafoglio - Risk_Free_Rate) / Volatilità_Portafoglio.
   - Ipotesi: L'investitore è avverso alla varianza (volatilità) simmetrica.

2. Roy's Safety First Criterion (SFRatio):
   - Obiettivo: Minimizzare la probabilità che il rendimento del portafoglio scenda 
     sotto una soglia minima accettabile (MAR - Minimum Acceptable Return).
   - Formula: (Rendimento_Portafoglio - MAR) / Volatilità_Portafoglio.
   - Ipotesi: L'investitore teme il "Disastro" (rendimento < MAR) più della volatilità generica.
   - Nota: Se i rendimenti sono distribuiti normalmente, massimizzare il rapporto SF equivale 
     a minimizzare la probabilità P(Rp < MAR).

INPUT:
- Rendimenti attesi e matrice di covarianza simulati per 3 asset (Bond, Equity, Gold).
- Risk Free Rate (per Sharpe): 1%.
- Minimum Acceptable Return (per Safety First): 2% (Soglia di sopravvivenza).

OUTPUT:
- Pesi ottimali per i due portafogli e statistiche di rischio/rendimento a confronto.
"""

# --- 1. CONFIGURAZIONE DATI (SIMULAZIONE) ---
np.random.seed(42)
assets = ['Bond Gov', 'Equity World', 'Gold']
num_assets = len(assets)

# Rendimenti attesi annuali (vettore delle medie)
# Bond basso rischio, Equity alto rendimento, Gold decorrelante
mu = np.array([0.03, 0.08, 0.05]) 

# Matrice di Covarianza (semplificata)
cov_matrix = np.array([
    [0.002, 0.001, 0.000],  # Bond: bassa var, bassa corr
    [0.001, 0.040, 0.005],  # Equity: alta var
    [0.000, 0.005, 0.025]   # Gold: media var, bassa corr
])

# Parametri Utilità
risk_free_rate = 0.01  # 1% per Sharpe
target_mar = 0.02      # 2% "Soglia Disastro" per Safety First

# --- 2. FUNZIONI DI CALCOLO ---

def portfolio_stats(weights, mu, cov_matrix):
    """Calcola rendimento e volatilità del portafoglio dato un array di pesi."""
    returns = np.sum(mu * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

def neg_sharpe_ratio(weights, mu, cov_matrix, rf):
    """Funzione obiettivo da minimizzare per Mean-Variance (-Sharpe)."""
    p_ret, p_vol = portfolio_stats(weights, mu, cov_matrix)
    return - (p_ret - rf) / p_vol

def neg_safety_first_ratio(weights, mu, cov_matrix, mar):
    """Funzione obiettivo da minimizzare per Safety First (-SFRatio)."""
    p_ret, p_vol = portfolio_stats(weights, mu, cov_matrix)
    # Roy's Ratio = (E[Rp] - MAR) / Sigma_p
    return - (p_ret - mar) / p_vol

# --- 3. OTTIMIZZAZIONE ---

# Vincoli: somma pesi = 1, pesi tra 0 e 1 (no short selling)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))
init_guess = num_assets * [1. / num_assets,]

# A. Ottimizzazione Mean-Variance (Sharpe)
opt_sharpe = sco.minimize(neg_sharpe_ratio, init_guess, args=(mu, cov_matrix, risk_free_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints)

# B. Ottimizzazione Safety First (Roy)
opt_safety = sco.minimize(neg_safety_first_ratio, init_guess, args=(mu, cov_matrix, target_mar),
                          method='SLSQP', bounds=bounds, constraints=constraints)

# --- 4. PRESENTAZIONE RISULTATI ---

def print_result(title, result_obj, mu, cov, benchmark_val, benchmark_name):
    weights = result_obj.x
    ret, vol = portfolio_stats(weights, mu, cov)
    print(f"--- {title} ---")
    print(f"Allocazione:")
    for i, asset in enumerate(assets):
        print(f"  {asset}: {weights[i]:.2%}")
    print(f"Rendimento Atteso: {ret:.2%}")
    print(f"Volatilità Attesa: {vol:.2%}")
    print(f"{benchmark_name}: {-result_obj.fun:.4f}\n")

print("\n=== RISULTATI OTTIMIZZAZIONE ===\n")

print_result("PORTAFOGLIO MEAN-VARIANCE (Max Sharpe)", 
             opt_sharpe, mu, cov_matrix, risk_free_rate, "Sharpe Ratio")

print_result(f"PORTAFOGLIO SAFETY FIRST (MAR = {target_mar:.1%})", 
             opt_safety, mu, cov_matrix, target_mar, "Roy's SF Ratio")

"""
NOTA TECNICA:
Osservando i risultati, noterai che il portafoglio Safety First tende ad essere 
più conservativo (pesando di più i Bond o diversificando per ridurre la coda sinistra)
se la soglia MAR è vicina al rendimento atteso degli asset rischiosi.
Se il MAR è molto basso, i due portafogli tendono a convergere.
"""