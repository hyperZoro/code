import numpy as np
from scipy.stats import norm
from scipy.special import erf

def verify_correction_logic():
    # Constants
    phi = norm.pdf
    Phi = norm.cdf
    z65 = norm.ppf(0.65)
    z80 = norm.ppf(0.80)
    z95 = norm.ppf(0.95)
    
    # 1. Verify Delta Q (Integral of phi^2)
    # Formula in correction doc: 1/(4*sqrt(pi)) * (erf(b) - erf(a))
    def delta_q_correction(a, b):
        factor = 1.0 / (4.0 * np.sqrt(np.pi))
        return factor * (erf(b) - erf(a))
        
    dq1 = delta_q_correction(z65, z80)
    dq2 = delta_q_correction(z80, z95)
    
    print(f"Delta Q1 (Correction Doc): {dq1:.5f}")
    print(f"Delta Q2 (Correction Doc): {dq2:.5f}")
    
    # 2. Verify K values (Slope Adjustments)
    # K1 = dQ1 - 0.15 * phi(z80)
    k1 = dq1 - 0.15 * phi(z80)
    
    # K2 = dQ2 - 0.15 * phi(z95)
    k2 = dq2 - 0.15 * phi(z95)
    
    print(f"K1 (Correction Doc): {k1:.5f}")
    print(f"K2 (Correction Doc): {k2:.5f}")
    
    # 3. Verify Coefficients
    denom = phi(0)
    
    # Base Weights
    w_base1 = (phi(0) - phi(z80)) / denom
    w_base2 = (phi(z80) - phi(z95)) / denom
    w_base3 = phi(z95) / denom
    
    print(f"W_base1: {w_base1:.4f}")
    print(f"W_base2: {w_base2:.4f}")
    print(f"W_base3: {w_base3:.4f}")
    
    # Adjustments
    adj1 = k1 / (0.15 * denom)
    adj2 = k2 / (0.15 * denom)
    
    print(f"Adj1: {adj1:.4f}")
    print(f"Adj2: {adj2:.4f}")
    
    # Final Weights
    w65 = w_base1 - adj1
    w80 = w_base2 + adj1 - adj2
    w95 = w_base3 + adj2
    
    print("-" * 20)
    print(f"Final Weights (Correction Logic):")
    print(f"w65: {w65:.4%}")
    print(f"w80: {w80:.4%}")
    print(f"w95: {w95:.4%}")
    print(f"Sum: {w65+w80+w95:.4%}")

if __name__ == "__main__":
    verify_correction_logic()
