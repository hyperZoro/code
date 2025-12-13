import numpy as np
from scipy.stats import norm

def calculate_terms():
    phi = norm.pdf
    Phi = norm.cdf
    q = norm.ppf
    
    x50 = 0
    x65 = q(0.65)
    x80 = q(0.80)
    x95 = q(0.95)
    
    # Integral of x*phi(x) from a to b is phi(a) - phi(b)
    def int_x_phi(a, b):
        val_b = 0 if b == np.inf else phi(b)
        val_a = phi(a)
        return val_a - val_b
    
    # Integral of Phi(x)*x*phi(x)
    # Antiderivative: -Phi(x)phi(x) + 1/(2*sqrt(pi)) * Phi(x*sqrt(2))
    def antideriv_Phi_x_phi(x):
        if x == np.inf:
            # lim x->inf of -Phi(x)phi(x) is 0
            # lim x->inf of Phi(x*sqrt(2)) is 1
            return 1.0 / (2 * np.sqrt(np.pi))
        return -Phi(x)*phi(x) + (1.0 / (2 * np.sqrt(np.pi))) * Phi(x * np.sqrt(2))
        
    def int_Phi_x_phi(a, b):
        return antideriv_Phi_x_phi(b) - antideriv_Phi_x_phi(a)

    # Term 1
    T1 = int_x_phi(x50, x65)
    
    # Region 2 integrals
    I_x_phi_2 = int_x_phi(x65, x80)
    I_Phi_x_phi_2 = int_Phi_x_phi(x65, x80)
    
    # Term 2
    T2 = (1/0.15) * (0.80 * I_x_phi_2 - I_Phi_x_phi_2)
    
    # Term 3
    T3 = (1/0.15) * (I_Phi_x_phi_2 - 0.65 * I_x_phi_2)
    
    # Region 3 integrals
    I_x_phi_3 = int_x_phi(x80, x95)
    I_Phi_x_phi_3 = int_Phi_x_phi(x80, x95)
    
    # Term 4
    T4 = (1/0.15) * (0.95 * I_x_phi_3 - I_Phi_x_phi_3)
    
    # Term 5
    T5 = (1/0.15) * (I_Phi_x_phi_3 - 0.80 * I_x_phi_3)
    
    # Term 6
    T6 = int_x_phi(x95, np.inf)
    
    D = phi(0)
    
    print(f"T1: {T1:.5f}")
    print(f"T2: {T2:.5f}")
    print(f"T3: {T3:.5f}")
    print(f"T4: {T4:.5f}")
    print(f"T5: {T5:.5f}")
    print(f"T6: {T6:.5f}")
    
    print(f"Omega65 (T1+T2): {T1+T2:.5f}")
    print(f"Omega80 (T3+T4): {T3+T4:.5f}")
    print(f"Omega95 (T5+T6): {T5+T6:.5f}")
    print(f"Total: {T1+T2+T3+T4+T5+T6:.5f}")
    print(f"Denominator: {D:.5f}")

if __name__ == "__main__":
    calculate_terms()
