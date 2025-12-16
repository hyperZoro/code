
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def find_theoretical_peak():
    # Parameters provided by user
    f65 = 2.0
    f80 = 1.5
    f95 = 1.0
    
    # Define interpolation slopes K = df / du
    K2 = (f80 - f65) / 0.15 # Region 2
    K3 = (f95 - f80) / 0.15 # Region 3
    
    # Quantile boundaries
    x65 = norm.ppf(0.65)
    x80 = norm.ppf(0.80)
    x95 = norm.ppf(0.95)
    
    # Define functions for density calculation
    def get_density(x_vals, f_start, u_start, K):
        densities = []
        y_vals = []
        for x in x_vals:
            u = norm.cdf(x)
            phi = norm.pdf(x)
            
            # S(u)
            s = f_start + K * (u - u_start)
            
            # g(x) = x * S(Phi(x))
            y = x * s
            
            # g'(x) = S(u) + x * S'(u) * u'(x)
            #       = s + x * K * phi
            g_prime = s + x * K * phi
            
            # density = phi(x) / |g'(x)|
            density = phi / abs(g_prime)
            
            densities.append(density)
            y_vals.append(y)
            
        return np.array(y_vals), np.array(densities)

    print(f"Searching for density peak in Region 2 and 3...")
    
    # --- Analyze Region 2 (65% -> 80%) ---
    x_r2 = np.linspace(x65, x80, 1000)
    y_r2, d_r2 = get_density(x_r2, f65, 0.65, K2)
    max_d2_idx = np.argmax(d_r2)
    print(f"Region 2 Peak Density: {d_r2[max_d2_idx]:.4f} at Y={y_r2[max_d2_idx]:.4f} (X={x_r2[max_d2_idx]:.4f})")
    
    # --- Analyze Region 3 (80% -> 95%) ---
    x_r3 = np.linspace(x80, x95, 1000)
    y_r3, d_r3 = get_density(x_r3, f80, 0.80, K3)
    max_d3_idx = np.argmax(d_r3)
    print(f"Region 3 Peak Density: {d_r3[max_d3_idx]:.4f} at Y={y_r3[max_d3_idx]:.4f} (X={x_r3[max_d3_idx]:.4f})")
    
    # --- Global Max ---
    if d_r2[max_d2_idx] > d_r3[max_d3_idx]:
        print(f"\nGlobal Theoretical Peak is likely in Region 2 at Y ≈ {y_r2[max_d2_idx]:.4f}")
    else:
        print(f"\nGlobal Theoretical Peak is likely in Region 3 at Y ≈ {y_r3[max_d3_idx]:.4f}")

if __name__ == "__main__":
    find_theoretical_peak()
