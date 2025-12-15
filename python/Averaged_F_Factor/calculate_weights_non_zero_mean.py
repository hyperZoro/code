
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

def calculate_analytical_weights(mu=0):
    """
    Calculates weights for N(mu, 1) assuming scale factors are applied 
    based on percentiles of the *underlying distribution* N(mu, 1).
    """
    # Distribution
    dist = norm(loc=mu, scale=1)
    
    # Denominator: Expected Positive Value of X ~ N(mu, 1)
    # E[max(0, X)] = int_0^inf x * phi(x-mu) dx
    # Let's compute this numerically to be safe for any mu
    def integrand_denom(x):
        return x * dist.pdf(x)
    E_total, _ = quad(integrand_denom, 0, np.inf)

    # Helper: Quantile Function based on Percentiles of N(mu, 1)
    # The scale factor S(u) depends on the *percentile* u.
    # We want to integrate S(u) * Q(u) du from u_start to 1.
    # Where Q(u) is the quantile function of N(mu, 1).
    # Since we are interested in Positive Value, we only integrate where Q(u) > 0.
    # For mu=0, Q(0.5)=0, so we integrate 0.5 to 1.
    # For mu != 0, Q(u)=0 happens at u_zero = CDF(0).
    # so we integrate from u_zero to 1.
    
    u_zero = dist.cdf(0)
    
    q = dist.ppf

    # Helper: Linear Interpolation Integrand
    def integrand(u, u_start, width, weight_type):
        factor = (u - u_start) / width if weight_type == 'rising' else ((u_start + width) - u) / width
        return factor * q(u)
    
    # Integration limits are intersection of [u_zero, 1] and the regions defined by 0.65, 0.80, 0.95
    # Region 1: effectively u_zero -> 0.65
    # We extend f65 to cover all positive values below the 65th percentile
    r1_start = u_zero
    r1_end = 0.65
    
    # Omega 65
    o65 = 0
    if r1_end > r1_start:
         val, _ = quad(q, r1_start, r1_end)
         o65 += val
         
    # Region 2: 0.65 -> 0.80
    r2_start = max(u_zero, 0.65)
    r2_end = 0.80
    
    # Contribution to f65 (Falling)
    if r2_end > r2_start:
        val, _ = quad(integrand, r2_start, r2_end, args=(0.65, 0.15, 'falling'))
        o65 += val
        
    w_65 = o65 / E_total
    
    # Omega 80
    o80 = 0
    # Region 2 (Rising)
    if r2_end > r2_start:
        val, _ = quad(integrand, r2_start, r2_end, args=(0.65, 0.15, 'rising'))
        o80 += val
        
    # Region 3: 0.80 -> 0.95
    r3_start = max(u_zero, 0.80)
    r3_end = 0.95
    
    # Contribution to f80 (Falling)
    if r3_end > r3_start:
        val, _ = quad(integrand, r3_start, r3_end, args=(0.80, 0.15, 'falling'))
        o80 += val
        
    w_80 = o80 / E_total
    
    # Omega 95
    o95 = 0
    # Region 3 (Rising)
    if r3_end > r3_start:
        val, _ = quad(integrand, r3_start, r3_end, args=(0.80, 0.15, 'rising'))
        o95 += val
        
    # Region 4: 0.95 -> 1.00
    r4_start = max(u_zero, 0.95)
    r4_end = 1.00
    
    if r4_end > r4_start:
        val, _ = quad(q, r4_start, r4_end)
        o95 += val
        
    w_95 = o95 / E_total

    return w_65, w_80, w_95

means = [0, 0.5, -0.5]
for mu in means:
    w65, w80, w95 = calculate_analytical_weights(mu)
    print(f"Mean {mu}:")
    print(f"  w_65: {w65:.4%}")
    print(f"  w_80: {w80:.4%}")
    print(f"  w_95: {w95:.4%}")
    print(f"  Sum:  {w65+w80+w95:.4%}\n")
