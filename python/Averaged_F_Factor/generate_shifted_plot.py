
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
N_SAMPLES = 100_000
means = [0.5, -0.5]
f65, f80, f95 = 2.0, 1.5, 1.0 # Extreme values to show effect

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, mu in enumerate(means):
    ax = axes[i]
    
    # Generate Data
    X = np.random.normal(mu, 1, N_SAMPLES)
    
    # Calculate Percentiles regarding the underlying N(mu, 1)
    # The scale factor logic depends on the percentile of the distribution itself
    U = norm.cdf(X, loc=mu, scale=1)
    
    # Determine S(u)
    S = np.ones_like(X) * f65 # Default to f65 for the body
    
    # Region 2
    mask2 = (U > 0.65) & (U <= 0.80)
    S[mask2] = f65 + (f80 - f65) * (U[mask2] - 0.65) / 0.15
    
    # Region 3
    mask3 = (U > 0.80) & (U <= 0.95)
    S[mask3] = f80 + (f95 - f80) * (U[mask3] - 0.80) / 0.15
    
    # Region 4
    mask4 = (U > 0.95)
    S[mask4] = f95
    
    X_scaled = X * S
    
    # Plot Histograms (Positive only)
    # Get thresholds in X-space
    x65 = norm.ppf(0.65, loc=mu, scale=1)
    x80 = norm.ppf(0.80, loc=mu, scale=1)
    x95 = norm.ppf(0.95, loc=mu, scale=1)
    
    pos_mask = X > 0
    ax.hist(X[pos_mask], bins=100, alpha=0.5, density=True, color='gray', label='Original Positive')
    ax.hist(X_scaled[pos_mask], bins=100, alpha=0.5, density=True, color='blue', label='Scaled Positive')
    
    # Vertical Lines
    ax.axvline(x65, color='orange', linestyle='--', linewidth=2, label='Standard Normal 65%')
    ax.axvline(x80, color='red', linestyle='--', linewidth=2, label='Standard Normal 80%')
    ax.axvline(x95, color='purple', linestyle='--', linewidth=2, label='Standard Normal 95%')
    
    ax.set_title(f"Mean $\mu = {mu}$\n$f_{{65}}={f65}, f_{{80}}={f80}, f_{{95}}={f95}$")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pic/shifted_distributions.png', dpi=300)
print("Plot saved to pic/shifted_distributions.png")
