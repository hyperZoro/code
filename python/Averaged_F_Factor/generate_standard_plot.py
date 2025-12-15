
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
N_SAMPLES = 100_000
mu = 0
f65, f80, f95 = 2.0, 1.5, 1.0 # Standard example values

plt.figure(figsize=(8, 6))
    
# Generate Data
X = np.random.normal(mu, 1, N_SAMPLES)

# Calculate Percentiles regarding the underlying N(mu, 1)
U = norm.cdf(X, loc=mu, scale=1)

# Determine S(u)
S = np.ones_like(X) # Default to 1 (will be overwritten for u > 0.5)

# Region 1 (extended to cover all u <= 0.65 for positive values)
mask1 = (U <= 0.65)
S[mask1] = f65

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
plt.hist(X[pos_mask], bins=100, alpha=0.5, density=True, color='gray', label='Original Positive')
plt.hist(X_scaled[pos_mask], bins=100, alpha=0.5, density=True, color='blue', label='Scaled Positive')

# Vertical Lines
plt.axvline(x65, color='orange', linestyle='--', linewidth=2, label='Standard Normal 65%')
plt.axvline(x80, color='red', linestyle='--', linewidth=2, label='Standard Normal 80%')
plt.axvline(x95, color='purple', linestyle='--', linewidth=2, label='Standard Normal 95%')

plt.title(f"Standard Normal (Mean $\mu = 0$)\n$f_{{65}}={f65}, f_{{80}}={f80}, f_{{95}}={f95}$")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pic/standard_distribution.png', dpi=300)
print("Plot saved to pic/standard_distribution.png")
