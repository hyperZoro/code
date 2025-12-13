import numpy as np
import matplotlib.pyplot as plt
import os

def interpolation_function(u, f65, f80, f95):
    """
    Calculates the scaling factor S(u) based on the methodology.
    """
    if 0.50 < u <= 0.65:
        return f65
    elif 0.65 < u <= 0.80:
        # Linear interpolation between f65 and f80
        # u goes from 0.65 to 0.80 (width 0.15)
        return f65 + (f80 - f65) * (u - 0.65) / 0.15
    elif 0.80 < u <= 0.95:
        # Linear interpolation between f80 and f95
        # u goes from 0.80 to 0.95 (width 0.15)
        return f80 + (f95 - f80) * (u - 0.80) / 0.15
    elif 0.95 < u <= 1.00:
        return f95
    else:
        return np.nan

# Vectorize the function to work with numpy arrays
S_vectorized = np.vectorize(interpolation_function)

# Define example factors for visualization
f65 = 1.2
f80 = 1.4
f95 = 1.3

# Generate u values from 0.5 to 1.0
u_values = np.linspace(0.5001, 1.00, 500) # Start slightly above 0.5 to match domain

# Calculate S(u)
s_values = S_vectorized(u_values, f65, f80, f95)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(u_values, s_values, label='Interpolation Function S(u)', color='#1f77b4', linewidth=2.5)

# Highlight key regions with vertical lines
plt.axvline(x=0.65, color='gray', linestyle='--', alpha=0.6)
plt.axvline(x=0.80, color='gray', linestyle='--', alpha=0.6)
plt.axvline(x=0.95, color='gray', linestyle='--', alpha=0.6)

# Add text labels for regions
plt.text(0.575, f65 + 0.05, f'Constant $f_{{65}}$', ha='center', fontsize=10, fontweight='bold')
plt.text(0.725, (f65+f80)/2 + 0.05, f'Linear Interp\\n$f_{{65}} \\to f_{{80}}$', ha='center', fontsize=10)
plt.text(0.875, (f80+f95)/2 + 0.05, f'Linear Interp\\n$f_{{80}} \\to f_{{95}}$', ha='center', fontsize=10)
plt.text(0.975, f95 + 0.05, f'Constant $f_{{95}}$', ha='center', fontsize=10, fontweight='bold')

# Scatter points at knots
plt.scatter([0.65, 0.80, 0.95], [f65, f80, f95], color='red', zorder=5)

plt.title('Interpolation Function S(u) in Percentage Space', fontsize=14)
plt.xlabel('Cumulative Probability u', fontsize=12)
plt.ylabel('Scale Factor S(u)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0.5, 1.01)
plt.ylim(min(f65, f80, f95) - 0.2, max(f65, f80, f95) + 0.2)
plt.legend()

# Save the plot
output_path = os.path.join('pic', 'interpolation_function.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
