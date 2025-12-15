# Averaged F Factor

This project implements a methodology to derive a single equivalent scale factor from a set of piecewise scale factors defined on specific percentiles of a Normal distribution.

The goal is to find a constant factor $f$ such that the **Expected Positive Value (EPV)** of the distribution scaled by $f$ matches the EPV of the distribution scaled by the piecewise function $S(u)$.

## Methodology

The detailed mathematical derivation is available in [doc/averaged_f_factor.md](doc/averaged_f_factor.md).

The derivation involves:
1.  **Value Space formulation**: Integrating in $x$-space for intuitive derivation.
2.  **Piecewise Scaling**: Handling scale factors ($f_{65}, f_{80}, f_{95}$) with linear interpolation.
3.  **Non-Zero Mean Extension**: Generalizing the result for shifted normal distributions $\mathcal{N}(\mu, 1)$.

### Key Results (Standard Normal $\mu=0$)

For a Standard Normal distribution, the equivalent single factor is a weighted average:

$$ f \approx 17.07\% \cdot f_{65} + 32.50\% \cdot f_{80} + 50.43\% \cdot f_{95} $$

Results for shifted means (e.g., $\mu=0.5, -0.5$) are also provided in the documentation.

## Project Structure

*   **`doc/`**: Contains the methodology documentation.
*   **`pic/`**: Contains generated visualizations.
*   **`verification/`**: Contains scripts and notebooks for numerical verification.
*   **`archive/`**: Contains archived attempts and corrections.

## Scripts

*   **`generate_standard_plot.py`**: Generates the visualization of the scaled standard normal distribution.
*   **`generate_shifted_plot.py`**: Generates visualizations for distributions with non-zero means ($\mu=0.5, -0.5$).
*   **`generate_plot.py`**: Generates the visualization of the interpolation function $S(u)$.
*   **`calculate_weights_non_zero_mean.py`**: Calculates the analytical weights for distributions with configurable means.
*   **`averaged_f_factor.ipynb`**: A Jupyter notebook demonstrating the interpolation function.
*   **`verification/numerical_verification_script.py`**: Monte Carlo simulation to verify the analytical results.

## Usage

To generate the plots:
```bash
python generate_standard_plot.py
python generate_shifted_plot.py
python generate_plot.py
```

To calculate the analytical weights for various means:
```bash
python calculate_weights_non_zero_mean.py
```

To run the numerical verification:
```bash
python verification/numerical_verification_script.py
```
