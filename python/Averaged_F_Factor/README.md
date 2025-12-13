# Averaged F Factor

This project implements a methodology to derive a single equivalent scale factor from a set of piecewise scale factors defined on specific percentiles of a Standard Normal distribution.

The goal is to find a constant factor $f$ such that the **Expected Positive Value (EPV)** of the distribution scaled by $f$ matches the EPV of the distribution scaled by the piecewise function $S(u)$.

## Methodology

The detailed mathematical derivation is available in [doc/averaged_f_factor.md](doc/averaged_f_factor.md).

The derivation involves:
1.  Defining the piecewise linear interpolation function $S(u)$ in probability space.
2.  Integrating the weighted contribution of each factor ($f_{65}, f_{80}, f_{95}$).
3.  Calculating the final weights numerically.

### Key Results

For a Standard Normal distribution, the equivalent single factor is a weighted average of the input factors:

$$ f \approx 17.07\% \cdot f_{65} + 32.50\% \cdot f_{80} + 50.43\% \cdot f_{95} $$

## Project Structure

*   **`doc/`**: Contains the methodology documentation.
*   **`pic/`**: Contains generated visualizations.
*   **`verification/`**: Contains scripts and notebooks for numerical verification of the results.
*   **`archive/`**: Contains archived attempts and corrections.

## Scripts

*   **`generate_plot.py`**: Generates the visualization of the interpolation function $S(u)$ and saves it to `pic/interpolation_function.png`.
*   **`calculate_terms.py`**: Performs the analytical integration to compute the exact weights for each factor.
*   **`averaged_f_factor.ipynb`**: A Jupyter notebook demonstrating the interpolation function.
*   **`verify_correction.py`**: A script used to verify the logic of alternative derivations.

## Usage

To generate the plot:
```bash
python generate_plot.py
```

To calculate the analytical weights:
```bash
python calculate_terms.py
```
