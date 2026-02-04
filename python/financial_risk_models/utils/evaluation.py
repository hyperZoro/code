import numpy as np
import scipy.stats as stats
from scipy.stats import kstest

def calculate_metrics(actual, predicted):
    """
    Calculate RMSE and MAE.
    """
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    return {'RMSE': rmse, 'MAE': mae}

def kupiec_pof_test(actual_returns, var_forecasts, alpha=0.05):
    """
    Kupiec Proportion of Failures Test.
    H0: The proportion of failures is equal to alpha.
    """
    failures = np.sum(actual_returns < -var_forecasts)
    N = len(actual_returns)
    p_hat = failures / N
    
    if failures == 0:
        return {'LR_POF': 0.0, 'p_value': 1.0, 'Result': 'Accept H0'}
        
    # Likelihood Ratio Test Statistic
    term1 = -2 * np.log(((1 - alpha)**(N - failures)) * (alpha**failures))
    term2 = 2 * np.log(((1 - p_hat)**(N - failures)) * (p_hat**failures))
    lr_pof = term1 + term2
    
    p_value = 1 - stats.chi2.cdf(lr_pof, df=1)
    
    return {
        'LR_POF': lr_pof, 
        'p_value': p_value, 
        'failures': failures,
        'N': N,
        'observed_rate': p_hat,
        'Result': 'Reject H0' if p_value < 0.05 else 'Accept H0'
    }

def christoffersen_ind_test(actual_returns, var_forecasts):
    """
    Christoffersen Independence Test.
    Checks if violations are clustered.
    """
    failures = (actual_returns < -var_forecasts).astype(int)
    
    n00 = n01 = n10 = n11 = 0
    
    for i in range(1, len(failures)):
        if failures[i-1] == 0 and failures[i] == 0: n00 += 1
        elif failures[i-1] == 0 and failures[i] == 1: n01 += 1
        elif failures[i-1] == 1 and failures[i] == 0: n10 += 1
        elif failures[i-1] == 1 and failures[i] == 1: n11 += 1
        
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    if (n00 + n01 + n10 + n11) == 0: 
         return {'LR_ind': 0, 'p_value': 1}

    # Likelihoods
    L_null = (1 - pi)**(n00 + n10) * pi**(n01 + n11)
    L_alt = (1 - pi0)**n00 * pi0**n01 * (1 - pi1)**n10 * pi1**n11
    
    if L_null == 0 or L_alt == 0:
        lr_ind = 0
    else:
        lr_ind = -2 * np.log(L_null / L_alt)
        
    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
    
    return {'LR_ind': lr_ind, 'p_value': p_value}

def ks_test_uniformity(pit_values):
    """
    Kolmogorov-Smirnov Test for Uniformity (for PIT values).
    H0: Data is uniformly distributed on [0, 1].
    """
    stat, p_value = kstest(pit_values, 'uniform')
    return {'statistic': stat, 'p_value': p_value}
