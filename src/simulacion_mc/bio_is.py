import numpy as np
from scipy.stats import lognorm, expon

def importance_sampling_failure(limit_hours: float = 150.0, n_samples: int = 100000, seed: int = 42) -> tuple:
    """
    Estima la probabilidad de fallo por tiempo en un pipeline usando Importance Sampling.
    
    Args:
        limit_hours (float): Límite de horas antes de la cancelación.
        n_samples (int): Cantidad de muestras a simular.
        seed (int): Semilla de reproducibilidad.
        
    Returns:
        tuple: (prob_IS, var_IS, prob_MC, var_MC)
    """
    np.random.seed(seed)
    
    # f(x): Lognormal (tiempos reales del pipeline)
    s_target = 1.0
    scale_target = np.exp(3)
    
    # g(x): Exponencial desplazada (enfocada en la región de fallo > limit_hours)
    loc_prop = limit_hours
    scale_prop = 20.0
    
    # --- 1. Importance Sampling ---
    # Generamos muestras directamente en la zona de peligro
    samples_g = expon.rvs(loc=loc_prop, scale=scale_prop, size=n_samples)
    
    # Pesos vectorizados: w(x) = f(x) / g(x)
    f_x = lognorm.pdf(samples_g, s=s_target, scale=scale_target)
    g_x = expon.pdf(samples_g, loc=loc_prop, scale=scale_prop)
    weights = f_x / g_x
    
    indicator_is = samples_g > limit_hours
    prob_is = np.mean(indicator_is * weights)
    var_is = np.var(indicator_is * weights) / n_samples
    
    # --- 2. Monte Carlo Estándar (Para comparar la varianza) ---
    samples_mc = lognorm.rvs(s=s_target, scale=scale_target, size=n_samples)
    indicator_mc = samples_mc > limit_hours
    prob_mc = np.mean(indicator_mc)
    var_mc = np.var(indicator_mc) / n_samples
    
    return prob_is, var_is, prob_mc, var_mc