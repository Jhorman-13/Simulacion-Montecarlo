import numpy as np

def bimodal_pdf(x: np.ndarray, mu1: float=0, sig1: float=1, mu2: float=5, sig2: float=1.5) -> np.ndarray:
    """Densidad de probabilidad bimodal no normalizada."""
    term1 = np.exp(-((x - mu1)**2) / (2 * sig1**2))
    term2 = 0.5 * np.exp(-((x - mu2)**2) / (2 * sig2**2))
    return term1 + term2

def rejection_sampling_traffic(n_samples: int = 10000, seed: int = 42) -> tuple:
    """
    Genera muestras de tráfico bimodal usando Rejection Sampling vectorizado por lotes.
    """
    np.random.seed(seed)
    samples = []
    attempts = 0
    
    # Envolvente q(x) (Uniforme) y constante k
    low, high = -5, 12
    q_pdf = 1 / (high - low)
    k = 25.0
    
    # Generación vectorizada por lotes (evita el for uno a uno)
    while len(samples) < n_samples:
        batch = n_samples
        x_prop = np.random.uniform(low, high, batch)
        u = np.random.uniform(0, k * q_pdf, batch)
        
        # Máscara booleana: acepta si u <= p*(x)
        accepted = x_prop[u <= bimodal_pdf(x_prop)]
        samples.extend(accepted)
        attempts += batch
        
    final_samples = np.array(samples[:n_samples])
    acceptance_rate = n_samples / attempts
    
    return final_samples, acceptance_rate