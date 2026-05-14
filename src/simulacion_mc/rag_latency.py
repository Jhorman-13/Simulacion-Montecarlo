import numpy as np

def simulate_rag_latency(n_samples: int = 100000, seed: int = 42) -> np.ndarray:
    """
    Simula la latencia de un sistema RAG mediante Monte Carlo de forma vectorizada.
    
    Args:
        n_samples (int): Número de consultas a simular.
        seed (int): Semilla para asegurar la reproducibilidad.
        
    Returns:
        np.ndarray: Arreglo 1D con los tiempos totales de respuesta simulados.
    """
    np.random.seed(seed)
    
    # 1. Generación de embeddings (Distribución Normal: media=50, std=10)
    t_emb = np.random.normal(loc=50, scale=10, size=n_samples) 
    
    # 2. Búsqueda Vectorial (Distribución Uniforme: min=10, max=30)
    t_bd = np.random.uniform(low=10, high=30, size=n_samples)
    
    # 3. Inferencia LLM (Distribución Lognormal: media=4.5, sigma=0.5)
    t_llm = np.random.lognormal(mean=4.5, sigma=0.5, size=n_samples)
    
    # Tiempo total vectorizado (suma de arreglos)
    t_total = t_emb + t_bd + t_llm
    
    return t_total