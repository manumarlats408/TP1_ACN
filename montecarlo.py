import pandas as pd

def resumen_montecarlo(resultados, alpha=0.05):
    import numpy as np
    resultados = np.array(resultados, dtype=float)
    n = len(resultados)
    mean = float(np.mean(resultados))
    std = float(np.std(resultados, ddof=1))
    se = std / np.sqrt(n)
    z = 1.96  # IC95%
    ci_low, ci_high = mean - z*se, mean + z*se

    # Devolver como DataFrame directamente
    df = pd.DataFrame([{
        "Media": mean,
        "Desvío Std": std,
        "Error Std": se,
        "IC95% bajo": ci_low,
        "IC95% alto": ci_high
    }])
    return df.round(3)
