import numpy as np
from .contracts import Array


def normalized_error(target: Array, prediction: Array) -> float:
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Target y predicción deben tener igual forma no vacía.")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise ValueError("NMSE requiere valores finitos.")
    energy = float(np.vdot(target, target).real)
    if energy <= np.finfo(float).tiny:
        raise ValueError("Target sin energía.")
    residual = target - prediction
    return float(np.vdot(residual, residual).real / energy)


def error_to_db(error: float) -> float:
    if error < 0 or not np.isfinite(error):
        raise ValueError("Error lineal inválido.")
    return float(10 * np.log10(error)) if error else -float("inf")


def db_to_error(db: float) -> float:
    return float(10 ** (db / 10))


def nmse(target: Array, prediction: Array) -> float:
    return error_to_db(normalized_error(target, prediction))


def bic_penalty_db(sample_count: int) -> float:
    """Becerra et al. (2020), eqs. (9)/(13); complex samples/coefficients."""
    if not isinstance(sample_count, (int, np.integer)) or sample_count < 1:
        raise ValueError("BIC requiere un número positivo de muestras complejas.")
    return float(10 * np.log10(2 * sample_count) / sample_count)


def bic_score_db(error: float, coefficients: int, sample_count: int) -> float:
    """BIC in NMSE units; differs from eq. (6) only by scale/constant.

    Clamp exact zero to the floating-point floor to avoid inf-inf in comparisons.
    Count independent fitted complex coefficients, not hardware operations.
    """
    if error < 0 or not np.isfinite(error) or coefficients < 1:
        raise ValueError("Error o número de coeficientes BIC inválido.")
    return error_to_db(max(error, np.finfo(float).tiny)) + coefficients * bic_penalty_db(sample_count)


def progress_fraction(baseline: float, candidate: float, reference: float) -> float:
    # No clipping: candidates may outperform the estimated reference.
    denominator = baseline - reference
    if denominator <= np.finfo(float).eps * max(baseline, np.finfo(float).tiny):
        return 0.0
    return (baseline - candidate) / denominator
