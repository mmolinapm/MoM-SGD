"""Schedules for three MoM-SGD regimes.

Theta_0 is the arbitrary initialization; step n >= 1 produces theta_n from
theta_{n-1}. Regimes 1 and 2 set eta_1 = m_1 = 1 and, for n >= 2, use
eta_n = 1 / (n * log(n)), where log is the natural logarithm.

The general regime regime1_logarithmic (R1-LOG) uses m_n = ceil(log(n)).
The strongly-convex regime regime2_sqrt_logarithmic (R2-SQRT-LOG) uses
m_n = ceil(sqrt(log(n))). It is motivated by the summability condition
sum_n eta_n / m_n^{2(p-1)/p} < infinity (for p > 1).

Regime 3 uses a constant block size m, to be chosen later as an experiment
hyperparameter; no block-size schedule is implemented. R3-CONST-LOG uses
eta_1 = 1 and eta_n = 1 / (n * log(n)) for n >= 2, enabling a controlled
comparison with regimes 1 and 2. R3-CONST-RM uses eta_n = 1 / n for n >= 1.
Both variants satisfy the Robbins-Monro conditions sum_n eta_n = infinity
and sum_n eta_n^2 < infinity.

Changing the finite initialization at n = 1 does not affect the asymptotic
summability conditions motivating these schedules.
"""

import math


def regime1_learning_rate(n: int) -> float:
    """Return eta_1 = 1.0, or eta_n = 1 / (n * log(n)) for n >= 2.

    Step n produces theta_n from theta_{n-1}, starting with arbitrary theta_0.
    The logarithm is natural. Raises ValueError unless n is an integer >= 1.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1; got {n!r}.")
    if n == 1:
        return 1.0
    return 1.0 / (n * math.log(n))


def regime1_block_size(n: int) -> int:
    """Return m_1 = 1, or m_n = ceil(log(n)) for n >= 2, as a Python int.

    Step n produces theta_n from theta_{n-1}, starting with arbitrary theta_0.
    The logarithm is natural. Raises ValueError unless n is an integer >= 1.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1; got {n!r}.")
    if n == 1:
        return 1
    return math.ceil(math.log(n))


def regime2_learning_rate(n: int) -> float:
    """Return eta_1 = 1.0, or eta_n = 1 / (n * log(n)) for n >= 2.

    Step n produces theta_n from theta_{n-1}, starting with arbitrary theta_0.
    The logarithm is natural. Raises ValueError unless n is an integer >= 1.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1; got {n!r}.")
    if n == 1:
        return 1.0
    return 1.0 / (n * math.log(n))


def regime2_block_size(n: int) -> int:
    """Return m_1 = 1, or m_n = ceil(sqrt(log(n))) for n >= 2, as a Python int.

    Step n produces theta_n from theta_{n-1}, starting with arbitrary theta_0.
    The logarithm is natural. Raises ValueError unless n is an integer >= 1.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1; got {n!r}.")
    if n == 1:
        return 1
    return math.ceil(math.sqrt(math.log(n)))


def regime3_log_learning_rate(n: int) -> float:
    """Return eta_1 = 1.0, or eta_n = 1 / (n * log(n)) for n >= 2.

    R3-CONST-LOG matches the learning rates of regimes 1 and 2 for a controlled
    comparison with constant block size. The logarithm is natural.
    Step n produces theta_n from theta_{n-1}, starting with arbitrary theta_0.
    Raises ValueError unless n is an integer >= 1.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1; got {n!r}.")
    if n == 1:
        return 1.0
    return 1.0 / (n * math.log(n))


def regime3_rm_learning_rate(n: int) -> float:
    """Return eta_n = 1 / n for every n >= 1.

    R3-CONST-RM uses the standard Robbins-Monro schedule with constant block
    size: sum_n eta_n = infinity and sum_n eta_n^2 < infinity.
    Step n produces theta_n from theta_{n-1}, starting with arbitrary theta_0.
    Raises ValueError unless n is an integer >= 1.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1; got {n!r}.")
    return 1.0 / n
