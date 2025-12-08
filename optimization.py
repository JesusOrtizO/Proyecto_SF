#optimization.py
"""
Funciones de optimización de portafolios para la aplicación de Seminario de Finanzas.

Todas las funciones trabajan con:
- vector de rendimientos esperados anuales (mu_annual)
- matriz de varianza-covarianza anualizada (cov_annual)

Se utilizan restricciones estándar:
- suma de pesos = 1
- sin ventas en corto (pesos entre 0 y 1), salvo que se indique lo contrario.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import scipy.optimize as op

PERIODS_PER_YEAR = 252


def annual_mean_cov(returns: pd.DataFrame, periods_per_year: int = PERIODS_PER_YEAR):
    """
    Calcula rendimientos medios y matriz var-covar anualizados a partir de rendimientos por periodo.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame con rendimientos por periodo (columnas = activos).
    periods_per_year : int, optional
        Número de periodos por año, por defecto 252.

    Returns
    -------
    mu_annual : np.ndarray
        Vector (n,) de rendimientos esperados anuales.
    cov_annual : np.ndarray
        Matriz (n,n) de varianza-covarianza anualizada.
    """
    mu_daily = returns.mean().values
    cov_daily = returns.cov().values
    mu_annual = mu_daily * periods_per_year
    cov_annual = cov_daily * periods_per_year
    return mu_annual, cov_annual


def _portfolio_return(weights: np.ndarray, mu_annual: np.ndarray) -> float:
    return float(weights @ mu_annual)


def _portfolio_volatility(weights: np.ndarray, cov_annual: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov_annual @ weights.T))


def minimize_volatility(mu_annual: np.ndarray, cov_annual: np.ndarray) -> np.ndarray:
    """
    Obtiene el portafolio de mínima volatilidad sin ventas en corto.

    Restricciones:
    - sum w_i = 1
    - 0 <= w_i <= 1
    """
    n = len(mu_annual)
    x0 = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w):
        return _portfolio_volatility(w, cov_annual)

    result = op.minimize(objective, x0=x0, bounds=bounds, constraints=constraints)
    return result.x


def max_sharpe_portfolio(
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    rf_annual: float = 0.04,
) -> np.ndarray:
    """
    Obtiene el portafolio de máximo Sharpe sin ventas en corto.

    Sharpe = (E[R_p] - r_f) / sigma_p
    """
    n = len(mu_annual)
    x0 = np.ones(n) / n
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w):
        port_ret = _portfolio_return(w, mu_annual)
        port_vol = _portfolio_volatility(w, cov_annual)
        if port_vol == 0:
            return 1e6
        sharpe = (port_ret - rf_annual) / port_vol
        return -sharpe  # minimizar el negativo equivale a maximizar Sharpe

    result = op.minimize(objective, x0=x0, bounds=bounds, constraints=constraints)
    return result.x


def target_return_portfolio(
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    target_return: float,
    allow_short: bool = False,
) -> np.ndarray:
    """
    Obtiene el portafolio de mínima volatilidad sujeto a un retorno anual objetivo.

    Problema:
        minimizar sigma_p(w)
        sujeto a:
            sum w_i = 1
            w^T mu = target_return
    """
    n = len(mu_annual)
    x0 = np.ones(n) / n
    if allow_short:
        bounds = tuple((-1.0, 1.0) for _ in range(n))
    else:
        bounds = tuple((0.0, 1.0) for _ in range(n))

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, mu=mu_annual, tr=target_return: w @ mu - tr},
    )

    def objective(w):
        return _portfolio_volatility(w, cov_annual)

    result = op.minimize(objective, x0=x0, bounds=bounds, constraints=constraints)
    return result.x
