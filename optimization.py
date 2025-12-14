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
# ============================
# Black-Litterman (BL)
# ============================

import numpy as np

def implied_risk_aversion(
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    w_mkt: np.ndarray,
    rf_annual: float
) -> float:
    """
    λ = (E[Rm] - rf) / Var(Rm)
    consistente con media-varianza en el portafolio de mercado.
    """
    w_mkt = np.asarray(w_mkt, dtype=float).reshape(-1)
    w_mkt = w_mkt / w_mkt.sum()

    mu_mkt = float(w_mkt @ mu_annual)
    var_mkt = float(w_mkt @ cov_annual @ w_mkt)

    if var_mkt <= 0:
        return np.nan

    lam = (mu_mkt - rf_annual) / var_mkt
    return float(lam)


def black_litterman_posterior(
    cov_annual: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float = 0.025,
    Omega: np.ndarray | None = None,
):
    """
    Implementación BL (Bayesiana):
    mu_bl = [ (tauΣ)^-1 + P'Ω^-1P ]^-1 [ (tauΣ)^-1 Π + P'Ω^-1 Q ]
    cov_mu = [ (tauΣ)^-1 + P'Ω^-1P ]^-1

    Devuelve:
      mu_bl (posterior mean, exceso o nominal según pi/Q),
      cov_bl (matriz para optimización),
      Omega (la usada)
    """
    Sigma = np.asarray(cov_annual, dtype=float)
    pi = np.asarray(pi, dtype=float).reshape(-1, 1)
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float).reshape(-1, 1)

    n = Sigma.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("cov_annual debe ser NxN.")
    if pi.shape[0] != n:
        raise ValueError("pi debe tener dimensión N.")
    if P.shape[1] != n:
        raise ValueError("P debe ser KxN.")
    if Q.shape[0] != P.shape[0]:
        raise ValueError("Q debe tener dimensión K.")

    # Ω por He-Litterman: Ω = diag(P (tau Σ) P')
    if Omega is None:
        mid = P @ (tau * Sigma) @ P.T
        Omega = np.diag(np.diag(mid))

    Omega = np.asarray(Omega, dtype=float)

    # Estabilización numérica ligera
    eps = 1e-12
    Sigma_t = tau * Sigma
    Sigma_t_inv = np.linalg.inv(Sigma_t + eps * np.eye(n))
    Omega_inv = np.linalg.inv(Omega + eps * np.eye(Omega.shape[0]))

    A = Sigma_t_inv + P.T @ Omega_inv @ P
    b = Sigma_t_inv @ pi + P.T @ Omega_inv @ Q

    cov_mu = np.linalg.inv(A + eps * np.eye(n))
    mu_bl = cov_mu @ b

    # cov posterior para optimización (práctico): Σ_BL = Σ + cov_mu
    cov_bl = Sigma + cov_mu

    return mu_bl.reshape(-1), cov_bl, Omega


def equilibrium_pi(
    cov_annual: np.ndarray,
    w_mkt: np.ndarray,
    lam: float
) -> np.ndarray:
    """
    Π = λ Σ w_mkt
    """
    Sigma = np.asarray(cov_annual, dtype=float)
    w = np.asarray(w_mkt, dtype=float).reshape(-1)
    w = w / w.sum()
    return (lam * (Sigma @ w)).reshape(-1)


def build_views_from_pairs(
    tickers: list[str],
    views: list[dict],
):
    """
    views: lista de dicts con formato:
      - tipo='absoluta': {'type':'absolute','asset':'SPLG','q':0.06}
      - tipo='relativa': {'type':'relative','asset_long':'SPLG','asset_short':'EEM','q':0.02}
    q en términos ANUALES (exceso o nominal según tu definición).

    Devuelve P (KxN), Q (K,)
    """
    n = len(tickers)
    idx = {t:i for i,t in enumerate(tickers)}

    P_rows = []
    Q = []

    for v in views:
        if v["type"] == "absolute":
            p = np.zeros(n)
            p[idx[v["asset"]]] = 1.0
            P_rows.append(p)
            Q.append(float(v["q"]))
        elif v["type"] == "relative":
            p = np.zeros(n)
            p[idx[v["asset_long"]]] = 1.0
            p[idx[v["asset_short"]]] = -1.0
            P_rows.append(p)
            Q.append(float(v["q"]))
        else:
            raise ValueError("Tipo de vista no soportado.")

    return np.vstack(P_rows), np.array(Q, dtype=float)
