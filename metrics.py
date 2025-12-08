# metrics.py
"""
Funciones de métricas de desempeño y riesgo para portafolios.

Todas las funciones trabajan sobre series de rendimientos por periodo (por ejemplo diarios).
Se asume una frecuencia de 252 días hábiles por año para anualizar.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 252
DEFAULT_RF_ANNUAL = 0.04  # tasa libre de riesgo anual aproximada


def annualize_rf(rf_annual: float = DEFAULT_RF_ANNUAL) -> float:
    """
    Convierte una tasa libre de riesgo anual a tasa por periodo (diaria).
    """
    return (1.0 + rf_annual) ** (1.0 / PERIODS_PER_YEAR) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """
    Calcula el máximo drawdown de una serie de rendimientos.
    Devuelve un valor negativo (máxima caída porcentual).
    """
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, rf_annual: float = DEFAULT_RF_ANNUAL) -> float:
    """
    Calcula el Sharpe ratio anualizado de una serie de rendimientos.
    """
    rf_daily = annualize_rf(rf_annual)
    excess = returns - rf_daily
    mean_excess = excess.mean()
    std_excess = excess.std()
    if std_excess == 0:
        return float("nan")
    return float(np.sqrt(PERIODS_PER_YEAR) * mean_excess / std_excess)


def downside_deviation(
    returns: pd.Series, rf_annual: float = DEFAULT_RF_ANNUAL
) -> float:
    """
    Desviación estándar a la baja respecto a la tasa libre de riesgo.
    Solo considera rendimientos por debajo de r_f.
    """
    rf_daily = annualize_rf(rf_annual)
    diff = returns - rf_daily
    downside = np.minimum(diff, 0.0)
    if len(downside) == 0:
        return float("nan")
    return float(np.sqrt((downside**2).mean()))


def sortino_ratio(returns: pd.Series, rf_annual: float = DEFAULT_RF_ANNUAL) -> float:
    """
    Calcula el Sortino ratio anualizado de una serie de rendimientos.
    """
    rf_daily = annualize_rf(rf_annual)
    dd = downside_deviation(returns, rf_annual)
    if dd == 0 or np.isnan(dd):
        return float("nan")
    mean_excess = (returns - rf_daily).mean()
    return float(np.sqrt(PERIODS_PER_YEAR) * mean_excess / dd)


def var_cvar_historic(returns: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """
    Calcula VaR y CVaR históricos para una serie de rendimientos.

    VaR y CVaR devueltos están en la misma escala que los rendimientos
    (por ejemplo, -0.03 representa una pérdida del 3 %).
    """
    r = returns.dropna().values
    if len(r) == 0:
        return {"VaR": float("nan"), "CVaR": float("nan")}

    sorted_r = np.sort(r)
    index = int(alpha * len(sorted_r))
    index = max(0, min(index, len(sorted_r) - 1))
    var = sorted_r[index]
    cvar = sorted_r[: index + 1].mean()
    return {"VaR": float(var), "CVaR": float(cvar)}


def beta_vs_benchmark(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calcula la beta del portafolio respecto a un benchmark.

    Beta = Cov(R_p, R_b) / Var(R_b)
    """
    data = pd.concat(
        [returns.rename("p"), benchmark_returns.rename("b")], axis=1
    ).dropna()
    if len(data) == 0:
        return float("nan")

    cov_matrix = np.cov(data["p"], data["b"])
    var_b = cov_matrix[1, 1]
    if var_b == 0:
        return float("nan")
    cov_pb = cov_matrix[0, 1]
    return float(cov_pb / var_b)


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = DEFAULT_RF_ANNUAL,
) -> float:
    """
    Calcula el Treynor ratio anualizado.

    Treynor = (E[R_p] - r_f) / beta_p
    """
    beta = beta_vs_benchmark(returns, benchmark_returns)
    if beta == 0 or np.isnan(beta):
        return float("nan")
    rf_daily = annualize_rf(rf_annual)
    mean_excess = (returns - rf_daily).mean()
    return float(np.sqrt(PERIODS_PER_YEAR) * mean_excess / beta)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calcula el Information Ratio.

    IR = mean(R_p - R_b) / std(R_p - R_b)
    """
    data = pd.concat(
        [returns.rename("p"), benchmark_returns.rename("b")], axis=1
    ).dropna()
    if len(data) == 0:
        return float("nan")
    diff = data["p"] - data["b"]
    if diff.std() == 0:
        return float("nan")
    return float(np.sqrt(PERIODS_PER_YEAR) * diff.mean() / diff.std())


def calmar_ratio(returns: pd.Series) -> float:
    """
    Calcula el Calmar ratio.

    Calmar = rendimiento anualizado / |max_drawdown|
    """
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    mean_annual = r.mean() * PERIODS_PER_YEAR
    md = max_drawdown(r)
    if md >= 0:
        return float("nan")
    return float(mean_annual / abs(md))


def compute_portfolio_metrics(
    portfolio_returns: pd.Series,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Calcula un conjunto de métricas para un portafolio:

    - Rendimiento medio anualizado.
    - Volatilidad anualizada.
    - Skewness.
    - Kurtosis.
    - Máximo drawdown.
    - Sharpe ratio.
    - Sortino ratio.
    - VaR y CVaR históricos al 5 %.
    - Calmar ratio.
    - (Opcionales, si se proporciona benchmark):
        * Beta vs benchmark.
        * Treynor ratio.
        * Information Ratio.
    """
    r = portfolio_returns.dropna()
    metrics: Dict[str, float] = {}

    if len(r) == 0:
        return metrics

    metrics["mean_annual"] = float(r.mean() * PERIODS_PER_YEAR)
    metrics["vol_annual"] = float(r.std() * np.sqrt(PERIODS_PER_YEAR))
    metrics["skew"] = float(r.skew())
    metrics["kurtosis"] = float(r.kurtosis())
    metrics["max_drawdown"] = max_drawdown(r)
    metrics["sharpe"] = sharpe_ratio(r, rf_annual)
    metrics["sortino"] = sortino_ratio(r, rf_annual)

    var_cvar = var_cvar_historic(r, alpha=0.05)
    metrics["VaR_5"] = var_cvar["VaR"]
    metrics["CVaR_5"] = var_cvar["CVaR"]

    metrics["calmar"] = calmar_ratio(r)

    # Métricas relativas a benchmark (si se proporciona)
    if benchmark_returns is not None:
        try:
            beta = beta_vs_benchmark(r, benchmark_returns)
            treynor = treynor_ratio(r, benchmark_returns, rf_annual)
            ir = information_ratio(r, benchmark_returns)
        except Exception:
            beta = float("nan")
            treynor = float("nan")
            ir = float("nan")

        metrics["beta"] = beta
        metrics["treynor"] = treynor
        metrics["information_ratio"] = ir

    return metrics
