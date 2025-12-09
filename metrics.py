# metrics.py
"""
Módulo de métricas para portafolios.

Define funciones para:
- Retornos y volatilidad anualizados
- Max drawdown y Calmar
- Ratios de Sharpe, Sortino, Treynor e Information Ratio
- VaR y CVaR históricos
- Cálculo consolidado de métricas de un portafolio
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# Convención: 252 días hábiles al año
PERIODS_PER_YEAR: int = 252
# Tasa libre de riesgo por defecto (4% anual, puedes cambiarla)
DEFAULT_RF_ANNUAL: float = 0.04


# ============================================================
# Utilidades básicas
# ============================================================

def _to_series(r: pd.Series | np.ndarray | list) -> pd.Series:
    """Convierte la entrada a pd.Series y elimina NaN."""
    if isinstance(r, pd.Series):
        s = r.copy()
    else:
        s = pd.Series(r)
    return s.dropna()


def annualized_mean(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """
    Media aritmética anualizada: E[R] * periods.

    Esta es la que se usa típicamente en Markowitz, Sharpe, Treynor, etc.
    """
    r = _to_series(returns)
    if len(r) == 0:
        return float("nan")
    return float(r.mean() * periods)


def annualized_return(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """
    Retorno anual compuesto (CAGR) a partir de rendimientos periódicos.

    CAGR = (∏(1 + r_t))^(1/años) - 1
    """
    r = _to_series(returns)
    if len(r) == 0:
        return float("nan")

    cum_return = float((1.0 + r).prod())
    # años efectivos cubiertos por la muestra
    years = len(r) / float(periods)
    if years <= 0:
        return float("nan")

    return float(cum_return ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Volatilidad anual = desviación estándar * sqrt(periods)."""
    r = _to_series(returns)
    if len(r) == 0:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """
    Máximo drawdown sobre la serie de rendimientos periódicos.

    Devuelve un valor negativo (ej. -0.25 = -25%).
    """
    r = _to_series(returns)
    if len(r) == 0:
        return float("nan")

    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1.0
    return float(drawdowns.min())


# ============================================================
# Ratios de rendimiento-ajustado-por-riesgo
# ============================================================

def sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    periods: int = PERIODS_PER_YEAR,
) -> float:
    """
    Ratio de Sharpe anualizado.

    Sharpe = (E[R] - rf) / sigma
    donde E[R] y sigma son anuales.
    """
    r = _to_series(returns)
    if len(r) == 0:
        return float("nan")

    mean_ann = annualized_mean(r, periods=periods)
    vol_ann = annualized_vol(r, periods=periods)

    if vol_ann == 0 or np.isnan(vol_ann):
        return float("nan")

    return float((mean_ann - rf_annual) / vol_ann)


def sortino_ratio(
    returns: pd.Series,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    periods: int = PERIODS_PER_YEAR,
) -> float:
    """
    Ratio de Sortino anualizado.

    Numerador: exceso de retorno anual (E[R] - rf)
    Denominador: desviación estándar de rendimientos por debajo de rf (downside risk).
    """
    r = _to_series(returns)
    if len(r) == 0:
        return float("nan")

    # rf diario aproximado a partir de rf anual
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods) - 1.0

    downside = r[r < rf_daily]
    if len(downside) == 0:
        return float("nan")

    mean_ann = annualized_mean(r, periods=periods)
    downside_vol_ann = float(downside.std(ddof=1) * np.sqrt(periods))

    if downside_vol_ann == 0 or np.isnan(downside_vol_ann):
        return float("nan")

    return float((mean_ann - rf_annual) / downside_vol_ann)


def calmar_ratio(
    returns: pd.Series,
    periods: int = PERIODS_PER_YEAR,
) -> float:
    """
    Calmar ratio estándar:

    Calmar = CAGR / |Max Drawdown|
    """
    r = _to_series(returns)
    if len(r) == 0:
        return float("nan")

    cagr = annualized_return(r, periods=periods)
    md = max_drawdown(r)

    if np.isnan(cagr) or np.isnan(md) or md >= 0:
        return float("nan")

    return float(cagr / abs(md))


def beta_vs_benchmark(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Beta del portafolio vs un benchmark, calculada con rendimientos periódicos.

    Beta = Cov(R_p, R_b) / Var(R_b)
    """
    r_p = _to_series(returns)
    r_b = _to_series(benchmark_returns)

    if len(r_p) == 0 or len(r_b) == 0:
        return float("nan")

    df = pd.concat([r_p, r_b], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")

    cov = float(np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)[0, 1])
    var_b = float(np.var(df.iloc[:, 1], ddof=1))

    if var_b == 0:
        return float("nan")

    return float(cov / var_b)


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    periods: int = PERIODS_PER_YEAR,
) -> float:
    """
    Treynor ratio anualizado:

    Treynor = (E[R] - rf) / Beta

    E[R] es la media aritmética anualizada (no se vuelve a anualizar después).
    """
    r = _to_series(returns)
    r_b = _to_series(benchmark_returns)

    if len(r) == 0 or len(r_b) == 0:
        return float("nan")

    beta = beta_vs_benchmark(r, r_b)
    if beta == 0 or np.isnan(beta):
        return float("nan")

    mean_ann = annualized_mean(r, periods=periods)
    return float((mean_ann - rf_annual) / beta)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods: int = PERIODS_PER_YEAR,
) -> float:
    """
    Information ratio anualizado:

    IR = E[R_p - R_b] / sigma(R_p - R_b)
    (ambos anualizados).
    """
    r_p = _to_series(returns)
    r_b = _to_series(benchmark_returns)

    if len(r_p) == 0 or len(r_b) == 0:
        return float("nan")

    active = (r_p - r_b).dropna()
    if len(active) == 0:
        return float("nan")

    mean_active_ann = float(active.mean() * periods)
    vol_active_ann = float(active.std(ddof=1) * np.sqrt(periods))

    if vol_active_ann == 0 or np.isnan(vol_active_ann):
        return float("nan")

    return float(mean_active_ann / vol_active_ann)


# ============================================================
# VaR y CVaR históricos
# ============================================================

def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    VaR histórico (cuantil) sobre la distribución de rendimientos periódicos.

    Devuelve el cuantil alpha (p.ej. 5% → pérdidas más extremas).
    """
    r = _to_series(returns)
    if len(r) == 0:
        return float("nan")
    return float(r.quantile(alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    CVaR (Expected Shortfall) histórico: media condicional de las pérdidas
    más extremas por debajo del cuantil alpha.
    """
    r = _to_series(returns)
    if len(r) == 0:
        return float("nan")

    var_level = r.quantile(alpha)
    tail = r[r <= var_level]
    if len(tail) == 0:
        return float("nan")

    return float(tail.mean())


# ============================================================
# Función principal para la app
# ============================================================

def compute_portfolio_metrics(
    portfolio_returns: pd.Series,
    rf_annual: float = DEFAULT_RF_ANNUAL,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Calcula todas las métricas que consume la app.

    Devuelve un diccionario con claves:
    - mean_annual         (media aritmética anualizada)
    - vol_annual          (volatilidad anual)
    - skew                (asimetría)
    - kurtosis            (curtosis)
    - max_drawdown        (máximo drawdown)
    - sharpe              (ratio de Sharpe)
    - sortino             (ratio de Sortino)
    - VaR_5               (VaR histórico 5%)
    - CVaR_5              (CVaR histórico 5%)
    - calmar              (Calmar ratio con CAGR)
    - beta                (beta vs benchmark)
    - treynor             (Treynor ratio)
    - information_ratio   (Information ratio)
    """
    r = _to_series(portfolio_returns)

    metrics: Dict[str, float] = {}

    if len(r) == 0:
        # Rellenar con NaN si no hay datos
        keys = [
            "mean_annual",
            "vol_annual",
            "skew",
            "kurtosis",
            "max_drawdown",
            "sharpe",
            "sortino",
            "VaR_5",
            "CVaR_5",
            "calmar",
            "beta",
            "treynor",
            "information_ratio",
        ]
        for k in keys:
            metrics[k] = float("nan")
        return metrics

    # Métricas básicas
    metrics["mean_annual"] = annualized_mean(r)
    metrics["vol_annual"] = annualized_vol(r)
    metrics["skew"] = float(r.skew())
    metrics["kurtosis"] = float(r.kurtosis())
    metrics["max_drawdown"] = max_drawdown(r)

    # Ratios de rendimiento-riesgo
    metrics["sharpe"] = sharpe_ratio(r, rf_annual=rf_annual)
    metrics["sortino"] = sortino_ratio(r, rf_annual=rf_annual)
    metrics["calmar"] = calmar_ratio(r)

    # Riesgo de cola
    metrics["VaR_5"] = historical_var(r, alpha=0.05)
    metrics["CVaR_5"] = historical_cvar(r, alpha=0.05)

    # Métricas relativas al benchmark
    if benchmark_returns is None:
        metrics["beta"] = float("nan")
        metrics["treynor"] = float("nan")
        metrics["information_ratio"] = float("nan")
    else:
        b = _to_series(benchmark_returns)
        if len(b) == 0:
            metrics["beta"] = float("nan")
            metrics["treynor"] = float("nan")
            metrics["information_ratio"] = float("nan")
        else:
            metrics["beta"] = beta_vs_benchmark(r, b)
            metrics["treynor"] = treynor_ratio(
                r,
                b,
                rf_annual=rf_annual,
            )
            metrics["information_ratio"] = information_ratio(r, b)

    return metrics
