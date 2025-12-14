# app.py
"""
Aplicación de Seminario de Finanzas.

Funcionalidad:
- Selector de universo de inversión (Regiones o Sectores USA).
- Carga de rendimientos diarios sincronizados desde archivos locales usando sf_library.
- Construcción de índices de valor normalizados (base 100).
- Pestañas para:
    * Datos (mercado).
    * Portafolio arbitrario (pesos definidos por el usuario).
    * Comparación con benchmark (portafolio de referencia con pesos fijos).
    * Portafolios optimizados (Markowitz: mínima varianza, máximo Sharpe, retorno objetivo).
    * Black-Litterman (prior + vistas + posterior + optimización).
"""

import datetime as dt
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import sf_library as sfl
from metrics import compute_portfolio_metrics
from optimization import (
    annual_mean_cov,
    minimize_volatility,
    max_sharpe_portfolio,
    target_return_portfolio,
    # --- BL (asegúrate de que existan en optimization.py)
    implied_risk_aversion,
    equilibrium_pi,
    black_litterman_posterior,
    build_views_from_pairs,
)

# ============================================================
# Universos de inversión y benchmark de referencia
# ============================================================

UNIVERSE_REGIONS = {
    "nombre": "Regiones (ETF)",
    "tickers": ["SPLG", "EWC", "IEUR", "EEM", "EWJ"],
}

UNIVERSE_SECTORS = {
    "nombre": "Sectores USA (ETF)",
    "tickers": [
        "XLC", "XLY", "XLP", "XLE", "XLF",
        "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU",
    ],
}

UNIVERSES: Dict[str, Dict] = {
    "Regiones": UNIVERSE_REGIONS,
    "Sectores USA": UNIVERSE_SECTORS,
}

# Pesos del benchmark por universo (según enunciado del proyecto)
# Valores en proporciones (no en %). Se renormalizan por seguridad.
BENCHMARK_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Regiones": {
        "SPLG": 0.7062,
        "EWC": 0.0323,
        "IEUR": 0.1176,
        "EEM": 0.0902,
        "EWJ": 0.0537,
    },
    "Sectores USA": {
        "XLC": 0.0999,
        "XLY": 0.1025,
        "XLP": 0.0482,
        "XLE": 0.0295,
        "XLF": 0.1307,
        "XLV": 0.0958,
        "XLI": 0.0809,
        "XLB": 0.0166,
        "XLRE": 0.0187,
        "XLK": 0.3535,
        "XLU": 0.0237,
    },
}

# Etiquetas amigables para métricas
METRIC_LABELS = {
    "mean_annual": "Rendimiento anual esperado",
    "vol_annual": "Volatilidad anual",
    "skew": "Asimetría",
    "kurtosis": "Curtosis",
    "max_drawdown": "Máximo drawdown",
    "sharpe": "Ratio de Sharpe",
    "sortino": "Ratio de Sortino",
    "VaR_5": "VaR 5% (histórico)",
    "CVaR_5": "CVaR 5% (histórico)",
    "calmar": "Ratio de Calmar",
    "beta": "Beta vs benchmark",
    "treynor": "Ratio de Treynor",
    "information_ratio": "Information Ratio",
}

# ============================================================
# Funciones auxiliares
# ============================================================

@st.cache_data
def load_synced_returns(
    tickers: List[str],
    start: str,
    end: str,
    data_dir: str = "MarketData",
) -> pd.DataFrame:
    """
    Descarga datos (si es necesario), sincroniza series y devuelve
    un DataFrame con rendimientos diarios en el rango de fechas indicado.
    """
    sfl.descargar_tickers(
        tickers=tickers,
        carpeta=data_dir,
        start=start,
        end=end,
    )

    df_sync, _, _ = sfl.sync_timeseries(
        tickers=tickers,
        data_dir=data_dir,
    )

    df_sync = df_sync.copy()
    df_sync["date"] = pd.to_datetime(df_sync["date"])
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    mask = (df_sync["date"] >= start_ts) & (df_sync["date"] <= end_ts)
    df_sync = df_sync.loc[mask].reset_index(drop=True)

    return df_sync


def build_price_index(returns: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Construye índices de valor normalizados a 'base' desde rendimientos."""
    cumulative = (1.0 + returns).cumprod()
    return base * cumulative


def compute_benchmark(
    returns_matrix: pd.DataFrame, universe_key: str
) -> Tuple[pd.Series, np.ndarray]:
    """
    Benchmark como portafolio con pesos fijos definidos en BENCHMARK_WEIGHTS.
    Devuelve:
    - benchmark_returns: Serie de rendimientos diarios del benchmark.
    - weights_vec: vector de pesos alineado con las columnas de returns_matrix.
    """
    weights_dict = BENCHMARK_WEIGHTS.get(universe_key)
    if weights_dict is None:
        return None, None

    weights_vec = np.array(
        [weights_dict.get(col, 0.0) for col in returns_matrix.columns],
        dtype=float,
    )
    total = weights_vec.sum()
    if total <= 0:
        return None, None

    weights_vec = weights_vec / total
    benchmark_returns = returns_matrix @ weights_vec
    return benchmark_returns, weights_vec


def format_pct(x: float) -> str:
    """Formatea un número como porcentaje con dos decimales."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * x:,.2f}%"


# ============================
# Configuración de la página
# ============================

st.set_page_config(
    page_title="Seminario de Finanzas - Portafolios",
    layout="wide",
)

# CSS ligero (mínimo) para pulir
st.markdown(
    """
    <style>
    h1 { font-weight: 700; letter-spacing: 0.02em; }
    .stDataFrame tbody td { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Seminario de Finanzas – ETF y Portafolios")
st.markdown(
    "Herramienta interactiva para analizar **ETF**, construir portafolios y comparar contra un "
    "portafolio de referencia (*benchmark*). Incluye optimización **Markowitz** y **Black-Litterman**."
)
st.markdown("---")

# ======================
# Panel lateral
# ======================

st.sidebar.header("Parámetros")

universe_key = st.sidebar.selectbox(
    "Universo",
    options=list(UNIVERSES.keys()),
    index=0,
)

universe = UNIVERSES[universe_key]
tickers = universe["tickers"]

st.sidebar.markdown(f"**Universo seleccionado:** {universe['nombre']}")
st.sidebar.caption("ETF incluidos:")
st.sidebar.write(", ".join(tickers))

default_start = dt.date(2015, 1, 1)
default_end = dt.date.today()

start_date = st.sidebar.date_input("Fecha inicial", value=default_start)
end_date = st.sidebar.date_input("Fecha final", value=default_end)

if start_date > end_date:
    st.sidebar.error("La fecha inicial debe ser anterior o igual a la fecha final.")

rf_annual = st.sidebar.number_input(
    "Tasa libre de riesgo (anual)",
    min_value=0.0,
    max_value=0.20,
    value=0.04,
    step=0.005,
    format="%.3f",
)

# ======================
# Carga de datos
# ======================

if start_date <= end_date:
    df_returns = load_synced_returns(
        tickers=tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        data_dir="MarketData",
    )

    dates = df_returns["date"]
    returns_matrix = df_returns.drop(columns="date")

    # Benchmark del universo
    benchmark_returns, benchmark_weights_vec = compute_benchmark(
        returns_matrix=returns_matrix,
        universe_key=universe_key,
    )

    # Media y var-covar anualizadas (según tu módulo)
    mu_annual, cov_annual = annual_mean_cov(returns_matrix)

    # Índices normalizados de activos
    price_index_assets = build_price_index(returns_matrix, base=100.0)
    price_index_assets["date"] = dates

    # Variables compartidas
    portfolio_index = None
    metrics_portfolio = None
    portfolio_returns = None

    # ======================
    # Pestañas principales
    # ======================
    tab_datos, tab_portafolio, tab_benchmark, tab_opt, tab_bl = st.tabs(
        [
            "Mercado",
            "Portafolio (manual)",
            "Benchmark",
            "Optimización (Markowitz)",
            "Black-Litterman",
        ]
    )

    # -------------------------------------------------
    # Tab 1: Mercado
    # -------------------------------------------------
    with tab_datos:
        st.subheader("Mercado (universo seleccionado)")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Observaciones", f"{len(df_returns):,}")
        with c2:
            horizonte = (dates.iloc[-1] - dates.iloc[0]).days / 365
            st.metric("Horizonte", f"{horizonte:,.1f} años")
        with c3:
            st.metric("ETF", f"{len(tickers)}")

        st.markdown("### Retornos diarios (primeras filas)")
        st.dataframe(df_returns.head(), use_container_width=True)

        st.markdown("### Índices normalizados (base 100)")
        df_plot = price_index_assets.melt(id_vars="date", var_name="Ticker", value_name="Índice")
        fig = px.line(df_plot, x="date", y="Índice", color="Ticker", title=f"Índices – {universe['nombre']}")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="Índice (base 100)")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Tab 2: Portafolio manual
    # -------------------------------------------------
    with tab_portafolio:
        st.subheader("Portafolio manual")
        st.caption("Define pesos relativos (pueden ser positivos o negativos). Se normaliza para que sumen 1.")

        cols = st.columns(2)
        raw_weights = []
        for i, t in enumerate(tickers):
            with cols[i % 2]:
                w = st.slider(
                    f"Peso relativo – {t}",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                )
            raw_weights.append(w)

        raw_weights = np.array(raw_weights, dtype=float)
        if raw_weights.sum() == 0:
            st.warning("La suma de pesos es 0. Ajusta sliders.")
        else:
            weights = raw_weights / raw_weights.sum()

            st.markdown("### Pesos (normalizados)")
            df_w = pd.DataFrame({"Ticker": tickers, "Peso": weights})
            st.dataframe(df_w.style.format({"Peso": "{:.4f}"}), use_container_width=True)

            portfolio_returns = returns_matrix @ weights
            portfolio_index = build_price_index(portfolio_returns.to_frame("Portafolio"), base=100.0)
            portfolio_index["date"] = dates

            metrics_portfolio = compute_portfolio_metrics(
                portfolio_returns=portfolio_returns,
                rf_annual=rf_annual,
                benchmark_returns=benchmark_returns,
            )

            st.markdown("### Indicadores clave")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Rend. anual", format_pct(metrics_portfolio.get("mean_annual", np.nan)))
            with k2:
                st.metric("Vol. anual", format_pct(metrics_portfolio.get("vol_annual", np.nan)))
            with k3:
                v = metrics_portfolio.get("sharpe", np.nan)
                st.metric("Sharpe", "—" if np.isnan(v) else f"{v:.2f}")
            with k4:
                st.metric("Max DD", format_pct(metrics_portfolio.get("max_drawdown", np.nan)))

            st.markdown("### Evolución (base 100)")
            fig_p = px.line(portfolio_index, x="date", y="Portafolio", title="Índice del portafolio (base 100)")
            fig_p.update_layout(xaxis_title="Fecha", yaxis_title="Índice")
            st.plotly_chart(fig_p, use_container_width=True)

            st.markdown("### Métricas detalladas")
            df_m = pd.DataFrame(
                [{"Métrica": METRIC_LABELS.get(k, k), "Valor": v} for k, v in metrics_portfolio.items()]
            )
            st.dataframe(df_m.style.format({"Valor": "{:.6f}"}), use_container_width=True)

    # -------------------------------------------------
    # Tab 3: Benchmark
    # -------------------------------------------------
    with tab_benchmark:
        st.subheader("Benchmark (portafolio de referencia)")

        if benchmark_returns is None or benchmark_weights_vec is None:
            st.warning("No se pudo construir el benchmark. Revisa BENCHMARK_WEIGHTS.")
        elif portfolio_index is None or metrics_portfolio is None:
            st.info("Primero construye un portafolio manual para comparar.")
        else:
            st.markdown("### Pesos del benchmark")
            df_bw = pd.DataFrame({"Ticker": returns_matrix.columns, "Peso benchmark": benchmark_weights_vec})
            st.dataframe(df_bw.style.format({"Peso benchmark": "{:.4f}"}), use_container_width=True)

            benchmark_index = build_price_index(benchmark_returns.to_frame("Benchmark"), base=100.0)
            benchmark_index["date"] = dates

            st.markdown("### Evolución comparada (base 100)")
            df_comp = pd.merge(portfolio_index, benchmark_index, on="date", how="inner")
            fig_c = px.line(df_comp, x="date", y=["Portafolio", "Benchmark"], title="Portafolio vs Benchmark (base 100)")
            fig_c.update_layout(xaxis_title="Fecha", yaxis_title="Índice")
            st.plotly_chart(fig_c, use_container_width=True)

            st.markdown("### Métricas: Portafolio vs Benchmark")
            benchmark_metrics = compute_portfolio_metrics(
                portfolio_returns=benchmark_returns,
                rf_annual=rf_annual,
                benchmark_returns=benchmark_returns,
            )

            df_cmp = pd.DataFrame(
                [
                    {
                        "Métrica": METRIC_LABELS.get(k, k),
                        "Portafolio": metrics_portfolio.get(k, np.nan),
                        "Benchmark": benchmark_metrics.get(k, np.nan),
                    }
                    for k in metrics_portfolio.keys()
                ]
            )
            st.dataframe(df_cmp.style.format({"Portafolio": "{:.4f}", "Benchmark": "{:.4f}"}), use_container_width=True)

    # -------------------------------------------------
    # Tab 4: Optimización Markowitz
    # -------------------------------------------------
    with tab_opt:
        st.subheader("Optimización (Markowitz)")
        st.caption("Optimización con μ y Σ anualizados del universo seleccionado (restricción long-only en tu módulo).")

        opt_choice = st.selectbox(
            "Criterio",
            ["Mínima varianza", "Máximo Sharpe", "Retorno objetivo"],
            key="opt_choice_mvo",
        )

        if opt_choice == "Retorno objetivo":
            min_ret = float(np.min(mu_annual))
            max_ret = float(np.max(mu_annual))
            default_ret = float(np.mean(mu_annual))

            target_ret = st.slider(
                "Retorno anual objetivo",
                min_value=min_ret,
                max_value=max_ret,
                value=default_ret,
                step=0.005,
                format="%.3f",
                key="target_mvo",
            )
            opt_weights = target_return_portfolio(
                mu_annual=mu_annual,
                cov_annual=cov_annual,
                target_return=target_ret,
                allow_short=False,
            )
        elif opt_choice == "Máximo Sharpe":
            opt_weights = max_sharpe_portfolio(mu_annual=mu_annual, cov_annual=cov_annual, rf_annual=rf_annual)
        else:
            opt_weights = minimize_volatility(mu_annual=mu_annual, cov_annual=cov_annual)

        st.markdown("### Pesos óptimos")
        df_ow = pd.DataFrame({"Ticker": tickers, "Peso óptimo": opt_weights})
        st.dataframe(df_ow.style.format({"Peso óptimo": "{:.4f}"}), use_container_width=True)

        opt_returns = returns_matrix @ opt_weights
        opt_index = build_price_index(opt_returns.to_frame("Óptimo"), base=100.0)
        opt_index["date"] = dates

        opt_metrics = compute_portfolio_metrics(
            portfolio_returns=opt_returns,
            rf_annual=rf_annual,
            benchmark_returns=benchmark_returns,
        )

        st.markdown("### Indicadores clave")
        a, b, c, d = st.columns(4)
        with a:
            st.metric("Rend. anual", format_pct(opt_metrics.get("mean_annual", np.nan)))
        with b:
            st.metric("Vol. anual", format_pct(opt_metrics.get("vol_annual", np.nan)))
        with c:
            v = opt_metrics.get("sharpe", np.nan)
            st.metric("Sharpe", "—" if np.isnan(v) else f"{v:.2f}")
        with d:
            st.metric("Max DD", format_pct(opt_metrics.get("max_drawdown", np.nan)))

        st.markdown("### Evolución (base 100)")
        fig_o = px.line(opt_index, x="date", y="Óptimo", title=f"Índice del portafolio óptimo – {opt_choice}")
        fig_o.update_layout(xaxis_title="Fecha", yaxis_title="Índice")
        st.plotly_chart(fig_o, use_container_width=True)

        st.markdown("### Métricas detalladas")
        df_optm = pd.DataFrame(
            [{"Métrica": METRIC_LABELS.get(k, k), "Valor": v} for k, v in opt_metrics.items()]
        )
        st.dataframe(df_optm.style.format({"Valor": "{:.6f}"}), use_container_width=True)

    # -------------------------------------------------
    # Tab 5: Black-Litterman
    # -------------------------------------------------
    with tab_bl:
        st.subheader("Black-Litterman")
        st.caption(
            "Se construye un prior de equilibrio Π con el benchmark como portafolio de mercado, "
            "y se combinan vistas (P, Q) con una Ω ajustada por confianza."
        )

        if benchmark_returns is None or benchmark_weights_vec is None:
            st.warning("No hay benchmark disponible para construir el prior (Π).")
            st.stop()

        # --- Parámetros BL
        c1, c2, c3 = st.columns(3)
        with c1:
            tau = st.number_input("τ (tau)", min_value=0.001, max_value=0.200, value=0.025, step=0.005, format="%.3f")
        with c2:
            use_implied_lambda = st.checkbox("λ implícito (desde benchmark)", value=True)
        with c3:
            lam_manual = st.number_input("λ manual", min_value=0.1, max_value=20.0, value=2.5, step=0.1)

        w_mkt = benchmark_weights_vec.copy()
        lam = implied_risk_aversion(mu_annual=mu_annual, cov_annual=cov_annual, w_mkt=w_mkt, rf_annual=rf_annual) if use_implied_lambda else lam_manual

        if np.isnan(lam) or lam <= 0:
            st.error("No se pudo calcular λ. Revisa datos o rango de fechas.")
            st.stop()

        pi = equilibrium_pi(cov_annual=cov_annual, w_mkt=w_mkt, lam=lam)

        st.markdown("### Prior (equilibrio)")
        st.write(f"λ: **{lam:.4f}** | τ: **{tau:.3f}**")
        df_pi = pd.DataFrame({"Ticker": tickers, "Π (prior anual)": pi})
        st.dataframe(df_pi.style.format({"Π (prior anual)": "{:.6f}"}), use_container_width=True)

        st.markdown("### Vistas (Views)")
        st.caption("Q está en términos anuales (ej. 0.02 = 2%).")

        k = st.number_input("Número de vistas (K)", min_value=1, max_value=6, value=2, step=1, key="k_views")

        views = []
        confs = []

        for i in range(int(k)):
            st.markdown(f"**Vista {i+1}**")
            colA, colB, colC, colD = st.columns([1.1, 1.4, 1.0, 1.2])

            with colA:
                vtype = st.selectbox("Tipo", ["Absoluta", "Relativa"], key=f"vtype_{i}")

            with colB:
                if vtype == "Absoluta":
                    asset = st.selectbox("Activo", tickers, key=f"abs_asset_{i}")
                    asset_long, asset_short = None, None
                else:
                    asset_long = st.selectbox("Long (mejor)", tickers, key=f"rel_long_{i}")
                    asset_short = st.selectbox("Short (peor)", tickers, key=f"rel_short_{i}")
                    asset = None

            with colC:
                q = st.number_input("Q (anual)", value=0.02, step=0.005, format="%.3f", key=f"q_{i}")

            with colD:
                conf = st.slider("Confianza", min_value=0.10, max_value=0.95, value=0.60, step=0.05, key=f"conf_{i}")

            if vtype == "Absoluta":
                views.append({"type": "absolute", "asset": asset, "q": float(q)})
            else:
                if asset_long == asset_short:
                    st.warning("En vistas relativas, Long y Short deben ser distintos.")
                    st.stop()
                views.append({"type": "relative", "asset_long": asset_long, "asset_short": asset_short, "q": float(q)})

            confs.append(float(conf))

        P, Q = build_views_from_pairs(tickers=tickers, views=views)

        # Ω tipo He-Litterman con ajuste por confianza (más confianza => menor var)
        base_mid = P @ (tau * cov_annual) @ P.T
        omega_diag = np.diag(base_mid)
        Omega = np.diag(omega_diag / np.array(confs, dtype=float))

        mu_bl, cov_bl, _ = black_litterman_posterior(
            cov_annual=cov_annual,
            pi=pi,
            P=P,
            Q=Q,
            tau=tau,
            Omega=Omega,
        )

        st.markdown("### Posterior (BL)")
        df_bl = pd.DataFrame({"Ticker": tickers, "μ_BL (anual)": mu_bl, "Π (prior)": pi})
        st.dataframe(df_bl.style.format({"μ_BL (anual)": "{:.6f}", "Π (prior)": "{:.6f}"}), use_container_width=True)

        st.markdown("### Optimización con Black-Litterman")
        opt_choice_bl = st.selectbox(
            "Criterio (con BL)",
            ["Mínima varianza", "Máximo Sharpe", "Retorno objetivo"],
            key="opt_choice_bl",
        )

        if opt_choice_bl == "Retorno objetivo":
            min_ret = float(np.min(mu_bl))
            max_ret = float(np.max(mu_bl))
            default_ret = float(np.mean(mu_bl))
            target_ret = st.slider(
                "Retorno anual objetivo (BL)",
                min_value=min_ret,
                max_value=max_ret,
                value=default_ret,
                step=0.005,
                format="%.3f",
                key="target_bl",
            )
            w_bl = target_return_portfolio(
                mu_annual=mu_bl,
                cov_annual=cov_bl,
                target_return=target_ret,
                allow_short=False,
            )
        elif opt_choice_bl == "Máximo Sharpe":
            w_bl = max_sharpe_portfolio(mu_annual=mu_bl, cov_annual=cov_bl, rf_annual=rf_annual)
        else:
            w_bl = minimize_volatility(mu_annual=mu_bl, cov_annual=cov_bl)

        st.markdown("### Pesos óptimos (BL)")
        df_wbl = pd.DataFrame({"Ticker": tickers, "Peso óptimo (BL)": w_bl})
        st.dataframe(df_wbl.style.format({"Peso óptimo (BL)": "{:.4f}"}), use_container_width=True)

        bl_returns = returns_matrix @ w_bl
        bl_index = build_price_index(bl_returns.to_frame("BL Óptimo"), base=100.0)
        bl_index["date"] = dates

        st.markdown("### Evolución (base 100)")
        fig_bl = px.line(bl_index, x="date", y="BL Óptimo", title=f"Índice del portafolio BL – {opt_choice_bl}")
        fig_bl.update_layout(xaxis_title="Fecha", yaxis_title="Índice")
        st.plotly_chart(fig_bl, use_container_width=True)

        bl_metrics = compute_portfolio_metrics(
            portfolio_returns=bl_returns,
            rf_annual=rf_annual,
            benchmark_returns=benchmark_returns,
        )
        st.markdown("### Métricas (BL)")
        df_blm = pd.DataFrame(
            [{"Métrica": METRIC_LABELS.get(k, k), "Valor": v} for k, v in bl_metrics.items()]
        )
        st.dataframe(df_blm.style.format({"Valor": "{:.6f}"}), use_container_width=True)

else:
    st.error("Corrige las fechas en el panel lateral para continuar.")
