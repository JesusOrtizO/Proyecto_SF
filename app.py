# app.py
"""
Aplicación de Seminario de Finanzas.

Funcionalidad:
- Selector de universo de inversión (Regiones o Sectores USA).
- Carga de rendimientos diarios sincronizados desde archivos locales usando sf_library.
- Construcción de índices de valor normalizados (base 100).
- Pestañas para:
    * Datos de mercado.
    * Portafolio arbitrario (pesos definidos por el usuario).
    * Comparación con benchmark (portafolio de referencia con pesos fijos).
    * Portafolios optimizados (mínima varianza, máximo Sharpe, retorno objetivo).
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
)

# ============================================================
# Definición de universos de inversión y benchmark de referencia
# ============================================================

UNIVERSE_REGIONS = {
    "nombre": "Regiones (ETF)",
    "tickers": ["SPLG", "EWC", "IEUR", "EEM", "EWJ"],
}

UNIVERSE_SECTORS = {
    "nombre": "Sectores USA (ETF)",
    "tickers": [
        "XLC",
        "XLY",
        "XLP",
        "XLE",
        "XLF",
        "XLV",
        "XLI",
        "XLB",
        "XLRE",
        "XLK",
        "XLU",
    ],
}

UNIVERSES: Dict[str, Dict] = {
    "Regiones": UNIVERSE_REGIONS,
    "Sectores USA": UNIVERSE_SECTORS,
}

# Pesos del benchmark por universo (según enunciado del proyecto)
# Valores expresados en proporciones (no en %), pero se vuelven a normalizar por seguridad.
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

# Etiquetas amigables para las métricas
METRIC_LABELS = {
    "mean_annual": "Rendimiento anual esperado",
    "vol_annual": "Volatilidad anual",
    "skew": "Asimetría (skewness)",
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


@st.cache_data
def load_synced_returns(
    tickers: List[str],
    start: str,
    end: str,
    data_dir: str = "MarketData",
) -> pd.DataFrame:
    """
    Descarga datos (si es necesario), sincroniza series y devuelve
    un DataFrame con retornos diarios en el rango de fechas indicado.

    La función está cacheada para evitar descargas y cálculos repetidos.
    El cache se invalida cuando cambian los argumentos.
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
    """
    A partir de un DataFrame de rendimientos por periodo (sin columna 'date'),
    construye índices de valor normalizados a 'base' en el primer periodo.
    """
    cumulative = (1.0 + returns).cumprod()
    index = base * cumulative
    return index


def compute_benchmark(
    returns_matrix: pd.DataFrame, universe_key: str
) -> Tuple[pd.Series, np.ndarray]:
    """
    Construye la serie de rendimientos del benchmark a partir de los pesos
    definidos para cada universo. Devuelve:

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

    weights_vec = weights_vec / total  # normalizar por seguridad
    benchmark_returns = returns_matrix @ weights_vec

    return benchmark_returns, weights_vec


# ============================
# Configuración de la página
# ============================

st.set_page_config(
    page_title="Seminario de Finanzas - Portafolios",
    layout="wide",
)

# CSS ligero para pulir estilo
st.markdown(
    """
    <style>
    h1 {
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    h3 {
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }
    .stDataFrame tbody td {
        font-size: 0.85rem;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Seminario de Finanzas – Análisis de ETF y Portafolios")
st.markdown(
    "Aplicación interactiva para análisis de **ETF**, construcción de portafolios "
    "arbitrarios y portafolios optimizados bajo el enfoque de Markowitz. "
    "Incluye comparación contra un portafolio de referencia (benchmark) definido "
    "en el enunciado del proyecto."
)
st.markdown("---")

# ======================
# Configuración lateral
# ======================

st.sidebar.header("Configuración de datos")

universe_key = st.sidebar.selectbox(
    "Universo de inversión",
    options=list(UNIVERSES.keys()),
    index=0,
)

universe = UNIVERSES[universe_key]
tickers = universe["tickers"]

st.sidebar.markdown(f"**Universo seleccionado:** {universe['nombre']}")
st.sidebar.write("Tickers:")
st.sidebar.write(", ".join(tickers))

default_start = dt.date(2015, 1, 1)
default_end = dt.date.today()

start_date = st.sidebar.date_input(
    "Fecha inicio",
    value=default_start,
    help="Formato AAAA-MM-DD",
)
end_date = st.sidebar.date_input(
    "Fecha fin",
    value=default_end,
    help="Formato AAAA-MM-DD",
)

if start_date > end_date:
    st.sidebar.error("La fecha de inicio debe ser anterior o igual a la fecha de fin.")

rf_annual = st.sidebar.number_input(
    "Tasa libre de riesgo anual",
    min_value=0.0,
    max_value=0.20,
    value=0.04,
    step=0.005,
    format="%.3f",
    help="Ejemplo: 0.040 corresponde a una tasa anual de 4%.",
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

    # Benchmark del universo (portafolio de referencia con pesos fijos)
    benchmark_returns, benchmark_weights_vec = compute_benchmark(
        returns_matrix=returns_matrix,
        universe_key=universe_key,
    )

    # Media y var-covar anualizadas de los activos del universo
    mu_annual, cov_annual = annual_mean_cov(returns_matrix)

    # Índices normalizados de cada activo
    price_index_assets = build_price_index(returns_matrix, base=100.0)
    price_index_assets["date"] = dates

    # Variables que se usarán en varias pestañas
    portfolio_index = None          # índice del portafolio arbitrario
    metrics_portfolio = None        # métricas del portafolio arbitrario
    portfolio_returns = None        # serie de rendimientos del portafolio arbitrario

    # ======================
    # Pestañas principales
    # ======================

    tab_datos, tab_portafolio, tab_benchmark, tab_opt = st.tabs(
        [
            "Datos de mercado",
            "Portafolio arbitrario",
            "Comparación con Benchmark",
            "Portafolios optimizados",
        ]
    )

    # -------------------------------------------------
    # Tab 1: Datos de mercado
    # -------------------------------------------------
    with tab_datos:
        st.subheader("1. Retornos diarios sincronizados")
        st.caption(
            "Los rendimientos se calculan con datos diarios sincronizados para "
            "todos los ETF del universo seleccionado."
        )
        st.write(
            f"Número de observaciones en el rango seleccionado: {len(df_returns)}"
        )
        st.dataframe(df_returns.head())

        st.subheader("2. Índices de valor normalizados (base 100)")
        df_plot = price_index_assets.melt(
            id_vars="date", var_name="Ticker", value_name="Índice"
        )
        fig = px.line(
            df_plot,
            x="date",
            y="Índice",
            color="Ticker",
            title=f"Índices de valor normalizados – {universe['nombre']}",
        )
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Índice (base 100)",
            legend_title="Ticker",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Tab 2: Portafolio arbitrario
    # ---------------------------
    with tab_portafolio:
        st.subheader("1. Configuración de pesos del portafolio")
        st.markdown(
            "Defina pesos relativos (positivos o negativos) para cada ETF. "
            "Los pesos se normalizan de forma automática para que la suma sea igual a 1."
        )

        cols = st.columns(2)
        raw_weights = []

        for i, ticker in enumerate(tickers):
            col = cols[i % 2]
            with col:
                w = st.slider(
                    f"Peso relativo para {ticker}",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                )
            raw_weights.append(w)

        raw_weights_array = np.array(raw_weights, dtype=float)
        weight_sum = raw_weights_array.sum()

        if weight_sum == 0.0:
            st.warning(
                "La suma de los pesos es cero. Ajuste los sliders para obtener un portafolio válido."
            )
        else:
            weights = raw_weights_array / weight_sum

            st.markdown("#### Pesos normalizados del portafolio")
            df_weights = pd.DataFrame({"Ticker": tickers, "Peso": weights})
            st.dataframe(df_weights.style.format({"Peso": "{:.4f}"}))

            # Rendimientos y valor del portafolio
            portfolio_returns = returns_matrix @ weights
            portfolio_index = build_price_index(
                portfolio_returns.to_frame("Portfolio"),
                base=100.0,
            )
            portfolio_index["date"] = dates

            st.subheader("2. Desempeño del portafolio")

            fig_port = px.line(
                portfolio_index,
                x="date",
                y="Portfolio",
                title="Índice de valor del portafolio arbitrario (base 100)",
            )
            fig_port.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Índice (base 100)",
            )
            st.plotly_chart(fig_port, use_container_width=True)

            # Cálculo de métricas
            st.subheader("3. Métricas del portafolio")

            metrics_portfolio = compute_portfolio_metrics(
                portfolio_returns=portfolio_returns,
                rf_annual=rf_annual,
                benchmark_returns=benchmark_returns,
            )

            rows = []
            for key, value in metrics_portfolio.items():
                nombre = METRIC_LABELS.get(key, key)
                rows.append({"Métrica": nombre, "Valor": value})

            df_metrics = pd.DataFrame(rows)
            st.dataframe(df_metrics.style.format({"Valor": "{:.6f}"}))

    # -------------------------------------------------------
    # Tab 3: Comparación contra Benchmark
    # -------------------------------------------------------
    with tab_benchmark:
        st.subheader("1. Portafolio benchmark del universo")

        if benchmark_returns is None or benchmark_weights_vec is None:
            st.warning(
                "No se ha podido construir el benchmark para el universo seleccionado. "
                "Revise la definición de BENCHMARK_WEIGHTS."
            )
        elif portfolio_index is None or metrics_portfolio is None:
            st.info(
                "Primero define un portafolio arbitrario (pestaña *Portafolio arbitrario*) "
                "para poder compararlo contra el benchmark."
            )
        else:
            # Tabla de pesos del benchmark alineados con los tickers del universo
            df_bench_w = pd.DataFrame(
                {
                    "Ticker": returns_matrix.columns,
                    "Peso benchmark": benchmark_weights_vec,
                }
            )
            st.dataframe(df_bench_w.style.format({"Peso benchmark": "{:.4f}"}))

            # Construcción del índice de valor del benchmark
            benchmark_index = build_price_index(
                benchmark_returns.to_frame("Benchmark"),
                base=100.0,
            )
            benchmark_index["date"] = dates

            st.subheader("2. Comparación de índices normalizados")

            df_comp = pd.merge(
                portfolio_index,
                benchmark_index,
                on="date",
                how="inner",
            )

            fig_comp = px.line(
                df_comp,
                x="date",
                y=["Portfolio", "Benchmark"],
                title="Comparación portafolio arbitrario vs benchmark (base 100)",
            )
            fig_comp.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Índice (base 100)",
                legend_title="Serie",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.subheader("3. Comparación de métricas")

            benchmark_metrics = compute_portfolio_metrics(
                portfolio_returns=benchmark_returns,
                rf_annual=rf_annual,
                benchmark_returns=benchmark_returns,
            )

            rows_comp = []
            for key, value in metrics_portfolio.items():
                nombre = METRIC_LABELS.get(key, key)
                rows_comp.append(
                    {
                        "Métrica": nombre,
                        "Portafolio": value,
                        "Benchmark": benchmark_metrics.get(key, np.nan),
                    }
                )

            df_metrics_comp = pd.DataFrame(rows_comp)

            st.dataframe(
                df_metrics_comp.style.format(
                    {"Portafolio": "{:.4f}", "Benchmark": "{:.4f}"}
                )
            )

    # -------------------------------------------------------
    # Tab 4: Portafolios optimizados
    # -------------------------------------------------------
    with tab_opt:
        st.subheader("1. Selección de criterio de optimización")
        st.caption(
            "Los portafolios se construyen bajo el enfoque clásico de Markowitz, "
            "a partir de rendimientos y matriz de varianza-covarianza anualizados."
        )

        opt_choice = st.selectbox(
            "Tipo de portafolio óptimo",
            ["Mínima varianza", "Máximo Sharpe", "Retorno objetivo"],
        )

        if opt_choice == "Retorno objetivo":
            min_ret = float(mu_annual.min())
            max_ret = float(mu_annual.max())
            default_ret = float(mu_annual.mean())

            target_ret = st.slider(
                "Retorno anual objetivo",
                min_value=min_ret,
                max_value=max_ret,
                value=default_ret,
                step=0.005,
                format="%.3f",
            )
            opt_weights = target_return_portfolio(
                mu_annual=mu_annual,
                cov_annual=cov_annual,
                target_return=target_ret,
                allow_short=False,
            )
        elif opt_choice == "Máximo Sharpe":
            opt_weights = max_sharpe_portfolio(
                mu_annual=mu_annual,
                cov_annual=cov_annual,
                rf_annual=rf_annual,
            )
        else:  # Mínima varianza
            opt_weights = minimize_volatility(
                mu_annual=mu_annual,
                cov_annual=cov_annual,
            )

        st.subheader("2. Pesos del portafolio óptimo")

        df_opt_weights = pd.DataFrame({"Ticker": tickers, "Peso óptimo": opt_weights})
        st.dataframe(df_opt_weights.style.format({"Peso óptimo": "{:.4f}"}))

        # Rendimientos diarios del portafolio optimizado
        opt_returns = returns_matrix @ opt_weights

        # Índice de valor del portafolio optimizado
        opt_index = build_price_index(
            opt_returns.to_frame("Óptimo"),
            base=100.0,
        )
        opt_index["date"] = dates

        st.subheader("3. Desempeño del portafolio optimizado")

        fig_opt = px.line(
            opt_index,
            x="date",
            y="Óptimo",
            title=f"Índice de valor del portafolio óptimo – {opt_choice} – {universe['nombre']}",
        )
        fig_opt.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Índice (base 100)",
        )
        st.plotly_chart(fig_opt, use_container_width=True)

        # Métricas del portafolio optimizado
        st.subheader("4. Métricas del portafolio optimizado")

        opt_metrics = compute_portfolio_metrics(
            portfolio_returns=opt_returns,
            rf_annual=rf_annual,
            benchmark_returns=benchmark_returns,
        )

        rows_opt = []
        for key, value in opt_metrics.items():
            nombre = METRIC_LABELS.get(key, key)
            rows_opt.append({"Métrica": nombre, "Valor": value})

        df_opt_metrics = pd.DataFrame(rows_opt)

        st.dataframe(df_opt_metrics.style.format({"Valor": "{:.6f}"}))
