# Seminario de Finanzas — ETF y Portafolios

Aplicación interactiva en **Streamlit** para analizar ETFs, construir portafolios y compararlos contra un benchmark de referencia. Incluye optimización de portafolios por **Markowitz** (media-varianza) y por **Black-Litterman**, además de métricas de riesgo y desempeño estándar de la industria (Sharpe, Sortino, VaR, CVaR, Calmar, Treynor, Information Ratio).

**Demo en vivo:** https://proyectosf-jpnfj7bhb9nvekwspq9pae.streamlit.app/

Proyecto desarrollado para la materia **Seminario de Finanzas**, Facultad de Ciencias, UNAM.

**Integrantes:** Ortiz Ordoñez Jesús Alberto · Terres Rodríguez Diana Laura · Alonso Bartolo Karla Rubí · Carbajal Sánchez Miguel

---

## 1. Qué hace la app

- **Dos universos de inversión** seleccionables: ETFs por región geográfica (`SPLG`, `EWC`, `IEUR`, `EEM`, `EWJ`) o ETFs sectoriales de EE. UU. (`XLC`, `XLY`, `XLP`, `XLE`, `XLF`, `XLV`, `XLI`, `XLB`, `XLRE`, `XLK`, `XLU`).
- **Mercado:** visualiza los rendimientos diarios y los índices de precio normalizados (base 100) de cada ETF del universo elegido.
- **Portafolio manual:** el usuario define pesos con sliders y ve sus métricas de riesgo/desempeño en tiempo real.
- **Benchmark:** compara el portafolio manual contra un portafolio de referencia con pesos fijos por universo.
- **Optimización (Markowitz):** calcula el portafolio de mínima varianza, máximo Sharpe o de retorno objetivo, sujeto a `w_i >= 0` y `Σw_i = 1`.
- **Black-Litterman:** construye un prior de equilibrio a partir del benchmark, permite capturar vistas del usuario (absolutas o relativas entre activos) y calcula el posterior (μ, Σ) para optimizar sobre él.

## 2. Estructura de archivos

### Núcleo de la aplicación (lo que corre en producción)

| Archivo | Qué hace |
|---|---|
| **`app.py`** | Punto de entrada de Streamlit. Arma la interfaz (sidebar de parámetros, pestañas), orquesta la carga de datos y llama a las funciones de `sf_library.py`, `metrics.py` y `optimization.py`. No contiene lógica financiera propia — es la capa de presentación. |
| **`sf_library.py`** | Capa de datos. Descarga precios históricos vía `yfinance`, los cachea como CSV en `MarketData/`, calcula rendimientos diarios por ticker y sincroniza varias series en un solo DataFrame por fecha común. Incluye validaciones defensivas (columnas MultiIndex de yfinance, respuestas corruptas de Yahoo, tolerancia a que un ticker individual falle sin tumbar todo el universo). |
| **`metrics.py`** | Funciones puras de métricas de riesgo/desempeño de un portafolio: rendimiento y volatilidad anualizados, máximo drawdown, Sharpe, Sortino, Calmar, beta, Treynor, Information Ratio, VaR y CVaR históricos. `compute_portfolio_metrics()` es la función que consume `app.py` para llenar cada pestaña. |
| **`optimization.py`** | Funciones de optimización de portafolios con `scipy.optimize`: mínima volatilidad, máximo Sharpe y retorno objetivo (todas long-only), más la implementación completa de Black-Litterman (aversión al riesgo implícita, prior de equilibrio Π, posterior bayesiano, construcción de matrices de vistas P/Q). |
| **`requirements.txt`** | Dependencias con versión fija (Streamlit, yfinance, pandas, numpy, scipy, plotly, matplotlib) para evitar que una actualización silenciosa de alguna librería rompa la app sin previo aviso. |

### Scripts de desarrollo / exploración (no los usa la app en producción)

Estos archivos documentan el proceso de investigación y las pruebas que dieron origen a la lógica final en `sf_library.py`, `metrics.py` y `optimization.py`. Se conservan como evidencia del trabajo pero **no son importados por `app.py`**:

| Archivo | Qué explora |
|---|---|
| `clase_capm.py` | Regresión CAPM (alpha, beta) de un activo contra el S&P 500, exportado de un notebook de Colab. |
| `clase_metricas.py` | Primeras versiones "sueltas" de Sharpe, Sortino, Treynor, Information Ratio y Calmar, antes de consolidarse en `metrics.py`. |
| `cova.py` | Exploración de matrices de varianza-covarianza y correlación entre ETFs sectoriales, con mapa de calor (`seaborn`). |
| `min_varianza.py` | Primeras implementaciones del portafolio de mínima varianza (por autovalores y por `scipy.optimize`), frontera eficiente y comparación de portafolios (mínima varianza, máximo retorno, máximo Sharpe) — precursor directo de `optimization.py`. |
| `test_datos.py` | Script de prueba manual para verificar que la descarga y sincronización de datos (`descargar_tickers` + `sync_timeseries` de `sf_library.py`) funcionara antes de integrarla a la app. |
| `integrantes.txt` | Lista de integrantes del equipo, para la entrega del proyecto. |

## 3. Cómo correrlo localmente

```bash
git clone https://github.com/JesusOrtizO/Proyecto_SF.git
cd Proyecto_SF
pip install -r requirements.txt
streamlit run app.py
```

La primera vez que se ejecuta, descarga los precios históricos de los ETFs del universo seleccionado y los guarda en una carpeta local `MarketData/` (se crea automáticamente) para acelerar corridas futuras.

## 4. Notas técnicas / limitaciones conocidas

- Los datos de mercado provienen de **Yahoo Finance vía `yfinance`**, una librería no oficial. Ocasionalmente Yahoo puede fallar en devolver datos para un ticker específico (rate-limiting o problemas temporales de su API). La app está diseñada para **degradarse con elegancia**: si un ETF del universo no puede descargarse, se excluye del análisis y se avisa en pantalla, en vez de que falle la aplicación completa.
- Las matrices de covarianza y los rendimientos esperados se anualizan asumiendo 252 días hábiles por año.
- Todas las optimizaciones son *long-only* (sin ventas en corto): los pesos están acotados entre 0 y 1.

## 5. Stack técnico

**Python** · **Streamlit** · **pandas** / **numpy** · **scipy.optimize** · **yfinance** · **Plotly**
