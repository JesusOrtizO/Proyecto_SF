# test_datos.py
"""
Prueba inicial del proyecto:
- Descargar datos de ETF de tu universo
- Calcular retornos diarios
- Sincronizar series

Usando tu propia librería sf_library.py
"""

import sf_library as sfl

TICKERS_REGIONES = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]

print("=== 1) DESCARGANDO DATOS ===")
sfl.descargar_tickers(
    tickers=TICKERS_REGIONES,
    carpeta="MarketData",
    start="2015-01-01",
    end=None
)
print("✔ Descarga completa")

print("\n=== 2) SINCRONIZANDO SERIES ===")
df_sync, var_cov, correl = sfl.sync_timeseries(
    tickers=TICKERS_REGIONES,
    data_dir="MarketData"
)

print("\nPrimeras filas del DataFrame sincronizado:")
print(df_sync.head())

print("\nDimensiones del DataFrame sincronizado:")
print(df_sync.shape)

print("\nMatriz de Var-Covar:")
print(var_cov)

print("\nMatriz de Correlaciones:")
print(correl)

print("\n🎉 ¡Todo correcto! Ya puedes usar estos datos en tu app.")
