import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def descargar_tickers(tickers, carpeta='MarketData', start='2000-01-01', end=None):
    """
    Descarga datos históricos de una lista de tickers usando yfinance
    y guarda un archivo CSV por cada ticker que tenga datos.

    Parámetros:
    - tickers: lista de símbolos (ej. ['AAPL', 'MSFT', '^GSPC'])
    - carpeta: carpeta donde se guardarán los CSV
    - start: fecha de inicio (YYYY-MM-DD)
    - end: fecha de fin (YYYY-MM-DD, opcional)
    """
    # Si no se especifica fecha final, usa la fecha actual
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    # Crear carpeta si no existe
    os.makedirs(carpeta, exist_ok=True)

    for tic in tickers:
        print(f"Descargando datos de {tic}...")
        try:
            data = yf.download(tic, start=start, end=end, progress=False)

            # Si no hay datos, saltar
            if data.empty:
                print(f"⚠️ No se encontraron datos para {tic}, se omite.")
                continue

            # Dejar solo las columnas necesarias
            df = data.reset_index()[['Date', 'Close']]

            # Guardar archivo CSV
            ruta = os.path.join(carpeta, f"{tic}.csv")
            df.to_csv(ruta, index=False)

        except Exception as e:
            print(f"Error descargando {tic}: {e}")
            continue



def daily_return(ticker, data_dir="MarketData"):
    """
    Carga una serie temporal para un ticker y calcula los rendimientos diarios.

    Comportamiento:
    - Si existe el archivo CSV local en data_dir, lo lee.
    - Si NO existe, intenta descargar los datos desde yfinance,
      construye el DataFrame y opcionalmente guarda el CSV para usos futuros.

    Parámetros
    ----------
    ticker : str
        Símbolo del activo (por ejemplo: 'AAPL').
    data_dir : str, opcional
        Directorio donde se encuentran (o se guardarán) los archivos CSV.
        Por defecto 'MarketData'.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas: ['date', 'close', 'return']
    """
    # Construir ruta del archivo local
    folder_path = os.path.join(os.getcwd(), data_dir)
    file_path = os.path.join(folder_path, f"{ticker}.csv")

    # Si el archivo NO existe, intentamos descargarlo con yfinance
    if not os.path.exists(file_path):
        # Asegurar que la carpeta exista
        os.makedirs(folder_path, exist_ok=True)

        try:
            # Descarga amplia; luego app filtra por fechas
            end_date = datetime.today().strftime("%Y-%m-%d")
            data = yf.download(ticker, start="2000-01-01", end=end_date, progress=False)
        except Exception as e:
            raise FileNotFoundError(
                f"No se pudo descargar datos para {ticker} desde yfinance: {e}"
            )

        if data.empty:
            raise FileNotFoundError(
                f"yfinance no devolvió datos para el ticker {ticker}. "
                "Verifica que el símbolo sea correcto."
            )

        # Dejar solo las columnas necesarias y guardar CSV (opcional pero útil)
        df_raw = data.reset_index()[["Date", "Close"]]
        df_raw.to_csv(file_path, index=False)
        df = df_raw
    else:
        # Leer las columnas necesarias del CSV existente
        df = pd.read_csv(
            file_path,
            usecols=["Date", "Close"],
            parse_dates=["Date"],
        )

    # Limpiar y preparar
    df = (
        df.sort_values("Date")
          .rename(columns={"Date": "date", "Close": "close"})
    )

    # Convertir a numérico (por si hay texto como 'N/A')
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Eliminar filas donde 'close' sea NaN antes del cálculo
    df = df.dropna(subset=["close"])

    # Calcular rendimientos diarios
    df["return"] = df["close"].pct_change()

    # Eliminar la primera fila (NaN en return)
    df = df.dropna(subset=["return"]).reset_index(drop=True)

    return df















































def sync_timeseries(tickers, data_dir="MarketData"):
    """
    Carga y sincroniza series temporales de retornos diarios para varios tickers.

    Parámetros
    ----------
    tickers : list[str]
        Lista de símbolos (por ejemplo: ['XLK', 'XLF', 'XLV']).
    data_dir : str, opcional
        Directorio donde se encuentran los archivos CSV. Por defecto 'MarketData'.

    Retorna
    -------
    df : pd.DataFrame
        DataFrame con las fechas y los retornos sincronizados de cada ticker.
    mtx_var_covar : np.ndarray
        Matriz de varianza-covarianza.
    mtx_correl : np.ndarray
        Matriz de correlaciones.
    """

    # Cargar y preparar todas las series de retornos
    all_returns = []

    for ticker in tickers:
        t = daily_return(ticker, data_dir=data_dir)
        t = t[['date', 'return']].rename(columns={'return': ticker})
        all_returns.append(t)

    # Unir todas las series por la columna 'date' (intersección automática)
    df = all_returns[0]
    for t in all_returns[1:]:
        df = pd.merge(df, t, on='date', how='inner')

    # Limpiar y ordenar
    df = df.dropna().sort_values('date').reset_index(drop=True)

    # Calcular matrices
    returns_only = df.drop(columns='date')
    mtx_var_covar = returns_only.cov().values
    mtx_correl = returns_only.corr().values

    # Mostrar resultados
    print("Primeras filas del DataFrame sincronizado:")
    print(df.head(), "\n")

    print("Matriz Varianza-Covarianza:")
    print(mtx_var_covar, "\n")

    print("Matriz de Correlaciones:")
    print(mtx_correl, "\n")

    return df, mtx_var_covar, mtx_correl



