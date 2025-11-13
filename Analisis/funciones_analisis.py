
import matplotlib.pyplot as plt  
import pandas as pd
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
# --- Outlier detection by threshold methods ---
def plot_outliers(df, columna: str, color: str='blue', marker: str='o',
                  ax=None, ph: bool=False, z_thresh: float=3.5):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    d = df.copy()

    s = pd.to_numeric(d[columna], errors='coerce')

    # Outliers por naturaleza variables
    if ph:
        mask_ph_out = (s < 0) | (s > 14)
    else:
        mask_ph_out = pd.Series(False, index=d.index)

    # z-score robusto
    if ph:
        base = s[(s >= 0) & (s <= 14)]
    else:
        base = s.copy()
    base = base.dropna()

    if len(base) >= 2:
        med = base.mean()
        mad = (base - med).abs().mean()
        if mad and mad > 0:
            z_rob = (s - med) / (3 * mad)
            mask_rob_out = z_rob.abs() >= z_thresh
        else:
            mask_rob_out = pd.Series(False, index=d.index)
    else:
        mask_rob_out = pd.Series(False, index=d.index)

    # Puntos normales
    mask_ok = (~mask_ph_out) & (~mask_rob_out)
    ax.scatter(d.loc[mask_ok, 'date_time'], s.loc[mask_ok],
               color=color, marker=marker, s=20, label=columna)

    # Outliers pH 
    if mask_ph_out.any():
        ax.scatter(d.loc[mask_ph_out, 'date_time'], s.loc[mask_ph_out],
                   color='red', marker=marker, s=24, label='Outliers pH')

    # Outliers robustos 
    mask_rob_only = mask_rob_out & (~mask_ph_out)
    if mask_rob_only.any():
        ax.scatter(d.loc[mask_rob_only, 'date_time'], s.loc[mask_rob_only],
                   color='orange', marker=marker, s=24, label='Outliers robustos')

    # Estética
    ax.set_title(f'{columna}', fontsize=16)
    ax.set_xlabel('fecha', fontsize=14)
    ax.set_ylabel(columna, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    if mask_ph_out.any() or mask_rob_out.any():
        ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()

def outliers_zscore(series, threshold=3):
    mu = series.mean()
    sigma = series.std()
    z = (series - mu) / sigma
    return z.abs() > threshold

def outliers_iqr(series, k=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return (series < lower) | (series > upper)

def outliers_rolling(series, window=30, k=3):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    mask = (series - rolling_mean).abs() > k * rolling_std
    return mask.fillna(False)
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
# Fechas sin Datos
def fechas_con_y_sin_datos(df, dt_col="date_time", min_rows=1):
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    fechas_validas = df.groupby(df[dt_col].dt.date).size()
    fechas_validas = fechas_validas[fechas_validas >= min_rows]

    fechas_con = set(fechas_validas.index)
    if not fechas_con:
        return {
            "con_datos": [],
            "sin_datos": [],
            "total_con": 0,
            "total_sin": 0,
            "porcentaje_con": 0.0,
            "porcentaje_sin": 0.0
        }

    rango_total = pd.date_range(min(fechas_con), max(fechas_con), freq='D').date
    fechas_sin = sorted(set(rango_total) - fechas_con)

    def agrupar_en_rangos(lista_fechas):
        bloques = []
        if not lista_fechas:
            return bloques
        inicio = fin = lista_fechas[0]
        for fecha in lista_fechas[1:]:
            if (fecha - fin).days == 1:
                fin = fecha
            else:
                bloques.append((inicio.isoformat(), fin.isoformat()))
                inicio = fin = fecha
        bloques.append((inicio.isoformat(), fin.isoformat()))
        return bloques

    total_dias = len(rango_total)
    total_con = len(fechas_con)
    total_sin = len(fechas_sin)

    porcentaje_con = round(100 * total_con / total_dias, 2)
    porcentaje_sin = round(100 * total_sin / total_dias, 2)

    return {
        "con_datos": agrupar_en_rangos(sorted(fechas_con)),
        "sin_datos": agrupar_en_rangos(fechas_sin),
        "total_con": total_con,
        "total_sin": total_sin,
        "porcentaje_con": porcentaje_con,
        "porcentaje_sin": porcentaje_sin
    }

def imprimir_bloques(nombre, bloques, total_dias, porcentaje):
    print(f"{nombre}:")
    total = 0
    for inicio, fin in bloques:
        inicio_dt = pd.to_datetime(inicio).date()
        fin_dt = pd.to_datetime(fin).date()
        dias = (fin_dt - inicio_dt).days + 1
        total += dias
        print(f"[{inicio}, {fin}], {dias} {'día' if dias == 1 else 'días'}")
    print(f"\nTotal {nombre.lower()}: {total} ({porcentaje}%)\n")

def analizar_columnas_por_fecha(df, columnas, dt_col="date_time", min_rows=1):
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])

    for col in columnas:
        print(f"\n==============================")
        print(f" Análisis de: '{col}'")
        print(f"==============================\n")

        # Filtra donde esa columna tiene datos
        df_col = df[~df[col].isna()]

        resultados = fechas_con_y_sin_datos(df_col, dt_col=dt_col, min_rows=min_rows)
        imprimir_bloques("Fechas con datos", resultados["con_datos"], resultados["total_con"], resultados["porcentaje_con"])
        imprimir_bloques("Fechas sin datos", resultados["sin_datos"], resultados["total_sin"], resultados["porcentaje_sin"])

def merge_blocks(blocks, gap=pd.Timedelta(seconds=0)):
    if isinstance(blocks, pd.DataFrame):
        blocks = list(zip(pd.to_datetime(blocks['start']), pd.to_datetime(blocks['end'])))
    elif not blocks:
        return []
    blocks = sorted(blocks, key=lambda x: x[0])
    merged = []
    for b in blocks:
        if not merged:
            merged.append(list(b))
        else:
            last = merged[-1]
            if b[0] <= last[1] + gap:
                last[1] = max(last[1], b[1])
            else:
                merged.append(list(b))
    return [tuple(x) for x in merged]

def plot_raw_with_drift(df, fecha_col, var, blocks, resample=None, color='tab:red', alpha=0.22, show_points=False):

    d = df[[fecha_col, var]].dropna().copy()
    d[fecha_col] = pd.to_datetime(d[fecha_col])
    d = d.sort_values(fecha_col)
    if resample:
        s = d.set_index(fecha_col)[var].resample(resample).median().dropna().reset_index()
    else:
        s = d.rename(columns={var: 'value'}).rename(columns={'value': var})
    plt.figure(figsize=(12, 4))
    if show_points:
        plt.plot(s[fecha_col], s[var], marker='.', linestyle='None', markersize=2)
    else:
        plt.plot(s[fecha_col], s[var])
    for start, end in blocks:
        plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=alpha)
    plt.title(f"Serie de tiempo – {var} (drift 区块高亮)")
    plt.xlabel("Tiempo"); plt.ylabel(var); plt.tight_layout(); plt.show()
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
def periocidad_data(df, columna: str, dia: int = None, mes: int = None):

    d = df.copy()
    d = d.dropna(subset=["date_time"])
    d = d[~d[columna].isna()].copy()

    if mes is not None:
        d = d[d["date_time"].dt.month == mes]
    if dia is not None:
        d = d[d["date_time"].dt.day == dia]
    
    diff = d["date_time"].diff()
    diff_v = d[columna].diff()

    prom = diff.mean()
    std = diff.std()
    minimo = diff.min()
    maximo = diff.max()

    prom_v = float(diff_v.mean())
    std_v = float(diff_v.std())

    diccionario = {"n_intervalos": len(diff),
                   "promedio": prom,
                   "promedio_minutos": prom.total_seconds() / 60,
                   "std_minutos": std.total_seconds() / 60, 
                   "minimo": minimo.total_seconds() / 60, 
                   "maximo": maximo.total_seconds() / 60,
                   "diff": diff,
                   "promedio valor": prom_v,
                   "std_valor": std_v}
    
    return diccionario


# ============================================================================
# Análisis de diferencias centradas (diff)
# ============================================================================

def calcular_diff_centrada(serie: pd.Series) -> pd.Series:
    """
    Calcula la diferencia centrada mínima para cada punto.
    
    Para cada valor, compara con el anterior y el siguiente,
    tomando la menor de las dos diferencias absolutas.
    
    Args:
        serie: Serie temporal con índice datetime
        
    Returns:
        Serie con las diferencias centradas
    """
    val_prev = serie.shift(1).fillna(serie.shift(2))
    val_sig = serie.shift(-1).fillna(serie.shift(-2))
    
    diff_prev = abs(val_prev - serie)
    diff_sig = abs(val_sig - serie)
    
    diff_centrada = pd.Series(
        data=[min(p, s) if pd.notna(p) and pd.notna(s) else float('nan') 
              for p, s in zip(diff_prev, diff_sig)],
        index=serie.index
    )
    
    return diff_centrada


def analizar_diferencias(df: pd.DataFrame, columnas: list = None,
                        fecha_inicio: str = None, fecha_fin: str = None,
                        null_threshold: float = 0.5,
                        std_multiplier: float = 12.0,
                        diff_absolute_threshold: float = 4.0) -> dict:
    """
    Detecta valores atípicos usando análisis de diferencias centradas.
    
    Algoritmo:
    1. Para cada punto, calcula la diferencia mínima con vecinos (diff_centrada)
    2. Agrupa por día y compara std diaria vs std global
    3. Marca como atípico si:
       - diff_centrada >= std_usado * std_multiplier, o
       - diff simple >= diff_absolute_threshold
    
    Args:
        df: DataFrame con columna 'date_time' y variables numéricas
        columnas: Lista de columnas a analizar (None = todas las numéricas)
        fecha_inicio: Fecha inicial del análisis (formato 'YYYY-MM-DD')
        fecha_fin: Fecha final del análisis (formato 'YYYY-MM-DD')
        null_threshold: Umbral de datos faltantes por día (0.5 = 50%)
        std_multiplier: Multiplicador de std para detección (default 12)
        diff_absolute_threshold: Umbral absoluto de diferencia simple (default 4)
        
    Returns:
        Diccionario con resultados por columna:
        {
            'columna_1': {
                'resultados': DataFrame con ['Datos', 'Etiqueta', 'Diff_centrada'],
                'n_outliers': int,
                'n_normales': int,
                'n_insuficientes': int
            },
            ...
        }
    """
    import numpy as np
    
    # Validar y preparar datos
    if 'date_time' not in df.columns:
        raise ValueError("El DataFrame debe tener columna 'date_time'")
    
    df_work = df.copy()
    df_work['date_time'] = pd.to_datetime(df_work['date_time'])
    
    # Seleccionar columnas
    if columnas is None:
        columnas = df_work.select_dtypes(include=['float64', 'int64']).columns
        columnas = [col for col in columnas if col != 'date_time']
    
    resultados = {}
    
    for columna in columnas:
        # Crear serie temporal
        serie = df_work.set_index('date_time')[columna]
        
        # Aplicar filtro de fechas si se especifica
        if fecha_inicio and fecha_fin:
            serie = serie[fecha_inicio:fecha_fin]
            rango_dias = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        elif fecha_inicio:
            serie = serie[fecha_inicio:]
            rango_dias = pd.date_range(start=fecha_inicio, end=serie.index[-1], freq='D')
        elif fecha_fin:
            serie = serie[:fecha_fin]
            rango_dias = pd.date_range(start=serie.index[0], end=fecha_fin, freq='D')
        else:
            rango_dias = pd.date_range(start=serie.index[0], end=serie.index[-1], freq='D')
        
        # Calcular diferencias
        diff_centrada = calcular_diff_centrada(serie)
        diff_simple = serie.diff()
        std_global = diff_centrada.std()
        
        # Si std_global es 0 o NaN, no hay variación real -> todos son normales
        if pd.isna(std_global) or std_global == 0:
            labels = ["Normal" if pd.notna(v) else None for v in serie]
            
            resultados_df = pd.DataFrame({
                'Datos': serie,
                'Etiqueta': labels,
                'Diff_centrada': diff_centrada
            })
            
            n_outliers = 0
            n_normales = sum(1 for x in labels if x == "Normal")
            n_insuficientes = 0
            
            resultados[columna] = {
                'resultados': resultados_df,
                'n_outliers': n_outliers,
                'n_normales': n_normales,
                'n_insuficientes': n_insuficientes,
                'std_global': std_global if pd.notna(std_global) else 0.0,
                'parametros': {
                    'null_threshold': null_threshold,
                    'std_multiplier': std_multiplier,
                    'diff_absolute_threshold': diff_absolute_threshold
                }
            }
            continue
        
        # Vectorizar etiquetado en lugar de iterar por día (mucho más rápido)
        labels = []
        
        # Agregar columna de fecha para agrupar
        serie_con_fecha = serie.to_frame(name='valor')
        serie_con_fecha['fecha'] = serie_con_fecha.index.date
        serie_con_fecha['diff_centrada'] = diff_centrada
        serie_con_fecha['diff_simple'] = diff_simple
        
        # Procesar por día
        for fecha, grupo in serie_con_fecha.groupby('fecha'):
            n = len(grupo)
            
            # Si hay demasiados nulos, marcar todo el día como insuficiente
            if grupo['valor'].isnull().sum() >= n * null_threshold:
                labels.extend(["No hay data suf"] * n)
                continue
            
            # Calcular std del día
            std_dia = grupo['diff_centrada'].std()
            std_usado = std_dia if std_dia > std_global else std_global
            
            # Etiquetar vectorialmente
            es_nan = grupo['diff_centrada'].isna()
            es_outlier_diff = grupo['diff_centrada'] >= std_usado * std_multiplier
            es_outlier_simple = grupo['diff_simple'].abs() >= diff_absolute_threshold
            es_outlier = es_outlier_diff | es_outlier_simple
            
            etiquetas_dia = []
            for i in range(n):
                if es_nan.iloc[i]:
                    etiquetas_dia.append(None)
                elif es_outlier.iloc[i]:
                    etiquetas_dia.append("Outlier")
                else:
                    etiquetas_dia.append("Normal")
            
            labels.extend(etiquetas_dia)
        
        # Crear DataFrame de resultados
        resultados_df = pd.DataFrame({
            'Datos': serie,
            'Etiqueta': labels,
            'Diff_centrada': diff_centrada
        })
        
        # Calcular estadísticas
        n_outliers = sum(1 for x in labels if x == "Outlier")
        n_normales = sum(1 for x in labels if x == "Normal")
        n_insuficientes = sum(1 for x in labels if x == "No hay data suf")
        
        resultados[columna] = {
            'resultados': resultados_df,
            'n_outliers': n_outliers,
            'n_normales': n_normales,
            'n_insuficientes': n_insuficientes,
            'std_global': std_global,
            'parametros': {
                'null_threshold': null_threshold,
                'std_multiplier': std_multiplier,
                'diff_absolute_threshold': diff_absolute_threshold
            }
        }
    
    return resultados