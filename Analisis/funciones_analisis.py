import matplotlib.pyplot as plt  
import seaborn as sns 
import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
# Limites pre-definidos
def get_limits(desc_df, variable):
    row = desc_df[desc_df['name'] == variable]
    if row.empty:
        return None, None, None, None
    def safe_get(col):
        v = row.iloc[0][col] if col in row else None
        return v if pd.notna(v) else None
    warning_min = safe_get('warning_min_value')
    warning_max = safe_get('warning_max_value')
    critical_min = safe_get('critical_min_value')
    critical_max = safe_get('critical_max_value')
    return warning_min, warning_max, critical_min, critical_max

def evaluar_estados(df_datos, df_desc):

    df_largo = df_datos.melt(
        id_vars=['date_time'],
        var_name='name',
        value_name='valor'
    )
    
    df_merged = df_largo.merge(df_desc, on='name', how='left')

    def evaluar_estado(row):
        v = row['valor']
        cmax = row['critical_max_value']
        wmax = row['warning_max_value']
        wmin = row['warning_min_value']
        cmin = row['critical_min_value']

        if pd.isna(v):
            return "sin_dato"
        if not pd.isna(cmax) and v > cmax:
            return "critico_max"
        if not pd.isna(cmin) and v < cmin:
            return "critico_min"
        if not pd.isna(wmax) and v > wmax:
            return "advertencia_max"
        if not pd.isna(wmin) and v < wmin:
            return "advertencia_min"
        return "ok"

    df_merged['estado'] = df_merged.apply(evaluar_estado, axis=1)

    resumen = df_merged.groupby(['name','estado']).size().unstack(fill_value=0)
    resumen['total'] = resumen.sum(axis=1)
    for col in resumen.columns:
        if col != 'total':
            resumen[f'%_{col}'] = (resumen[col] / resumen['total'] * 100).round(2)

    return df_merged, resumen

def filtrar_por_estado(df_merged, variable=None, estados=None):
    df_filtrado = df_merged.copy()
    
    if variable is not None:
        if isinstance(variable, str):
            variable = [variable]
        df_filtrado = df_filtrado[df_filtrado['name'].isin(variable)]
    
    if estados is not None:
        if isinstance(estados, str):
            estados = [estados]
        df_filtrado = df_filtrado[df_filtrado['estado'].isin(estados)]
    
    return df_filtrado.reset_index(drop=True)

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

def plot_hist_intervalos(df, columna: str, dia: int=None, mes: int=None):
    # Calcula periodicidad con tu función
    stats = periocidad_data(df, columna, dia=dia, mes=mes)
    diff = stats["diff"].dropna().dt.total_seconds() / 60  # en minutos
    
    prom = stats["promedio_minutos"]
    std = stats["std_minutos"]

    # Gráfico
    plt.figure(figsize=(10,6))
    sns.histplot(diff, kde=True, bins=50, color="skyblue")

    plt.title(f"Distribución de intervalos (min) para {columna}", fontsize=16)
    plt.xlabel("Intervalo (min)", fontsize=14)
    plt.ylabel("Frecuencia", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)

    # Leyenda con valores de promedio y std
    plt.legend([f"Promedio = {prom:.2f} min,  Std = {std:.2f} min"])

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
#--------------------------------------------------------------------------------------------------------#
# Drift y tendencia
def detectar_drift_ks(df, columnas=None, fecha_col="date_time", min_dias=60):
    """
    Aplica test de Kolmogorov-Smirnov para detección de drift en sensores.
    
    Args:
        df: DataFrame con datos de sensores.
        columnas: lista de columnas a analizar (si None, toma todas numéricas).
        fecha_col: nombre de la columna de fechas.
        min_dias: mínimo de días con datos para considerar la variable.
        
    Returns:
        DataFrame con resultados de KS test.
    """
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df["fecha"] = df[fecha_col].dt.date

    # Si no se especifican columnas, usar todas las numéricas
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns.tolist()

    resultados = []

    for col in columnas:
        # Fechas con datos
        fechas_con_datos = df.loc[df[col].notna(), "fecha"].nunique()
        if fechas_con_datos < min_dias:
            resultados.append({
                "variable": col,
                "stat": None,
                "pvalue": None,
                "drift_detectado": None,
                "detalle": "pocos datos"
            })
            continue

        # Dividir en referencia (primer bloque) y comparación (último bloque)
        valores = df[[fecha_col, col]].dropna().sort_values(fecha_col)
        mitad = int(len(valores) / 2)
        ref = valores[col].iloc[:mitad]
        nuevo = valores[col].iloc[mitad:]

        # KS test
        stat, pvalue = ks_2samp(ref, nuevo)
        drift = pvalue < 0.05  # nivel de significancia

        resultados.append({
            "variable": col,
            "stat": stat,
            "pvalue": pvalue,
            "drift_detectado": drift,
            "detalle": "ok"
        })

    return pd.DataFrame(resultados)



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

    # 1) Puntos "normales"
    mask_ok = (~mask_ph_out) & (~mask_rob_out)
    ax.scatter(d.loc[mask_ok, 'date_time'], s.loc[mask_ok],
               color=color, marker=marker, s=20, label=columna)

    # 2) Outliers pH (rojo)
    if mask_ph_out.any():
        ax.scatter(d.loc[mask_ph_out, 'date_time'], s.loc[mask_ph_out],
                   color='red', marker=marker, s=24, label='Outliers pH')

    # 3) Outliers robustos (naranjo), excluyendo los ya rojos
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
    """
    Z-score method: 标准差法，返回True为异常点。
    """
    mu = series.mean()
    sigma = series.std()
    z = (series - mu) / sigma
    return z.abs() > threshold

def outliers_iqr(series, k=1.5):
    """
    IQR method: 四分位距法，返回True为异常点。
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return (series < lower) | (series > upper)

def outliers_rolling(series, window=30, k=3):
    """
    Rolling window method: 滚动窗口法，返回True为异常点。
    window: 窗口大小（样本点数）
    k: 超过均值 ± k*std 判定为异常
    """
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
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#

