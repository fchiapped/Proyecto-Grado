
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