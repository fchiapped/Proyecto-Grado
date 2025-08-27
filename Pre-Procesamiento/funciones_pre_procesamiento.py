import matplotlib.pyplot as plt  
import seaborn as sns 
import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
# Plots Análisis Exploratorio
def plot_temporal(df, columna: str, color: str='blue', marker: str='o', ax=None,
                  warning_min=None, warning_max=None, critical_min=None, critical_max=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    d = df.copy()
    d['date_time'] = pd.to_datetime(d['date_time'], errors='coerce')
    ax.plot(d['date_time'], d[columna], color=color, marker=marker, markersize=3, linewidth=1)
    # Líneas de límites si existen
    if warning_min is not None:
        ax.axhline(warning_min, color='orange', linestyle='--', label='Warning Min')
    if warning_max is not None:
        ax.axhline(warning_max, color='orange', linestyle='--', label='Warning Max')
    if critical_min is not None:
        ax.axhline(critical_min, color='red', linestyle=':', label='Critical Min')
    if critical_max is not None:
        ax.axhline(critical_max, color='red', linestyle=':', label='Critical Max')
    ax.set_title(f'{columna} vs fecha')
    ax.set_xlabel('fecha')
    ax.set_ylabel(columna)
    ax.grid(True)
    # Mostrar leyenda solo si hay límites
    if any(x is not None for x in [warning_min, warning_max, critical_min, critical_max]):
        ax.legend()
    if ax is None:  # retrocompat
        plt.tight_layout(); plt.show()

def plot_avg_hora(df, columna: str, color: str='skyblue', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    d = df.copy()
    d['date_time'] = pd.to_datetime(d['date_time'], errors='coerce')
    d['hour'] = d['date_time'].dt.hour
    avg_by_hour = d.groupby('hour', as_index=False)[columna].mean()
    sns.barplot(x='hour', y=columna, data=avg_by_hour, color=color, ax=ax)
    ax.set_title(f'Promedio de {columna} por Hora')
    ax.set_xlabel('Hora del Día')
    ax.set_ylabel(f'Promedio {columna}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    if ax is None:
        plt.tight_layout(); plt.show()

def plot_densidad(df, columna: str, color: str='purple', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df[columna], color=color, fill=True, linewidth=2, ax=ax)
    ax.set_title(f'Densidad de {columna}')
    ax.set_xlabel(columna)
    ax.set_ylabel('Densidad')
    ax.grid(True, linestyle='--', alpha=0.7)
    if ax is None:
        plt.tight_layout(); plt.show()

def heatmap_hour(df, columna: str, agg: str='mean', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    d = df.copy()
    d['date_time'] = pd.to_datetime(d['date_time'], errors='coerce')
    d = d.dropna(subset=['date_time'])
    d['hour'] = d['date_time'].dt.hour
    d['dow_num'] = d['date_time'].dt.dayofweek
    dow_labels = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
    pivot = d.pivot_table(index='dow_num', columns='hour', values=columna, aggfunc=agg)
    sns.heatmap(pivot, annot=False, cbar=True, ax=ax)
    ax.set_title(f'{agg.capitalize()} de {columna} por Hora y Día de la semana')
    ax.set_xlabel('Hora del día')
    ax.set_ylabel('Día de la semana')
    ax.set_yticks([i + 0.5 for i in range(7)])
    ax.set_yticklabels(dow_labels, rotation=0)
    if ax is None:
        plt.tight_layout(); plt.show()

def plot_corr(df):
    corr = df.select_dtypes(include='number').corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

    plt.title('Mapa de Correlación entre Variables Numéricas')
    plt.show()

def plot_all(df, columna: str,
             color_temporal='blue', marker_temporal='o',
             color_avg='skyblue', color_densidad='purple',
             agg_heatmap='mean',
             warning_min=None, warning_max=None, critical_min=None, critical_max=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    plot_temporal(df, columna, color_temporal, marker_temporal, ax=axes[0, 0],
                  warning_min=warning_min, warning_max=warning_max,
                  critical_min=critical_min, critical_max=critical_max)

    plot_avg_hora(df, columna, color_avg, ax=axes[0, 1])

    plot_densidad(df, columna, color_densidad, ax=axes[1, 0])

    heatmap_hour(df, columna, agg_heatmap, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

def plot_all_timeseries(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if 'date_time' not in df.columns:
        raise ValueError("El DataFrame debe tener una columna 'date_time'.")
    fig, axs = plt.subplots(len(num_cols), 1, figsize=(12, 3*len(num_cols)), sharex=True)
    if len(num_cols) == 1:
        axs = [axs]
    for i, col in enumerate(num_cols):
        axs[i].plot(df['date_time'], df[col], label=col)
        axs[i].set_ylabel(col)
        axs[i].legend(loc='upper right')
    plt.xlabel('date_time')
    plt.show()
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
# Fechas sin Datos
def fechas_con_y_sin_datos(df, dt_col="date_time", min_rows=60):
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

def analizar_columnas_por_fecha(df, columnas, dt_col="date_time", min_rows=60):
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
# Outliers
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

#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
# Ventanas Activas (Planta 1)
def build_active_flag(
    df: pd.DataFrame,
    fecha_col: str,
    rules: dict,
    smooth_window: int = 15,       # ~15 min si tienes dato por minuto
    min_block_len: int = 10        # descarta bloques < 10 muestras
) -> pd.Series:
    """
    rules: diccionario {nombre_variable: {"op": ">", "thr": valor}} o {"op": "diff>", "thr": valor}
           Soporta:
             - ">"  : variable > thr
             - "<"  : variable < thr
             - "diff>" : |diff(variable)| > thr  (movimiento)
    """
    df2 = df.copy()
    df2[fecha_col] = pd.to_datetime(df2[fecha_col])
    df2 = df2.sort_values(fecha_col)

    conds = []
    for col, spec in rules.items():
        op = spec.get("op", ">")
        thr = spec.get("thr", 0)
        if col not in df2.columns:
            continue
        s = df2[col]
        if op == ">":
            c = s > thr
        elif op == "<":
            c = s < thr
        elif op == "diff>":
            c = s.diff().abs() > thr
        else:
            raise ValueError(f"Operador no soportado: {op}")
        conds.append(c.fillna(False))

    if not conds:
        return pd.Series(False, index=df2.index, name="proceso_activo")

    raw = np.logical_or.reduce(conds)

    # Suavizado para evitar parpadeos (cierre morfológico sencillo)
    smooth = pd.Series(raw, index=df2.index).rolling(smooth_window, min_periods=1).max().astype(bool)

    # Enforce min_block_len: elimina bloques muy cortos (ruido)
    flag = smooth.copy()
    run_start = None
    for i, v in enumerate(flag.values):
        if v and run_start is None:
            run_start = i
        if (not v or i == len(flag)-1) and run_start is not None:
            end = i if not v else i  # cierre
            length = end - run_start + (1 if v and i == len(flag)-1 else 0)
            if length < min_block_len:
                flag.iloc[run_start:end] = False
            run_start = None

    flag.name = "proceso_activo"
    return flag

def blocks_from_flag(df: pd.DataFrame, fecha_col: str, flag: pd.Series) -> pd.DataFrame:
    df2 = df.copy()
    df2[fecha_col] = pd.to_datetime(df2[fecha_col])
    df2 = df2.sort_values(fecha_col)
    f = flag.reindex(df2.index).fillna(False).values

    starts, ends = [], []
    in_block = False
    for i, v in enumerate(f):
        if v and not in_block:
            starts.append(df2[fecha_col].iloc[i])
            in_block = True
        if not v and in_block:
            ends.append(df2[fecha_col].iloc[i-1])
            in_block = False
    if in_block:
        ends.append(df2[fecha_col].iloc[-1])

    return pd.DataFrame({"start": starts, "end": ends})

def plot_with_active_blocks(
    df: pd.DataFrame,
    fecha_col: str,
    col: str,
    blocks: pd.DataFrame,
    resample: str = "15min",
    color: str = "tab:red",
    alpha: float = 0.18,
    show_points: bool = False
):
    df2 = df[[fecha_col, col]].dropna().copy()
    df2[fecha_col] = pd.to_datetime(df2[fecha_col])
    df2 = df2.sort_values(fecha_col)

    if resample is not None:
        s = (df2.set_index(fecha_col)[col]
                .resample(resample).median().dropna().reset_index())
    else:
        s = df2.rename(columns={col: "value"}).rename(columns={"value": col})

    plt.figure(figsize=(12, 4))
    if show_points:
        plt.plot(s[fecha_col], s[col], marker='.', linestyle='None', markersize=2)
    else:
        plt.plot(s[fecha_col], s[col])

    for _, r in blocks.iterrows():
        plt.axvspan(pd.to_datetime(r["start"]), pd.to_datetime(r["end"]), color=color, alpha=alpha)

    plt.title(f"Serie de tiempo – {col} (ventanas ACTIVAS sombreadas)")
    plt.xlabel("Tiempo"); plt.ylabel(col); plt.tight_layout(); plt.show()
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#