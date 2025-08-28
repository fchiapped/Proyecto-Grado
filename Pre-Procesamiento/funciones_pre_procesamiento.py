import matplotlib.pyplot as plt  
import seaborn as sns 
import pandas as pd
import numpy as np

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
             warning_min=None, warning_max=None, critical_min=None, critical_max=None,
             desc_df=None):   # <-- NUEVO parámetro opcional
    # --- NUEVO: si no me diste límites, los intento sacar de desc_df ---
    if desc_df is not None:
        row = desc_df[desc_df['name'] == columna]
        if not row.empty:
            def sg(col):
                v = row.iloc[0][col] if col in row else None
                return v if pd.notna(v) else None
            if warning_min  is None: warning_min  = sg('warning_min_value')
            if warning_max  is None: warning_max  = sg('warning_max_value')
            if critical_min is None: critical_min = sg('critical_min_value')
            if critical_max is None: critical_max = sg('critical_max_value')
    # -------------------------------------------------------------------

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
# Ventanas Activas (Planta 1)
def _bool_closing(x_bool: pd.Series, close_window: int) -> pd.Series:
    if close_window <= 1:
        return x_bool
    dil = x_bool.rolling(close_window, center=True, min_periods=1).max().astype(bool)
    ero = dil.rolling(close_window, center=True, min_periods=1).min().astype(bool)
    return ero

def _blocks_from_flag_indexed(active_flag: pd.Series) -> pd.DataFrame:
    af = active_flag.fillna(False).astype(bool)
    edge = af.astype(int).diff().fillna(af.astype(int))
    starts = af.index[edge == 1]
    ends   = af.index[edge == -1]

    if len(starts) and len(ends):
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            ends = ends.append(pd.Index([af.index[-1]]))
    elif len(starts) and not len(ends):
        ends = pd.Index([af.index[-1]])
    elif len(ends) and not len(starts):
        starts = pd.Index([af.index[0]])

    return pd.DataFrame({'inicio': starts, 'fin': ends}).reset_index(drop=True)
    
def merge_blocks(blocks: pd.DataFrame, max_gap_minutes: int = 20) -> pd.DataFrame:
    if blocks.empty:
        return blocks
    blocks = blocks.sort_values('inicio').reset_index(drop=True)
    merged = []
    cur_start = blocks.loc[0, 'inicio']
    cur_end   = blocks.loc[0, 'fin']
    max_gap = pd.to_timedelta(max_gap_minutes, unit='m')
    for i in range(1, len(blocks)):
        s, e = blocks.loc[i, 'inicio'], blocks.loc[i, 'fin']
        if (s - cur_end) <= max_gap:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return pd.DataFrame(merged, columns=['inicio', 'fin'])

def rasterize_blocks(index: pd.DatetimeIndex, blocks: pd.DataFrame) -> pd.Series:
    out = pd.Series(False, index=index)
    for _, row in blocks.iterrows():
        out.loc[(index >= row['inicio']) & (index <= row['fin'])] = True
    return out

def _evaluate_rule(series: pd.Series, rule: dict, smooth_window: int) -> pd.Series:
    s = series.copy()
    s = s.rolling(smooth_window, center=True, min_periods=1).median()

    op = rule.get("op")
    thr = rule.get("thr")

    if op == ">":
        return (s >= thr)
    elif op == "diff>":
        # actividad por "movimiento"
        diff = s.diff().abs().rolling(3, min_periods=1).mean()
        return (diff >= thr)
    else:
        raise ValueError(f"Operación de regla no soportada: {op}")

def build_active_flag(
    df: pd.DataFrame,
    fecha_col: str,
    rules: dict,
    smooth_window: int = 15,
    min_block_len: int = 10,
    close_window: int = 7,
    max_gap_minutes: int = 25,
    edge_buffer: int = 0
) -> pd.Series:
    """
    Construye flag OR entre reglas sobre múltiples señales, estabiliza con closing,
    filtra por duración mínima y fusiona bloques separados por gaps cortos.
    """
    d = df.copy()
    d[fecha_col] = pd.to_datetime(d[fecha_col])
    d = d.sort_values(fecha_col).set_index(fecha_col)

    # asegurar frecuencia uniforme (por minuto si aplica)
    freq = pd.infer_freq(d.index) or 'T'
    d = d.asfreq(freq)

    flags = []
    for col, rule in rules.items():
        if col not in d.columns:
            continue
        flags.append(_evaluate_rule(d[col], rule, smooth_window))
    if not flags:
        return pd.Series(False, index=d.index)

    # OR entre todas las reglas
    active = flags[0].copy()
    for f in flags[1:]:
        active = (active | f)

    # closing para tapar microcortes
    active = _bool_closing(active, close_window=close_window)

    # pasar a bloques, aplicar buffer/min_len/merge y rasterizar de vuelta
    blocks = _blocks_from_flag_indexed(active)

    if edge_buffer > 0 and not blocks.empty:
        blocks['inicio'] = blocks['inicio'] - pd.to_timedelta(edge_buffer, unit='m')
        blocks['fin']    = blocks['fin']    + pd.to_timedelta(edge_buffer, unit='m')

    # filtrar por duración mínima
    if not blocks.empty:
        blocks = blocks[(blocks['fin'] - blocks['inicio']).dt.total_seconds() >= (min_block_len*60)]

    # merge por gaps cortos
    blocks = merge_blocks(blocks, max_gap_minutes=max_gap_minutes)

    # rasterizar nuevamente para obtener el flag final estable
    active_final = rasterize_blocks(d.index, blocks)
    return active_final

def blocks_from_flag(df: pd.DataFrame, fecha_col: str, flag: pd.Series) -> pd.DataFrame:
    """
    Versión compatible con tu firma: toma df+fecha para asegurar índice y
    devuelve bloques [inicio, fin].
    """
    d = df.copy()
    d[fecha_col] = pd.to_datetime(d[fecha_col])
    d = d.sort_values(fecha_col).set_index(fecha_col)

    # alinear flag a ese índice
    flag = flag.reindex(d.index)
    return _blocks_from_flag_indexed(flag)

def plot_with_active_blocks(
    df: pd.DataFrame,
    fecha_col: str,
    y_col: str,
    blocks: pd.DataFrame,
    resample: str | None = None,
    title: str | None = None
):
    s = df[[fecha_col, y_col]].dropna().copy()
    s[fecha_col] = pd.to_datetime(s[fecha_col])
    s = s.sort_values(fecha_col).set_index(fecha_col)[y_col]

    if resample:
        s = s.resample(resample).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    s.plot(ax=ax, lw=1)
    for _, row in blocks.iterrows():
        ax.axvspan(row['inicio'], row['fin'], alpha=0.15)
    ax.set_xlabel("")
    ax.set_title(title or f"{y_col} con ventanas activas (sombreado)")
    ax.grid(True, alpha=0.3)
    plt.show()
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#