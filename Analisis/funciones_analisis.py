
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
# Drift y tendencia

import json
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from pathlib import Path
from typing import Optional, Literal

from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift, DriftedColumnsCount
from evidently import DataDefinition, Dataset
#--------------------------------------------------------------------------------------------------------#

def strip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Si existe is_outlier, elimina filas marcadas como outlier (True/1/'true')."""
    if "is_outlier" not in df.columns:
        return df
    s = df["is_outlier"]
    # considerar 1/True/'true'/'True' como outlier
    mask = ~(s.astype(str).str.lower().isin(["1","true","t","yes","y"]))
    return df.loc[mask].drop(columns=["is_outlier"])

def resample_mixed(df: pd.DataFrame, freq: str, agg: str) -> pd.DataFrame:
    """Resample para mixto: numéricas por mean/median, categóricas por moda."""
    if df.empty: return df
    num = df.select_dtypes(include="number")
    other_cols = [c for c in df.columns if c not in num.columns]
    if agg == "median":
        num_rs = num.resample(freq).median()
    else:
        num_rs = num.resample(freq).mean()
    if other_cols:
        # moda por bloque (si hay empate, toma la primera)
        def _mode(s: pd.Series):
            s = s.dropna()
            if s.empty: return np.nan
            counts = s.value_counts()
            return counts.index[0]
        other = df[other_cols]
        other_rs = other.resample(freq).agg(_mode)
        out = pd.concat([num_rs, other_rs], axis=1)
    else:
        out = num_rs
    # reordenar columnas como original
    out = out[[c for c in df.columns if c in out.columns]]
    return out

def build_types_keep_all(ref: pd.DataFrame, cur: pd.DataFrame, dt_col: str) -> tuple[list[str], list[str], list[str]]:
    """
    Devuelve (numeric_cols, categorical_cols, dropped_all_nan)
    Incluye TODAS las columnas comunes salvo:
      - dt_col, EXCLUDE_COLUMNS
      - columnas 100% NaN en ref y cur
    """
    common = [c for c in ref.columns.intersection(cur.columns) if c != dt_col and c not in EXCLUDE_COLUMNS]
    numeric_cols, categorical_cols, dropped_all_nan = [], [], []
    for c in common:
        r, k = ref[c], cur[c]
        if r.dropna().empty and k.dropna().empty:
            dropped_all_nan.append(c); continue
        if is_bool_dtype(r) or is_bool_dtype(k):
            categorical_cols.append(c)
        elif is_numeric_dtype(r) or is_numeric_dtype(k):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols, dropped_all_nan

def window_starts(index: pd.DatetimeIndex, win: pd.Timedelta, step: pd.Timedelta):
    if len(index) == 0: return []
    t, tmax = index.min(), index.max()
    out = []
    while t + win <= tmax:
        out.append(t); t = t + step
    return out

def ref_decay_prefix_mass(df_hist: pd.DataFrame, now: pd.Timestamp,
                          half_life_hours=24*7, target_mass=0.95) -> pd.DataFrame:
    """
    Decay determinístico: calcula pesos w = exp(-Δt/τ), ordena por recencia,
    y toma el prefijo más reciente cuya masa acumulada >= target_mass.
    """
    if df_hist.empty: return df_hist
    tau = pd.Timedelta(hours=half_life_hours) / np.log(2)
    dt = (now - df_hist.index)
    w = np.exp(-dt / tau).astype(float)
    order = np.argsort(-df_hist.index.view("i8"))  # descendente por tiempo
    w_sorted = w.values[order]
    cum = np.cumsum(w_sorted) / w_sorted.sum()
    cut_idx = np.searchsorted(cum, target_mass, side="left")
    # tomar hasta cut_idx (inclusive)
    take_pos = order[: (cut_idx + 1)]
    sel = df_hist.iloc[np.sort(take_pos)]
    return sel

def ref_golden(df_hist: pd.DataFrame, win="30min", step="10min", k=40) -> pd.DataFrame:
    """Elige K ventanas históricas más 'estables' (score robusto)."""
    win_td, step_td = pd.to_timedelta(win), pd.to_timedelta(step)
    starts = window_starts(df_hist.index, win_td, step_td)
    if not starts: return df_hist.iloc[:0]
    rows = []
    for t0 in starts:
        t1 = t0 + win_td - pd.Timedelta(nanoseconds=1)
        sub = df_hist.loc[t0:t1]
        if len(sub) < 3: continue
        num = sub.select_dtypes(include="number")
        if num.shape[1] == 0: continue
        med = num.median()
        iqr = num.quantile(0.75) - num.quantile(0.25)
        rsd = (iqr / (med.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
        score = rsd.median(skipna=True)
        rows.append((t0, t1, float(score)))
    if not rows: return df_hist.iloc[:0]
    stab = pd.DataFrame(rows, columns=["t0","t1","score"]).sort_values("score").head(k)
    parts = [df_hist.loc[t0:t1] for t0, t1, _ in stab.itertuples(index=False)]
    return pd.concat(parts, axis=0) if parts else df_hist.iloc[:0]

def ref_seasonal(df_hist: pd.DataFrame, current_end: pd.Timestamp, weeks_back=12) -> pd.DataFrame:
    """Referencia estacional: misma hora-del-día y día-de-semana, W semanas atrás."""
    if df_hist.empty: return df_hist.iloc[:0]
    slot = current_end.dayofweek * 24 + current_end.hour
    dw, hh = df_hist.index.dayofweek, df_hist.index.hour
    mask = (dw * 24 + hh) == slot
    hist = df_hist.loc[mask].loc[:current_end]
    if hist.empty: return df_hist.iloc[:0]
    start_lim = current_end - pd.Timedelta(weeks=weeks_back)
    return hist.loc[start_lim:]



def extract_value_drift_table(report, snap=None) -> pd.DataFrame:
    """
    Devuelve una tabla por columna con: col, drifted, score, method, threshold.
    Soporta tanto Report como Snapshot según la versión de Evidently.
    """
    # 1) Obtenemos el dict del reporte de la forma que exista
    d = None
    if hasattr(report, "as_dict"):        # Evidently >=0.7 (usualmente en Report)
        d = report.as_dict()
    elif hasattr(report, "json"):         # Algunas versiones exponen .json() en Report
        d = json.loads(report.json())
    elif snap is not None and hasattr(snap, "json"):  # O en Snapshot
        d = json.loads(snap.json())
    else:
        raise RuntimeError("No se pudo serializar el Report/Snapshot a dict (as_dict/json no disponibles).")

    # 2) Parseamos métricas ValueDrift
    rows = []
    for m in d.get("metrics", []):
        if m.get("metric") == "ValueDrift":
            res = m.get("result", {}) or {}
            rows.append({
                "col":       res.get("column_name") or res.get("column"),
                "drifted":   res.get("drift_detected"),
                "score":     res.get("drift_score"),
                "method":    res.get("stattest_name") or res.get("stattest"),
                "threshold": res.get("drift_threshold") or res.get("threshold"),
            })
    return pd.DataFrame(rows)


def make_report_for_plant(
    df: pd.DataFrame,
    strategy: Literal["decay","golden","seasonal"] = BASELINE_STRATEGY,
    forced_dt_col: Optional[str] = "date_time",
    out_prefix: str = "planta",
) -> Path:

    # --- índice temporal y limpieza de outliers (igual que antes)
    dt = "date_time"
    df = df.copy()
    df[dt] = pd.to_datetime(df[dt], errors="coerce")
    df = df.dropna(subset=[dt]).sort_values(dt).set_index(dt)
    df = strip_outliers(df)

    if df.empty:
        raise ValueError("Dataset vacío tras filtrar/parsear fechas.")

    # === agregado: integrar flags si existen ==========================
    flag_path = flag_files.get(out_prefix)
    if flag_path and flag_path.exists():
        flags = pd.read_csv(flag_path, parse_dates=["date_time"])
        flags["date_time"] = pd.to_datetime(flags["date_time"]).dt.floor("min")
        df.index = df.index.floor("min")
        df = df.merge(flags, left_index=True, right_on="date_time", how="left").set_index("date_time")

        # --- filtro por columna: solo se eliminan datos faltantes de ESA variable
        nd_cols = [c for c in df.columns if c.startswith("nd_")]
        for nd_col in nd_cols:
            var = nd_col.replace("nd_", "")
            if var in df.columns:
                mask = ~df[nd_col]  # True donde hay dato válido
                df.loc[~mask, var] = np.nan  # marca como NaN solo esa columna

        # eliminar columnas de control de los flags
        drop_cols = ["valid_for_drift", "nd_any", "nd_all"] + nd_cols
        df = df[[c for c in df.columns if c not in drop_cols]]

        print(f"[{out_prefix}] Flags integrados (por columna) → {len(df)} filas totales, sin eliminar registros completos")
    else:
        print(f"[{out_prefix}] Sin flags o archivo no encontrado, se usa DF completo.")
    # ================================================================

    # --- split temporal (idéntico al tuyo)
    now = df.index.max()
    cur_start = now - pd.to_timedelta(CURRENT_WINDOW)
    cur = df.loc[cur_start:now]
    hist = df.loc[:cur_start - pd.Timedelta(nanoseconds=1)]

    # --- baseline determinístico
    if strategy == "decay":
        ref_global = ref_decay_prefix_mass(hist, now, DECAY_HALF_LIFE_HOURS, DECAY_WEIGHT_MASS)
    elif strategy == "golden":
        ref_global = ref_golden(hist, GOLDEN_WIN, GOLDEN_STEP, GOLDEN_K)
    elif strategy == "seasonal":
        ref_global = ref_seasonal(hist, now, SEASONAL_WEEKS_BACK)
    else:
        raise ValueError("strategy inválida")

    if ref_global.empty:
        ref_global = hist

    # --- columnas comunes menos dt/exclude
    common_cols = sorted(set(ref_global.columns).intersection(cur.columns) - {dt} - set(EXCLUDE_COLUMNS))
    if not common_cols:
        raise ValueError("No hay columnas comunes para comparar.")
    ref_final = ref_global[common_cols].copy()
    cur_final = cur[common_cols].copy()

    # --- RESAMPLE
    if RESAMPLE:
        ref_final = resample_mixed(ref_final, RESAMPLE, RESAMPLE_AGG).dropna(how="all")
        cur_final = resample_mixed(cur_final, RESAMPLE, RESAMPLE_AGG).dropna(how="all")

    # --- tipos y columnas 100% NaN (igual que antes)
    numeric_cols, categorical_cols, dropped_all_nan = build_types_keep_all(ref_final, cur_final, dt_col=dt)

    # --- (resto de make_report_for_plant sin tocar) -------------------
    audit_rows = []
    for c in common_cols:
        reason = []
        if c in EXCLUDE_COLUMNS: reason.append("in_EXCLUDE_COLUMNS")
        if c in dropped_all_nan: reason.append("all_nan_ref_and_cur")
        kept = (c not in dropped_all_nan) and (c not in EXCLUDE_COLUMNS)
        audit_rows.append({"col": c, "kept": kept, "reason": ";".join(reason)})
    audit_df = pd.DataFrame(audit_rows).sort_values(["kept","col"])
    tag_base = f"{strategy}_{'resamp'+RESAMPLE if RESAMPLE else 'raw'}_{pd.Timestamp(now).strftime('%Y%m%d_%H%M%S')}"

    if not numeric_cols and not categorical_cols:
        raise ValueError("Todas las columnas quedaron 100% NaN en ref y cur tras el resample.")

    definition = DataDefinition(
        numerical_columns=numeric_cols if numeric_cols else None,
        categorical_columns=categorical_cols if categorical_cols else None
    )
    preset_kwargs = {}
    if NUM_METHOD != "auto":
        preset_kwargs["num_method"] = NUM_METHOD
        if NUM_THRESHOLD is not None:
            preset_kwargs["num_threshold"] = NUM_THRESHOLD

    metrics = [
        DataDriftPreset(**preset_kwargs),
        DriftedColumnsCount(**preset_kwargs),
        *[
            (ValueDrift(column=c) if NUM_METHOD == "auto"
             else (ValueDrift(column=c, method=NUM_METHOD) if NUM_THRESHOLD is None
                   else ValueDrift(column=c, method=NUM_METHOD, threshold=NUM_THRESHOLD)))
            for c in (numeric_cols + categorical_cols)
        ],
    ]

    report = Report(metrics=metrics)
    ds_ref = Dataset.from_pandas(ref_final.reset_index(drop=True), data_definition=definition)
    ds_cur = Dataset.from_pandas(cur_final.reset_index(drop=True), data_definition=definition)
    snap = report.run(reference_data=ds_ref, current_data=ds_cur)

    # (tu código original de guardado igual)
    df_cols = extract_value_drift_table(report, snap)
    drifted_count = int(df_cols.get("drifted", pd.Series(dtype=bool)).fillna(False).sum())
    total_cols = int(df_cols.shape[0])

    kpi_tag = ""
    kpi_present = KPI_ENABLED and (KPI_COL in df_cols["col"].tolist())
    kpi_drifted = bool(df_cols.query("col == @KPI_COL and drifted == True").shape[0]) if kpi_present else False
    if KPI_ENABLED and KPI_IN_FILENAME:
        kpi_tag = "_KPI-DRIFT" if kpi_drifted else "_KPI-OK" if kpi_present else "_KPI-N/A"

    out_html = output_dir / f"{out_prefix}_{strategy}.html"
    snap.save_html(str(out_html))


    print(f"[{out_prefix}] cols={total_cols} | drifted={drifted_count} | KPI={'DRIFT' if kpi_drifted else 'OK' if kpi_present else 'N/A'}")
    print(f"OK → {out_html.name} (carpeta: {output_dir})")
    return out_html
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



# 合并重叠或相邻的时间区块
def merge_blocks(blocks, gap=pd.Timedelta(seconds=0)):
    """
    合并重叠或相邻的时间区块。
    blocks: list of (start, end) 元组，或DataFrame有'start','end'列
    gap: 允许合并的最大间隔（如0表示仅合并重叠/相邻，1min表示间隔1分钟内也合并）
    返回合并后的区块列表 [(start, end), ...]
    """
    if isinstance(blocks, pd.DataFrame):
        blocks = list(zip(pd.to_datetime(blocks['start']), pd.to_datetime(blocks['end'])))
    elif not blocks:
        return []
    # 排序
    blocks = sorted(blocks, key=lambda x: x[0])
    merged = []
    for b in blocks:
        if not merged:
            merged.append(list(b))
        else:
            last = merged[-1]
            # 如果当前区块与上一区块重叠或间隔小于gap，则合并
            if b[0] <= last[1] + gap:
                last[1] = max(last[1], b[1])
            else:
                merged.append(list(b))
    # 转回元组
    return [tuple(x) for x in merged]

# --- 原始时序数据与drift区块可视化 ---
def plot_raw_with_drift(df, fecha_col, var, blocks, resample=None, color='tab:red', alpha=0.22, show_points=False):
    """
    绘制原始时序数据，并用色块标注drift区块。
    df: DataFrame，包含时间和变量
    fecha_col: 时间列名
    var: 变量名
    blocks: [(start, end), ...] 区块列表
    resample: 例如'15min'，对数据重采样（中位数），None为原始
    color, alpha: 区块色彩与透明度
    show_points: 是否显示原始点
    """
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