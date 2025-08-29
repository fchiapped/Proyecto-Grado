
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
#--------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------#
# Drift y tendencia

def detectar_drift_ks(
    df, columnas=None, fecha_col="date_time",

    # Ventanas Activas
    flag_col=None,           # ej. "planta1_activa" o None para no filtrar
    flag_value=None,         # True -> solo activos, False -> solo inactivos, None -> no filtra
    estado_label=None, 

    # ventanas
    window_days=14, step_days=3, min_dias=10, min_points=5000,
    compare="adjacent",             # "adjacent" ó "baseline"
    skip_first_days=0,              # ignora los primeros N días (calentamiento)
    # sensibilidad
    alpha=0.005, ks_min=0.15,       # p-valor y tamaño de efecto KS
    fdr=True,                       # Benjamini–Hochberg por variable
    min_consecutive=3,              # persistencia (N ventanas seguidas)
    # preproc
    winsor=None                     # (q_low, q_high) ej. (0.01, 0.99) o None
):

    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(fecha_col)

    # --- NUEVO: filtrar por flag si se pide ---
    if flag_col is not None and flag_value is not None:
        # aceptamos bool o 0/1; convertimos a bool para evitar sorpresas
        df[flag_col] = df[flag_col].astype(bool)
        df = df.loc[df[flag_col] == bool(flag_value)].copy()

    # columnas numéricas por defecto (excluye flag_col si quedó en el DF)
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns.tolist()
        if flag_col in columnas:
            columnas.remove(flag_col)

    # winsorize opcional
    if winsor is not None and len(df):
        ql, qh = winsor
        for c in columnas:
            q1, q2 = df[c].quantile(ql), df[c].quantile(qh)
            if pd.notna(q1) and pd.notna(q2):
                df[c] = df[c].clip(q1, q2)

    # rango temporal (con burn-in opcional)
    tmin = df[fecha_col].min()
    tmax = df[fecha_col].max()
    if pd.isna(tmin) or pd.isna(tmax):
        return pd.DataFrame(columns=[
            "variable","window_start","window_end","ref_start","ref_end",
            "n_ref","n_new","stat","pvalue","drift_detectado","detalle","estado"
        ])

    tmin = tmin.normalize()
    if skip_first_days and skip_first_days > 0:
        tmin = tmin + pd.Timedelta(days=skip_first_days)
    tmax = tmax.normalize()

    # generar ventanas
    wins = []
    step = pd.Timedelta(days=step_days)
    span = pd.Timedelta(days=window_days) - pd.Timedelta(seconds=1)
    t = tmin
    while t <= tmax:
        wins.append((t, min(t + span, tmax + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))))
        t = t + step

    resultados = []
    for col in columnas:
        # cobertura & conteo por ventana
        cov_dias, npts, series = [], [], []
        for (ws, we) in wins:
            s = df.loc[(df[fecha_col] >= ws) & (df[fecha_col] <= we) & (df[col].notna()), [fecha_col, col]]
            cov_dias.append(s[fecha_col].dt.date.nunique())
            npts.append(len(s))
            series.append(s[col].values)

        # ventanas válidas
        valid = [i for i in range(len(wins)) if cov_dias[i] >= min_dias and npts[i] >= min_points]
        if len(valid) < 2:
            for i, (ws, we) in enumerate(wins):
                resultados.append(dict(
                    variable=col, window_start=ws, window_end=we,
                    ref_start=None, ref_end=None,
                    n_ref=None, n_new=npts[i],
                    stat=np.nan, pvalue=np.nan,
                    drift_detectado=None, detalle="pocos datos"
                ))
            continue

        base_i = valid[0]
        for k, i in enumerate(valid[1:]):
            cur_i = i
            ref_i = valid[k] if compare == "adjacent" else base_i
            ws_ref, we_ref = wins[ref_i]
            ws_cur, we_cur = wins[cur_i]
            x, y = series[ref_i], series[cur_i]
            stat, p = ks_2samp(x, y)
            resultados.append(dict(
                variable=col,
                window_start=ws_cur, window_end=we_cur,
                ref_start=ws_ref, ref_end=we_ref,
                n_ref=len(x), n_new=len(y),
                stat=float(stat), pvalue=float(p),
                drift_detectado=(p < alpha and stat >= ks_min),
                detalle="ok"
            ))

    res = pd.DataFrame(resultados).sort_values(["variable", "window_start"]).reset_index(drop=True)

    # --- FDR por variable (opcional) ---
    if fdr and not res.empty:
        for v in res["variable"].dropna().unique():
            mask = (res["variable"] == v) & (res["detalle"] == "ok")
            p = res.loc[mask, "pvalue"].values.astype(float)
            if p.size == 0:
                continue
            order = np.argsort(p)
            ranked = np.arange(1, len(p)+1)
            thr = alpha * ranked / len(p)
            passed = np.zeros_like(p, dtype=bool)
            ok = np.where(p[order] <= thr)[0]
            if ok.size:
                passed[order[:ok.max()+1]] = True
            res.loc[mask, "drift_detectado"] = res.loc[mask, "drift_detectado"].values & passed

    # --- Persistencia: ≥ N consecutivas ---
    if min_consecutive and min_consecutive > 1 and not res.empty:
        for v in res["variable"].dropna().unique():
            m = (res["variable"] == v) & (res["detalle"] == "ok")
            flags = res.loc[m, "drift_detectado"].fillna(False).values.astype(bool)
            run = 0
            hard = np.zeros_like(flags, dtype=bool)
            for i, f in enumerate(flags):
                run = run + 1 if f else 0
                hard[i] = run >= min_consecutive
            # backfill dentro de cada racha
            i = len(hard) - 1
            while i >= 0:
                if hard[i]:
                    j = i
                    while j >= 0 and flags[j]:
                        hard[j] = True; j -= 1
                    i = j
                i -= 1
            res.loc[m, "drift_detectado"] = hard

    # --- NUEVO: etiqueta de estado en la salida ---
    if flag_col is not None and flag_value is not None:
        lbl = estado_label
        if lbl is None:
            lbl = "activa" if bool(flag_value) else "inactiva"
        res["estado"] = lbl
    else:
        res["estado"] = "general"

    return res

def _drift_blocks(res, var):
    sub = res[(res["variable"] == var) & (res["detalle"] == "ok")].copy()
    if sub.empty: 
        return []
    sub = sub.sort_values("window_start")
    blocks, open_s, open_e = [], None, None
    for _, r in sub.iterrows():
        if bool(r.get("drift_detectado", False)):
            s, e = pd.to_datetime(r["window_start"]), pd.to_datetime(r["window_end"])
            if open_s is None:
                open_s, open_e = s, e
            else:
                if s <= open_e + pd.Timedelta(seconds=1):
                    open_e = max(open_e, e)
                else:
                    blocks.append((open_s, open_e))
                    open_s, open_e = s, e
        else:
            if open_s is not None:
                blocks.append((open_s, open_e)); open_s, open_e = None, None
    if open_s is not None:
        blocks.append((open_s, open_e))
    return blocks

# --- una figura para una variable
def plot_ks_one(df, res, fecha_col, var, resample="15min",
                shade_color="tab:red", shade_alpha=0.22,
                line_kwargs=None):
    d = df[[fecha_col, var]].dropna().sort_values(fecha_col)
    s = d.set_index(fecha_col)[var].resample(resample).median().dropna().reset_index()
    blocks = _drift_blocks(res, var)

    plt.figure(figsize=(12, 4))
    plt.plot(s[fecha_col], s[var], **(line_kwargs or {}))
    for s0, s1 in blocks:
        plt.axvspan(s0, s1, color=shade_color, alpha=shade_alpha)
    n_blocks = len(blocks)
    plt.title(f"Serie – {var}  (bloques con drift: {n_blocks})")
    plt.xlabel("Tiempo"); plt.ylabel(var); plt.tight_layout(); plt.show()

# --- galería: muchas variables en grilla
def plot_ks_gallery(df, res, fecha_col, variables=None, resample="15min",
                    ncols=2, height_per_row=2.6, only_with_drift=True):
    if variables is None:
        variables = res["variable"].dropna().unique().tolist()
    if only_with_drift:
        have = res[(res["detalle"]=="ok") & (res["drift_detectado"]==True)]["variable"].unique().tolist()
        variables = [v for v in variables if v in have]

    n = len(variables)
    if n == 0:
        print("No hay variables con drift para mostrar."); 
        return

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, height_per_row*nrows), squeeze=False)
    axes = axes.flatten()

    # ordenar por # de ventanas con drift (desc)
    counts = (res[(res["detalle"]=="ok") & (res["drift_detectado"]==True)]
              .groupby("variable", as_index=False).size()
              .sort_values("size", ascending=False))
    order = counts["variable"].tolist()
    variables = sorted(variables, key=lambda v: order.index(v) if v in order else 1e9)

    for i, var in enumerate(variables):
        ax = axes[i]
        d = df[[fecha_col, var]].dropna().sort_values(fecha_col)
        s = d.set_index(fecha_col)[var].resample(resample).median().dropna().reset_index()
        ax.plot(s[fecha_col], s[var])
        for s0, s1 in _drift_blocks(res, var):
            ax.axvspan(pd.to_datetime(s0), pd.to_datetime(s1), color="tab:red", alpha=0.22)
        n_blocks = len(_drift_blocks(res, var))
        ax.set_title(f"{var}  (bloques: {n_blocks})", fontsize=10)
        ax.set_xlabel("Tiempo"); ax.set_ylabel(var)
    # apaga ejes sobrantes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout(); plt.show()

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