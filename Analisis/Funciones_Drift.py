# Funciones_Drift.py — backend limpio para detección de drift
from __future__ import annotations

import warnings
from typing import Optional, Literal, List, Dict, Tuple

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

warnings.filterwarnings("ignore")

# ==========================
# 1. UTILIDADES BASE
# ==========================

def resample_mixed(
    df: pd.DataFrame,
    freq: str,
    agg: Literal["mean", "median"] = "mean",
) -> pd.DataFrame:
    """
    Resamplea numéricas con mean/median y el resto por moda.
    - Se asume que el índice es DatetimeIndex.
    - Si no quieres resample → no llames a esta función.
    """
    if df.empty:
        return df

    num = df.select_dtypes(include="number")
    other_cols = [c for c in df.columns if c not in num.columns]

    if agg == "median":
        num_rs = num.resample(freq).median()
    else:
        num_rs = num.resample(freq).mean()

    if other_cols:
        def _mode(s: pd.Series):
            s = s.dropna()
            if s.empty:
                return np.nan
            return s.value_counts().index[0]

        other = df[other_cols]
        other_rs = other.resample(freq).agg(_mode)
        out = pd.concat([num_rs, other_rs], axis=1)
    else:
        out = num_rs

    # Reordenamos columnas para mantener orden original si existen
    return out[[c for c in df.columns if c in out.columns]]

def build_types_keep_all(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    dt_col: str = "date_time",
    exclude: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Dado ref y cur, detecta columnas numéricas/categóricas comunes.
    - No toca la columna de tiempo (dt_col).
    - Excluye columnas en `exclude`.
    - Devuelve: (numeric_cols, categorical_cols, dropped_all_nan)
    """
    exclude = exclude or []
    common = [
        c for c in ref.columns.intersection(cur.columns)
        if c != dt_col and c not in exclude
    ]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    dropped_all_nan: List[str] = []

    for c in common:
        r, k = ref[c], cur[c]
        if r.dropna().empty and k.dropna().empty:
            dropped_all_nan.append(c)
            continue

        if is_bool_dtype(r) or is_bool_dtype(k):
            categorical_cols.append(c)
        elif is_numeric_dtype(r) or is_numeric_dtype(k):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols, dropped_all_nan

def window_starts(
    index: pd.DatetimeIndex,
    win: pd.Timedelta,
    step: pd.Timedelta,
) -> List[pd.Timestamp]:
    """
    Genera inicios de ventanas deslizantes (para golden).
    """
    if len(index) == 0:
        return []

    t = index.min()
    tmax = index.max()
    out = []
    while t + win <= tmax:
        out.append(t)
        t = t + step
    return out

# ==========================
# 2. ESTRATEGIAS DE REFERENCIA
# ==========================

def ref_decay_prefix_mass(
    df_hist: pd.DataFrame,
    now: pd.Timestamp,
    half_life_hours: float = 24 * 7,
    target_mass: float = 0.95,
) -> pd.DataFrame:
    """
    Referencia "decay":
    - Pondera por exponencial hacia el pasado.
    - Se queda con el prefijo de muestras que acumula ~target_mass del peso.
    """
    if df_hist.empty:
        return df_hist

    tau = pd.Timedelta(hours=half_life_hours) / np.log(2)
    dt = (now - df_hist.index)
    w = np.exp(-dt / tau).astype(float)

    # ordenamos por tiempo descendente (más recientes primero)
    order = np.argsort(-df_hist.index.view("i8"))
    w_sorted = w.values[order]
    cum = np.cumsum(w_sorted) / w_sorted.sum()
    cut_idx = np.searchsorted(cum, target_mass, side="left")
    take_pos = order[: (cut_idx + 1)]
    return df_hist.iloc[np.sort(take_pos)]

def ref_golden(
    df_hist: pd.DataFrame,
    win: str = "30min",
    step: str = "10min",
    k: int = 40,
) -> pd.DataFrame:
    """
    Referencia "golden":
    - Divide el historial en ventanas deslizantes.
    - Calcula "estabilidad relativa" (IQR/|mediana|).
    - Se queda con las k ventanas más estables.
    """
    win_td = pd.to_timedelta(win)
    step_td = pd.to_timedelta(step)
    starts = window_starts(df_hist.index, win_td, step_td)

    if not starts:
        return df_hist.iloc[:0]

    rows = []
    for t0 in starts:
        t1 = t0 + win_td - pd.Timedelta(nanoseconds=1)
        sub = df_hist.loc[t0:t1]
        if len(sub) < 3:
            continue

        num = sub.select_dtypes(include="number")
        if num.shape[1] == 0:
            continue

        med = num.median()
        iqr = num.quantile(0.75) - num.quantile(0.25)
        rsd = (iqr / (med.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
        score = rsd.median(skipna=True)
        rows.append((t0, t1, float(score)))

    if not rows:
        return df_hist.iloc[:0]

    stab = (
        pd.DataFrame(rows, columns=["t0", "t1", "score"])
        .sort_values("score")
        .head(k)
    )
    parts = [df_hist.loc[t0:t1] for t0, t1, _ in stab.itertuples(index=False)]
    return pd.concat(parts, axis=0) if parts else df_hist.iloc[:0]

def ref_seasonal(
    df_hist: pd.DataFrame,
    current_end: pd.Timestamp,
    weeks_back: int = 12,
) -> pd.DataFrame:
    """
    Referencia "seasonal":
    - Se queda con el histórico que cae en el MISMO slot (día_semana, hora)
      que current_end, en las últimas `weeks_back` semanas.
    """
    if df_hist.empty:
        return df_hist.iloc[:0]

    slot = current_end.dayofweek * 24 + current_end.hour
    dw, hh = df_hist.index.dayofweek, df_hist.index.hour
    mask = (dw * 24 + hh) == slot
    hist = df_hist.loc[mask].loc[:current_end]
    if hist.empty:
        return df_hist.iloc[:0]

    start_lim = current_end - pd.Timedelta(weeks=weeks_back)
    return hist.loc[start_lim:]


# ==========================
# 3. MÉTRICAS DE DRIFT
# ==========================

# --- SciPy para KS y Wasserstein (si está instalado) ---
try:
    from scipy.stats import ks_2samp, wasserstein_distance
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _safe_prop(counts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normaliza un vector de conteos → proporciones, con suavizado.
    """
    total = counts.sum()
    if total <= 0:
        return np.full_like(counts, 0.0, dtype=float)

    p = counts.astype(float) / float(total)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return p


def _psi_from_props(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    """
    PSI dado p_ref y p_cur.
    """
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_numeric(
    ref: pd.Series,
    cur: pd.Series,
    n_bins: int = 10,
) -> float | None:
    """
    PSI para columnas numéricas usando cuantiles de ref como bins.
    """
    r = pd.to_numeric(ref, errors="coerce").dropna().values
    c = pd.to_numeric(cur, errors="coerce").dropna().values

    if r.size < 5 or c.size < 5:
        return None

    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.nanpercentile(r, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0

    r_bins = np.histogram(r, bins=edges)[0]
    c_bins = np.histogram(c, bins=edges)[0]
    p_r = _safe_prop(r_bins)
    p_c = _safe_prop(c_bins)
    return _psi_from_props(p_r, p_c)


def psi_categorical(
    ref: pd.Series,
    cur: pd.Series,
) -> float | None:
    """
    PSI para columnas categóricas.
    """
    r = ref.dropna().astype(str)
    c = cur.dropna().astype(str)
    if r.size < 1 or c.size < 1:
        return None

    cats = sorted(set(r.unique()).union(set(c.unique())))
    r_counts = r.value_counts().reindex(cats, fill_value=0).values
    c_counts = c.value_counts().reindex(cats, fill_value=0).values
    p_r = _safe_prop(r_counts)
    p_c = _safe_prop(c_counts)
    return _psi_from_props(p_r, p_c)


def _score_numeric_series(
    a: pd.Series,
    b: pd.Series,
    metric: str,
) -> float | None:
    """
    Calcula estadístico de drift numérico:
    - 'psi'
    - 'ks'
    - 'wasserstein'
    """
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return None

    metric = metric.lower()
    if metric == "psi":
        return psi_numeric(a, b, n_bins=10)

    if metric == "ks":
        if not _HAVE_SCIPY:
            return None
        return float(
            ks_2samp(a, b, alternative="two-sided", mode="auto").statistic
        )

    if metric == "wasserstein":
        if not _HAVE_SCIPY:
            return None
        return float(wasserstein_distance(a, b))

    # fallback: PSI
    return psi_numeric(a, b, n_bins=10)


def _score_categorical_series(
    a: pd.Series,
    b: pd.Series,
    metric: str,
) -> float | None:
    """
    Para categóricas usamos PSI siempre (metric se ignora).
    """
    return psi_categorical(a, b)


# ==========================
# 4. MOTOR STATEFUL POR VENTANAS
# ==========================

def run_drift_for_strategy_multi_metric(
    df: pd.DataFrame,
    window: str,
    strategy: Literal["decay", "golden", "seasonal"],
    metrics: Tuple[str, ...] = ("psi", "ks", "wasserstein"),
    thresholds: Optional[Dict[str, float]] = None,
    min_points: int = 5,
) -> pd.DataFrame:
    """
    Ejecuta detección de drift para UNA estrategia y UNA ventana,
    para TODAS las variables numéricas del df, usando múltiples métricas.

    Supone:
    - df tiene índice DatetimeIndex (ya convertido externamente desde 'date_time').
    - En cada ventana se construye la referencia según `strategy` usando TODO
      el historial hasta t0 (no hay referencia congelada).

    Devuelve un DataFrame con filas por (variable, metric, ventana):
      ['variable','strategy','window','metric','t0','t1','drift_flag',
       'episode_id','stat_value','threshold','state']
    """
    default_thr = {"psi": 0.2, "ks": 0.15, "wasserstein": np.nan}
    thresholds = thresholds or {}

    w = pd.to_timedelta(window)
    t_min = df.index.min()
    t_max = df.index.max()
    t_ends = pd.date_range(t_min + w, t_max, freq=window)

    variables = list(df.columns)

    state = {
        metric: {var: "NORMAL" for var in variables}
        for metric in metrics
    }
    current_episode = {
        metric: {var: 0 for var in variables}
        for metric in metrics
    }

    rows = []

    for t_end in t_ends:
        t0 = t_end - w

        df_hist = df.loc[: t0 - pd.Timedelta(microseconds=1)]
        df_cur = df.loc[t0:t_end]

        if df_hist.empty or df_cur.empty:
            continue

        # Referencia según estrategia
        if strategy == "decay":
            ref_global = ref_decay_prefix_mass(df_hist, now=t_end)
        elif strategy == "golden":
            ref_global = ref_golden(df_hist)
        elif strategy == "seasonal":
            ref_global = ref_seasonal(df_hist, current_end=t_end)
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")

        if ref_global is None or ref_global.empty:
            ref_global = df_hist

        for var in variables:
            cur_series = df_cur[var].dropna()

            if cur_series.size < min_points:
                for metric_name in metrics:
                    rows.append({
                        "variable": var,
                        "strategy": strategy,
                        "window": window,
                        "metric": metric_name,
                        "t0": t0,
                        "t1": t_end,
                        "drift_flag": False,
                        "episode_id": np.nan,
                        "stat_value": None,
                        "threshold": None,
                        "state": state[metric_name][var],
                    })
                continue

            if var in ref_global.columns:
                ref_series = ref_global[var].dropna()
            else:
                ref_series = df_hist[var].dropna()

            for metric_name in metrics:
                base_thr = default_thr.get(metric_name, 0.2)
                thr = thresholds.get(metric_name, base_thr)

                if ref_series.empty:
                    stat_val = None
                    eff_thr = thr
                    drift_flag = False
                else:
                    stat_val = _score_numeric_series(ref_series, cur_series, metric_name)

                    eff_thr = thr
                    if metric_name == "wasserstein" and (
                        eff_thr is None
                        or (isinstance(eff_thr, float) and np.isnan(eff_thr))
                    ):
                        std_ref = pd.to_numeric(
                            ref_series, errors="coerce"
                        ).dropna().std()
                        eff_thr = float(std_ref) * 0.5 if pd.notna(std_ref) else 0.5

                    if stat_val is None or np.isnan(stat_val):
                        drift_flag = False
                    else:
                        drift_flag = bool(stat_val >= eff_thr)

                # Actualizar estado/episodios
                if drift_flag:
                    if state[metric_name][var] == "NORMAL":
                        current_episode[metric_name][var] += 1
                        state[metric_name][var] = "DRIFT"
                else:
                    if state[metric_name][var] == "DRIFT":
                        state[metric_name][var] = "NORMAL"

                rows.append({
                    "variable": var,
                    "strategy": strategy,
                    "window": window,
                    "metric": metric_name,
                    "t0": t0,
                    "t1": t_end,
                    "drift_flag": drift_flag,
                    "episode_id": (
                        current_episode[metric_name][var]
                        if drift_flag else np.nan
                    ),
                    "stat_value": stat_val,
                    "threshold": eff_thr,
                    "state": state[metric_name][var],
                })

    return pd.DataFrame(rows)


def run_drift_all_multi_metric(
    df: pd.DataFrame,
    windows: Tuple[str, ...],
    strategies: Tuple[str, ...],
    metrics: Tuple[str, ...] = ("psi", "ks", "wasserstein"),
    thresholds: Optional[Dict[str, float]] = None,
    min_points: int = 5,
) -> pd.DataFrame:
    """
    Runner multi-ventana, multi-estrategia y multi-métrica.
    Devuelve la concatenación de todos los resultados.
    """
    all_frames = []
    for win in windows:
        for strat in strategies:
            dfw = run_drift_for_strategy_multi_metric(
                df=df,
                window=win,
                strategy=strat,
                metrics=metrics,
                thresholds=thresholds,
                min_points=min_points,
            )
            all_frames.append(dfw)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


# ==========================
# 5. EPISODIOS AUTOMÁTICOS
# ==========================

def windows_to_episodes_multi_metric(
    df_windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compacta secuencias de ventanas con drift_flag=True en episodios automáticos,
    por (window, strategy, metric, variable, episode_id).

    Devuelve:
      ['window','strategy','metric','variable','episode_id',
       'seg_start','seg_end','seg_length','stat_max']
    """
    dfw = df_windows.copy()
    dfw = dfw[dfw["drift_flag"] == True].dropna(subset=["episode_id"])

    if dfw.empty:
        return pd.DataFrame(
            columns=[
                "window", "strategy", "metric", "variable",
                "episode_id", "seg_start", "seg_end",
                "seg_length", "stat_max",
            ]
        )

    rows = []
    for keys, sub in dfw.groupby(
        ["window", "strategy", "metric", "variable", "episode_id"],
        dropna=False
    ):
        win, strat, metric, var, eid = keys
        sub = sub.sort_values("t0")
        seg_start = sub["t0"].min()
        seg_end = sub["t1"].max()
        stat_max = sub["stat_value"].max()
        rows.append({
            "window": win,
            "strategy": strat,
            "metric": metric,
            "variable": var,
            "episode_id": int(eid),
            "seg_start": seg_start,
            "seg_end": seg_end,
            "seg_length": seg_end - seg_start,
            "stat_max": stat_max,
        })

    return pd.DataFrame(rows)
