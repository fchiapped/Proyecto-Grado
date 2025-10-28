# drift_funcs.py — versión limpia con métricas propias (PSI) para CSVs confiables
from __future__ import annotations
import warnings, json
from typing import Optional, Literal, List, Iterable, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# Evidently solo para HTML (opcional)
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently import DataDefinition, Dataset

warnings.filterwarnings("ignore")


# ---------------------- Utilidades base ---------------------- #
def strip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Si existe una columna is_outlier booleana/1-0, quita outliers."""
    if "is_outlier" not in df.columns:
        return df
    s = df["is_outlier"].astype(str).str.lower()
    return df.loc[~s.isin(["1","true","t","yes","y"])].drop(columns=["is_outlier"])


def resample_mixed(df: pd.DataFrame, freq: str, agg: Literal["mean","median"]) -> pd.DataFrame:
    """Resamplea numéricas con mean/median y el resto por moda."""
    if df.empty: 
        return df
    num = df.select_dtypes(include="number")
    other_cols = [c for c in df.columns if c not in num.columns]
    num_rs = num.resample(freq).median() if agg == "median" else num.resample(freq).mean()
    if other_cols:
        def _mode(s: pd.Series):
            s = s.dropna()
            if s.empty: return np.nan
            return s.value_counts().index[0]
        other = df[other_cols]
        other_rs = other.resample(freq).agg(_mode)
        out = pd.concat([num_rs, other_rs], axis=1)
    else:
        out = num_rs
    return out[[c for c in df.columns if c in out.columns]]


def build_types_keep_all(ref: pd.DataFrame, cur: pd.DataFrame, dt_col: str, exclude: List[str]) -> tuple[list[str], list[str], list[str]]:
    """Clasifica columnas comunes en numéricas/categóricas (conservador)."""
    common = [c for c in ref.columns.intersection(cur.columns) if c != dt_col and c not in (exclude or [])]
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
    """Para 'golden': genera inicio de ventanas deslizantes."""
    if len(index) == 0: return []
    t, tmax = index.min(), index.max()
    out = []
    while t + win <= tmax:
        out.append(t); t = t + step
    return out


# ---------------------- Estrategias de referencia ---------------------- #
def ref_decay_prefix_mass(df_hist: pd.DataFrame, now: pd.Timestamp, half_life_hours=24*7, target_mass=0.95) -> pd.DataFrame:
    if df_hist.empty: return df_hist
    tau = pd.Timedelta(hours=half_life_hours) / np.log(2)
    dt = (now - df_hist.index)
    w = np.exp(-dt / tau).astype(float)
    order = np.argsort(-df_hist.index.view("i8"))
    w_sorted = w.values[order]
    cum = np.cumsum(w_sorted) / w_sorted.sum()
    cut_idx = np.searchsorted(cum, target_mass, side="left")
    take_pos = order[: (cut_idx + 1)]
    return df_hist.iloc[np.sort(take_pos)]


def ref_golden(df_hist: pd.DataFrame, win="30min", step="10min", k=40) -> pd.DataFrame:
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
    if df_hist.empty: return df_hist.iloc[:0]
    slot = current_end.dayofweek * 24 + current_end.hour
    dw, hh = df_hist.index.dayofweek, df_hist.index.hour
    mask = (dw * 24 + hh) == slot
    hist = df_hist.loc[mask].loc[:current_end]
    if hist.empty: return df_hist.iloc[:0]
    start_lim = current_end - pd.Timedelta(weeks=weeks_back)
    return hist.loc[start_lim:]


# ---------------------- PSI (métricas propias) ---------------------- #
def _safe_prop(counts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.full_like(counts, 0.0, dtype=float)
    p = counts.astype(float) / float(total)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return p

def _psi_from_props(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))

def psi_numeric(ref: pd.Series, cur: pd.Series, n_bins: int = 10) -> float | None:
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

def psi_categorical(ref: pd.Series, cur: pd.Series) -> float | None:
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

def build_metrics_table(
    ref_final: pd.DataFrame,
    cur_final: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    psi_threshold_numeric: float = 0.2,      # 0.1 leve, 0.2 moderado, 0.3 alto
    psi_threshold_categorical: float = 0.2,
    n_bins_numeric: int = 10
) -> pd.DataFrame:
    rows = []

    # NUMÉRICAS
    for col in numeric_cols:
        if col not in ref_final.columns or col not in cur_final.columns:
            continue
        r = pd.to_numeric(ref_final[col], errors="coerce")
        c = pd.to_numeric(cur_final[col], errors="coerce")
        psi = psi_numeric(r, c, n_bins=n_bins_numeric)
        row = {
            "col": col,
            "type": "numeric",
            "ref_count": int(r.count()),
            "cur_count": int(c.count()),
            "ref_missing_pct": float(r.isna().mean() * 100),
            "cur_missing_pct": float(c.isna().mean() * 100),
            "ref_mean": float(r.mean()) if r.count() else np.nan,
            "cur_mean": float(c.mean()) if c.count() else np.nan,
            "ref_std": float(r.std()) if r.count() else np.nan,
            "cur_std": float(c.std()) if c.count() else np.nan,
            "ref_median": float(r.median()) if r.count() else np.nan,
            "cur_median": float(c.median()) if c.count() else np.nan,
            "ref_min": float(r.min()) if r.count() else np.nan,
            "ref_max": float(r.max()) if r.count() else np.nan,
            "cur_min": float(c.min()) if c.count() else np.nan,
            "cur_max": float(c.max()) if c.count() else np.nan,
            "psi": psi,
            "drift_detected": (psi is not None and psi >= psi_threshold_numeric),
            "method": "PSI-quantiles",
            "threshold": psi_threshold_numeric,
        }
        rows.append(row)

    # CATEGÓRICAS
    for col in categorical_cols:
        if col not in ref_final.columns or col not in cur_final.columns:
            continue
        r = ref_final[col]
        c = cur_final[col]
        psi = psi_categorical(r, c)
        rvc = r.value_counts(dropna=True)
        cvc = c.value_counts(dropna=True)
        row = {
            "col": col,
            "type": "categorical",
            "ref_count": int(r.count()),
            "cur_count": int(c.count()),
            "ref_missing_pct": float(r.isna().mean() * 100),
            "cur_missing_pct": float(c.isna().mean() * 100),
            "ref_n_distinct": int(r.nunique(dropna=True)),
            "cur_n_distinct": int(c.nunique(dropna=True)),
            "ref_top": (None if rvc.empty else str(rvc.index[0])),
            "ref_top_freq": (0 if rvc.empty else int(rvc.iloc[0])),
            "cur_top": (None if cvc.empty else str(cvc.index[0])),
            "cur_top_freq": (0 if cvc.empty else int(cvc.iloc[0])),
            "psi": psi,
            "drift_detected": (psi is not None and psi >= psi_threshold_categorical),
            "method": "PSI-categorical",
            "threshold": psi_threshold_categorical,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    preferred = [
        "col","type","drift_detected","psi","method","threshold",
        "ref_count","cur_count","ref_missing_pct","cur_missing_pct",
        "ref_mean","cur_mean","ref_std","cur_std","ref_median","cur_median",
        "ref_min","ref_max","cur_min","cur_max",
        "ref_n_distinct","cur_n_distinct","ref_top","ref_top_freq","cur_top","cur_top_freq",
    ]
    df = df[[c for c in preferred if c in df.columns]]
    return df

def summarize_overall(ref_final: pd.DataFrame, cur_final: pd.DataFrame, metrics_df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> dict:
    n_total = len(set(ref_final.columns).intersection(cur_final.columns))
    n_drifted = int(metrics_df["drift_detected"].fillna(False).sum()) if not metrics_df.empty else 0
    out = {
        "ref_rows": int(ref_final.shape[0]),
        "cur_rows": int(cur_final.shape[0]),
        "n_columns_total": int(n_total),
        "n_columns_drifted": int(n_drifted),
        "drift_rate_pct": float(100.0 * n_drifted / n_total) if n_total else 0.0,
        "n_numeric": int(len(numeric_cols)),
        "n_categorical": int(len(categorical_cols)),
    }
    return out

# ---------------------- Función principal por planta ---------------------- #
def make_report_for_plant(
    df: pd.DataFrame,
    output_dir: Path,
    strategy: Literal["decay","golden","seasonal"],
    CURRENT_WINDOW: str,
    RESAMPLE: Optional[str],
    RESAMPLE_AGG: Literal["mean","median"],
    EXCLUDE_COLUMNS: list[str],
    NUM_METHOD: Literal["auto","ks","wasserstein","psi","anderson","cramer","mannwhitney"],
    NUM_THRESHOLD: Optional[float],
    DECAY_HALF_LIFE_HOURS: int = 24*7,
    DECAY_WEIGHT_MASS: float = 0.95,
    GOLDEN_WIN: str = "30min",
    GOLDEN_STEP: str = "10min",
    GOLDEN_K: int = 40,
    SEASONAL_WEEKS_BACK: int = 12,
    plant_name: str = "planta",
    flag_csv: Optional[Path] = None,
    *,
    SAVE_HTML: bool = True  # permite desactivar Evidently si no quieres HTML
) -> Path:
    dt = "date_time"
    df = df.copy()
    df[dt] = pd.to_datetime(df[dt], errors="coerce")
    df = df.dropna(subset=[dt]).sort_values(dt).set_index(dt)
    df = strip_outliers(df)

    # Aplicar flags (si existen)
    if flag_csv and Path(flag_csv).exists():
        flags = pd.read_csv(flag_csv, parse_dates=["date_time"])
        flags["date_time"] = pd.to_datetime(flags["date_time"]).dt.floor("min")
        df.index = df.index.floor("min")
        df = df.merge(flags, left_index=True, right_on="date_time", how="left").set_index("date_time")
        nd_cols = [c for c in df.columns if c.startswith("nd_")]
        for nd_col in nd_cols:
            var = nd_col.replace("nd_", "")
            if var in df.columns:
                mask = ~df[nd_col]
                df.loc[~mask, var] = np.nan
        drop_cols = ["valid_for_drift", "nd_any", "nd_all"] + nd_cols
        df = df[[c for c in df.columns if c not in drop_cols]]

    now = df.index.max()
    cur_start = now - pd.to_timedelta(CURRENT_WINDOW)
    cur = df.loc[cur_start:now]
    hist = df.loc[:cur_start - pd.Timedelta(nanoseconds=1)]

    # Selección de referencia
    if strategy == "decay":
        ref_global = ref_decay_prefix_mass(hist, now, DECAY_HALF_LIFE_HOURS, DECAY_WEIGHT_MASS)
    elif strategy == "golden":
        ref_global = ref_golden(hist, GOLDEN_WIN, GOLDEN_STEP, GOLDEN_K)
    else:
        ref_global = ref_seasonal(hist, now, SEASONAL_WEEKS_BACK)
    if ref_global.empty:
        ref_global = hist

    # Intersección de columnas
    common_cols = sorted(set(ref_global.columns).intersection(cur.columns) - {dt} - set(EXCLUDE_COLUMNS or []))
    if not common_cols:
        raise ValueError("No hay columnas comunes para comparar.")
    ref_final = ref_global[common_cols].copy()
    cur_final = cur[common_cols].copy()

    # Resample (opcional)
    if RESAMPLE:
        ref_final = resample_mixed(ref_final, RESAMPLE, RESAMPLE_AGG).dropna(how="all")
        cur_final = resample_mixed(cur_final, RESAMPLE, RESAMPLE_AGG).dropna(how="all")

    # Tipos
    numeric_cols, categorical_cols, _dropped_all_nan = build_types_keep_all(ref_final, cur_final, dt_col=dt, exclude=(EXCLUDE_COLUMNS or []))

    # Métricas propias (PSI + stats) -> CSV confiable
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = build_metrics_table(
        ref_final, cur_final,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        psi_threshold_numeric=(NUM_THRESHOLD if (NUM_THRESHOLD is not None and NUM_METHOD in ["psi","auto"]) else 0.2),
        psi_threshold_categorical=(NUM_THRESHOLD if (NUM_THRESHOLD is not None and NUM_METHOD in ["psi","auto"]) else 0.2),
        n_bins_numeric=10,
    )
    metrics_path = output_dir / f"{plant_name}_{strategy}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    # Resumen global -> JSON
    overall = summarize_overall(ref_final, cur_final, metrics_df, numeric_cols, categorical_cols)
    (output_dir / f"{plant_name}_{strategy}_overall.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # HTML de Evidently (opcional)
    out_html = output_dir / f"{plant_name}_{strategy}.html"
    if SAVE_HTML:
        definition = DataDefinition(
            numerical_columns=numeric_cols if numeric_cols else None,
            categorical_columns=categorical_cols if categorical_cols else None
        )
        report = Report(metrics=[DataDriftPreset()])
        ds_ref = Dataset.from_pandas(ref_final.reset_index(drop=True), data_definition=definition)
        ds_cur = Dataset.from_pandas(cur_final.reset_index(drop=True), data_definition=definition)
        snap = report.run(reference_data=ds_ref, current_data=ds_cur)
        snap.save_html(str(out_html))
    else:
        # crear un marcador vacío por si algo espera el path
        try:
            out_html.write_text("<html><body><p>HTML deshabilitado (SAVE_HTML=False)</p></body></html>", encoding="utf-8")
        except Exception:
            pass

    return out_html

def run_drift_batch(
    plant_names: Iterable[str],
    strategies: Iterable[Literal["decay","golden","seasonal"]],
    plant_files: Dict[str, Path],
    flag_files: Dict[str, Path],
    output_root: Path,
    *,
    CURRENT_WINDOW: str,
    RESAMPLE: Optional[str],
    RESAMPLE_AGG: Literal["mean","median"],
    EXCLUDE_COLUMNS: list[str],
    NUM_METHOD: Literal["auto","ks","wasserstein","psi","anderson","cramer","mannwhitney"],  # compat
    NUM_THRESHOLD: Optional[float],
    DECAY_HALF_LIFE_HOURS: int = 24*7,
    DECAY_WEIGHT_MASS: float = 0.95,
    GOLDEN_WIN: str = "30min",
    GOLDEN_STEP: str = "10min",
    GOLDEN_K: int = 40,
    SEASONAL_WEEKS_BACK: int = 12,
    COMMON_LAST_Q: float = 0.25,  # compat (no usado)
    SAVE_HTML: bool = True
) -> Tuple[Dict[Tuple[str,str], Path], Dict[Tuple[str,str], str]]:

    paths: Dict[Tuple[str,str], Path] = {}
    errors: Dict[Tuple[str,str], str] = {}

    for plant in plant_names:
        try:
            df = pd.read_csv(plant_files[plant])
        except Exception as e:
            for strat in strategies:
                errors[(plant, strat)] = f"ERROR al leer CSV de {plant}: {e}"
            continue

        out_dir = output_root / plant
        for strat in strategies:
            key = (plant, strat)
            try:
                out_path = make_report_for_plant(
                    df=df,
                    output_dir=out_dir,
                    strategy=strat,
                    CURRENT_WINDOW=CURRENT_WINDOW,
                    RESAMPLE=RESAMPLE,
                    RESAMPLE_AGG=RESAMPLE_AGG,
                    EXCLUDE_COLUMNS=EXCLUDE_COLUMNS,
                    NUM_METHOD=NUM_METHOD,
                    NUM_THRESHOLD=NUM_THRESHOLD,
                    DECAY_HALF_LIFE_HOURS=DECAY_HALF_LIFE_HOURS,
                    DECAY_WEIGHT_MASS=DECAY_WEIGHT_MASS,
                    GOLDEN_WIN=GOLDEN_WIN,
                    GOLDEN_STEP=GOLDEN_STEP,
                    GOLDEN_K=GOLDEN_K,
                    SEASONAL_WEEKS_BACK=SEASONAL_WEEKS_BACK,
                    plant_name=plant,
                    flag_csv=flag_files.get(plant),
                    SAVE_HTML=SAVE_HTML
                )
                paths[key] = out_path
                print(f"[OK] {plant} · {strat} → {out_path.name}")
            except Exception as e:
                errors[key] = str(e)
                print(f"[FAIL] {plant} · {strat}: {e}")

    return paths, errors
#--------------------------------------------------------------------------------------------------------#

