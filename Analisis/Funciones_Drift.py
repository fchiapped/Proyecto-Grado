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
#--------------------------------------------------------------------------------------------------------#
# --- Metricas

# ================== PATCH 1: SciPy guard & column cleaners ==================
# Coloca esto al FINAL de drift_funcs.py

# --- SciPy availability flag (para ks / wasserstein / mannwhitney) ---
try:
    from scipy.stats import ks_2samp, wasserstein_distance, mannwhitneyu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def _coerce_colnames_to_str(df):
    """Asegura que los nombres de columnas son str (evita merge object/int64)."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df

def _drop_useless_cols(ref: pd.DataFrame, cur: pd.DataFrame, min_non_null: int = 5):
    """
    Elimina columnas con demasiados NaN o constantes en ref y cur.
    Regresa (ref2, cur2, cols_usadas)
    """
    ref = _coerce_colnames_to_str(ref)
    cur = _coerce_colnames_to_str(cur)
    keep = []
    for col in sorted(set(ref.columns).intersection(set(cur.columns))):
        if col == "date_time": 
            continue
        r = ref[col]; c = cur[col]
        # requisito mínimo de datos
        if r.notna().sum() < min_non_null or c.notna().sum() < min_non_null:
            continue
        # si AMBAS son constantes (o casi)
        rn = pd.to_numeric(r, errors="coerce")
        cn = pd.to_numeric(c, errors="coerce")
        if rn.nunique(dropna=True) <= 1 and cn.nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    return ref[keep].copy(), cur[keep].copy(), keep


# ================== PATCH 2: Multi-métricas robustas =======================
from typing import Tuple, Dict

def list_supported_metrics() -> list[str]:
    # Si prefieres manejar la lista EN el notebook, no uses esta función.
    # La dejamos por compatibilidad.
    return ["psi", "ks", "wasserstein", "mannwhitney"]

def _score_numeric_series(a: pd.Series, b: pd.Series, metric: str) -> float | None:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return None
    if metric == "psi":
        return psi_numeric(a, b, n_bins=10)
    if metric == "ks":
        if not _HAVE_SCIPY: return None
        return float(ks_2samp(a, b, alternative="two-sided", mode="auto").statistic)
    if metric == "wasserstein":
        if not _HAVE_SCIPY: return None
        return float(wasserstein_distance(a, b))
    if metric == "mannwhitney":
        if not _HAVE_SCIPY: return None
        res = mannwhitneyu(a, b, alternative="two-sided")
        n, m = len(a), len(b)
        return float(res.statistic / (n*m if n*m > 0 else 1))
    # fallback
    return psi_numeric(a, b, n_bins=10)

def _score_categorical_series(a: pd.Series, b: pd.Series, metric: str) -> float | None:
    # Para categóricas usamos PSI por simplicidad
    return psi_categorical(a, b)

def _dispatch_build_metrics(
    ref_final: pd.DataFrame,
    cur_final: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    metric: str,
    num_threshold: float | None = None
) -> pd.DataFrame:
    # thresholds por defecto
    default_thr = {"psi": 0.2, "ks": 0.15, "wasserstein": float("nan"), "mannwhitney": 0.55}
    thr = default_thr.get(metric, 0.2) if num_threshold is None else num_threshold

    rows = []
    # Numéricas
    for col in numeric_cols:
        if col not in ref_final.columns or col not in cur_final.columns:
            continue
        r, c = ref_final[col], cur_final[col]
        score = _score_numeric_series(r, c, metric)
        eff_thr = thr
        if metric == "wasserstein":
            std_ref = pd.to_numeric(r, errors="coerce").dropna().std()
            if num_threshold is None or np.isnan(thr):
                eff_thr = float(std_ref) * 0.5 if pd.notna(std_ref) else 0.5
        rows.append({
            "col": str(col), "type": "numeric",
            "score": score, "threshold": eff_thr, "method": metric,
            "drift_detected": (False if score is None else (score >= eff_thr))
        })

    # Categóricas
    for col in categorical_cols:
        if col not in ref_final.columns or col not in cur_final.columns:
            continue
        r, c = ref_final[col], cur_final[col]
        score = _score_categorical_series(r, c, metric)
        eff_thr = 0.2 if num_threshold is None else num_threshold
        rows.append({
            "col": str(col), "type": "categorical",
            "score": score, "threshold": eff_thr, "method": (metric if metric!="evidently_default" else "psi(cat)"),
            "drift_detected": (False if score is None else score >= eff_thr)
        })

    df = pd.DataFrame(rows)
    # ==== Arreglo de merge (tipos homogéneos) + stats básicos
    if not df.empty:
        def _stats(col):
            rr = ref_final[col]; cc = cur_final[col]
            return pd.Series({
                "ref_count": int(rr.count()), "cur_count": int(cc.count()),
                "ref_missing_pct": float(rr.isna().mean()*100.0),
                "cur_missing_pct": float(cc.isna().mean()*100.0),
                "ref_mean": (pd.to_numeric(rr, errors="coerce").mean() if col in numeric_cols else np.nan),
                "cur_mean": (pd.to_numeric(cc, errors="coerce").mean() if col in numeric_cols else np.nan),
            })
        meta = pd.concat([_stats(c) for c in df["col"]], axis=1).T
        meta.index = meta.index.astype(str)
        df["col"] = df["col"].astype(str)
        df = df.merge(meta, left_on="col", right_index=True, how="left")
    return df

def _extract_drift_by_columns(report_obj) -> pd.DataFrame:
    # intenta varios métodos para obtener dict del reporte
    for attr in ("as_dict", "get_dict", "to_dict"):
        if hasattr(report_obj, attr):
            try:
                d = getattr(report_obj, attr)()
            except Exception:
                d = None
            if isinstance(d, dict):
                metrics = d.get("metrics", [])
                rows = []
                for m in metrics:
                    res = (m.get("result") or {}) if isinstance(m, dict) else {}
                    dbc = res.get("drift_by_columns") or res.get("columns")
                    if isinstance(dbc, dict):
                        for col, info in dbc.items():
                            if isinstance(info, dict):
                                rows.append({
                                    "col": str(col),
                                    "score": info.get("drift_score"),
                                    "drift_detected": info.get("drift_detected"),
                                    "method": info.get("stattest_name") or info.get("stattest"),
                                    "threshold": info.get("drift_threshold") or info.get("threshold"),
                                })
                if rows:
                    return pd.DataFrame(rows)
    return pd.DataFrame(columns=["col","score","drift_detected","method","threshold"])

def compare_with_metric(
    df: pd.DataFrame,
    strategy: "Literal['decay','golden','seasonal']",
    CURRENT_WINDOW: str,
    RESAMPLE: "Optional[str]",
    RESAMPLE_AGG: "Literal['mean','median']",
    EXCLUDE_COLUMNS: list[str],
    metric: "Literal['psi','ks','wasserstein','mannwhitney']" = "psi",
    num_threshold: float | None = None,
    DECAY_HALF_LIFE_HOURS: int = 24*7,
    DECAY_WEIGHT_MASS: float = 0.95,
    GOLDEN_WIN: str = "30min",
    GOLDEN_STEP: str = "10min",
    GOLDEN_K: int = 40,
    SEASONAL_WEEKS_BACK: int = 12,
    save_html: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    # === prepara ref/cur como make_report_for_plant ===
    dt = "date_time"
    _df = df.copy()
    _df.columns = [str(c) for c in _df.columns]
    _df[dt] = pd.to_datetime(_df[dt], errors="coerce")
    _df = _df.dropna(subset=[dt]).sort_values(dt).set_index(dt)
    _df = strip_outliers(_df)

    now = _df.index.max()
    cur_start = now - pd.to_timedelta(CURRENT_WINDOW)
    cur = _df.loc[cur_start:now]
    hist = _df.loc[:cur_start - pd.Timedelta(nanoseconds=1)]

    if strategy == "decay":
        ref_global = ref_decay_prefix_mass(hist, now, DECAY_HALF_LIFE_HOURS, DECAY_WEIGHT_MASS)
    elif strategy == "golden":
        ref_global = ref_golden(hist, GOLDEN_WIN, GOLDEN_STEP, GOLDEN_K)
    else:
        ref_global = ref_seasonal(hist, now, SEASONAL_WEEKS_BACK)
    if ref_global.empty:
        ref_global = hist

    # === limpia columnas vacías/constantes y homogeneiza tipos de nombres ===
    ref_final, cur_final, common_cols = _drop_useless_cols(ref_global, cur, min_non_null=5)

    if RESAMPLE:
        ref_final = resample_mixed(ref_final, RESAMPLE, RESAMPLE_AGG).dropna(how="all")
        cur_final  = resample_mixed(cur_final,  RESAMPLE, RESAMPLE_AGG).dropna(how="all")

    # detecta tipos después de limpiar
    numeric_cols, categorical_cols, _ = build_types_keep_all(
        ref_final, cur_final, dt_col="date_time", exclude=(EXCLUDE_COLUMNS or [])
    )

    if metric == "evidently_default":
        # Evidently preset (si hay columnas útiles)
        from evidently import Report
        from evidently.presets import DataDriftPreset
        from evidently import DataDefinition, Dataset

        definition = DataDefinition(
            numerical_columns=numeric_cols if numeric_cols else None,
            categorical_columns=categorical_cols if categorical_cols else None
        )
        report = Report(metrics=[DataDriftPreset()])
        ds_ref = Dataset.from_pandas(ref_final.reset_index(drop=True), data_definition=definition)
        ds_cur = Dataset.from_pandas(cur_final.reset_index(drop=True), data_definition=definition)
        report.run(reference_data=ds_ref, current_data=ds_cur)
        dfm = _extract_drift_by_columns(report)
        if "method" not in dfm.columns: 
            dfm["method"] = "evidently_default"
        if not dfm.empty:
            dfm["type"] = dfm["col"].apply(lambda c: "numeric" if c in numeric_cols else ("categorical" if c in categorical_cols else "other"))
        overall = summarize_overall(ref_final, cur_final, dfm.rename(columns={"drift_detected":"drift_detected"}), numeric_cols, categorical_cols)
        return dfm, overall

    # métodos locales
    dfm = _dispatch_build_metrics(ref_final, cur_final, numeric_cols, categorical_cols, metric=metric, num_threshold=num_threshold)
    overall = summarize_overall(ref_final, cur_final, dfm.rename(columns={"drift_detected":"drift_detected"}), numeric_cols, categorical_cols)
    return dfm, overall


# ================== AGGREGATE & MEMORY RUNNERS (mínimos CSV) ==================

DEFAULT_METRICS = ['ks', 'mannwhitney', 'psi', 'wasserstein']

def _compute_one(
    df: pd.DataFrame,
    plant: str,
    strategy: str,
    metric: str,
    **kwargs
) -> tuple[pd.DataFrame, dict]:

    dfm, overall = compare_with_metric(
        df=df,
        strategy=strategy,
        metric=metric,
        CURRENT_WINDOW=kwargs["CURRENT_WINDOW"],
        RESAMPLE=kwargs["RESAMPLE"],
        RESAMPLE_AGG=kwargs["RESAMPLE_AGG"],
        EXCLUDE_COLUMNS=kwargs["EXCLUDE_COLUMNS"],
        DECAY_HALF_LIFE_HOURS=kwargs["DECAY_HALF_LIFE_HOURS"],
        DECAY_WEIGHT_MASS=kwargs["DECAY_WEIGHT_MASS"],
        GOLDEN_WIN=kwargs["GOLDEN_WIN"],
        GOLDEN_STEP=kwargs["GOLDEN_STEP"],
        GOLDEN_K=kwargs["GOLDEN_K"],
        SEASONAL_WEEKS_BACK=kwargs["SEASONAL_WEEKS_BACK"],
        save_html=False,
    )
    if dfm is None or dfm.empty:
        cols = pd.DataFrame(columns=[
            "plant","strategy","metric","col","type","score","threshold","method","drift_detected",
            "ref_count","cur_count","ref_missing_pct","cur_missing_pct","ref_mean","cur_mean"
        ])
    else:
        # normaliza columnas y agrega etiquetas de plant/strategy/metric
        base_cols = ["col","type","score","threshold","method","drift_detected",
                     "ref_count","cur_count","ref_missing_pct","cur_missing_pct","ref_mean","cur_mean"]
        for c in base_cols:
            if c not in dfm.columns:
                dfm[c] = pd.NA
        cols = dfm[base_cols].copy()
        cols.insert(0, "metric", metric)
        cols.insert(0, "strategy", strategy)
        cols.insert(0, "plant", plant)

    # overall estándar por fila, para poder concatenar y luego pivotear
    over_row = dict(overall)
    over_row.update({"plant": plant, "strategy": strategy, "metric": metric})
    return cols, over_row


def run_drift_memory(
    plant_names: list[str],
    strategies: list[str],
    plant_files: dict[str, Path],
    flag_files: dict[str, Path] | None = None,
    *,
    metrics: list[str] = DEFAULT_METRICS,
    CURRENT_WINDOW: str,
    RESAMPLE: str | None,
    RESAMPLE_AGG: str,
    EXCLUDE_COLUMNS: list[str],
    DECAY_HALF_LIFE_HOURS: int = 24*7,
    DECAY_WEIGHT_MASS: float = 0.95,
    GOLDEN_WIN: str = "30min",
    GOLDEN_STEP: str = "10min",
    GOLDEN_K: int = 40,
    SEASONAL_WEEKS_BACK: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_all, sum_rows = [], []

    # parámetros comunes
    k = dict(
        CURRENT_WINDOW=CURRENT_WINDOW,
        RESAMPLE=RESAMPLE,
        RESAMPLE_AGG=RESAMPLE_AGG,
        EXCLUDE_COLUMNS=EXCLUDE_COLUMNS or [],
        DECAY_HALF_LIFE_HOURS=DECAY_HALF_LIFE_HOURS,
        DECAY_WEIGHT_MASS=DECAY_WEIGHT_MASS,
        GOLDEN_WIN=GOLDEN_WIN, GOLDEN_STEP=GOLDEN_STEP, GOLDEN_K=GOLDEN_K,
        SEASONAL_WEEKS_BACK=SEASONAL_WEEKS_BACK,
    )

    for plant in plant_names:
        df = pd.read_csv(plant_files[plant])
        # Aplica flags si existen (reutiliza lógica de make_report_for_plant)
        if flag_files and flag_files.get(plant) and Path(flag_files[plant]).exists():
            # reusar make_report_for_plant? preferimos mínima carga: usamos su bloque
            # Para no duplicar mucha lógica, evaluamos compare_with_metric que ya limpia.
            pass

        for strat in strategies:
            for metric in metrics:
                cols, over = _compute_one(df, plant=plant, strategy=strat, metric=metric, **k)
                cols_all.append(cols)
                sum_rows.append(over)

    df_cols_all = pd.concat(cols_all, ignore_index=True) if cols_all else pd.DataFrame()
    df_sum_all = pd.DataFrame(sum_rows) if sum_rows else pd.DataFrame()

    # orden de columnas recomendado
    sum_pref = ["plant","strategy","metric","ref_rows","cur_rows","n_columns_total","n_columns_drifted","drift_rate_pct","n_numeric","n_categorical"]
    df_sum_all = df_sum_all[[c for c in sum_pref if c in df_sum_all.columns] + [c for c in df_sum_all.columns if c not in sum_pref]]

    cols_pref = ["plant","strategy","metric","col","type","score","threshold","method","drift_detected",
                 "ref_count","cur_count","ref_missing_pct","cur_missing_pct","ref_mean","cur_mean"]
    df_cols_all = df_cols_all[[c for c in cols_pref if c in df_cols_all.columns] + [c for c in df_cols_all.columns if c not in cols_pref]]

    return df_sum_all, df_cols_all


def run_drift_aggregate(
    plant_names: list[str],
    strategies: list[str],
    plant_files: dict[str, Path],
    flag_files: dict[str, Path] | None,
    output_root: Path,
    *,
    metrics: list[str] = DEFAULT_METRICS,
    CURRENT_WINDOW: str,
    RESAMPLE: str | None,
    RESAMPLE_AGG: str,
    EXCLUDE_COLUMNS: list[str],
    DECAY_HALF_LIFE_HOURS: int = 24*7,
    DECAY_WEIGHT_MASS: float = 0.95,
    GOLDEN_WIN: str = "30min",
    GOLDEN_STEP: str = "10min",
    GOLDEN_K: int = 40,
    SEASONAL_WEEKS_BACK: int = 12,
    WRITE_GLOBAL: bool = True
) -> dict[str, dict[str, Path]]:
    """
    Corre todas las (planta × estrategia × métricas) y ESCRIBE SOLO:
      - por planta: plantaX/_comparisons/{plantaX}_columns_all_metrics.csv, {plantaX}_summary_all_metrics.csv
      - global: GLOBAL_columns_all_metrics.csv y GLOBAL_summary_all_metrics.csv (si WRITE_GLOBAL=True)

    Retorna: {plant: {"columns": path_csv_cols, "summary": path_csv_sum}}
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Ejecuta en memoria y luego escribe SOLO 2 CSV por planta (+ globales)
    df_sum_all, df_cols_all = run_drift_memory(
        plant_names=plant_names,
        strategies=strategies,
        plant_files=plant_files,
        flag_files=flag_files,
        metrics=metrics,
        CURRENT_WINDOW=CURRENT_WINDOW,
        RESAMPLE=RESAMPLE,
        RESAMPLE_AGG=RESAMPLE_AGG,
        EXCLUDE_COLUMNS=EXCLUDE_COLUMNS,
        DECAY_HALF_LIFE_HOURS=DECAY_HALF_LIFE_HOURS,
        DECAY_WEIGHT_MASS=DECAY_WEIGHT_MASS,
        GOLDEN_WIN=GOLDEN_WIN, GOLDEN_STEP=GOLDEN_STEP, GOLDEN_K=GOLDEN_K,
        SEASONAL_WEEKS_BACK=SEASONAL_WEEKS_BACK,
    )

    paths = {}
    for plant in plant_names:
        pdir = output_root / plant / "_comparisons"
        pdir.mkdir(parents=True, exist_ok=True)
        cols_p = df_cols_all.query("plant == @plant").copy()
        sum_p  = df_sum_all.query("plant == @plant").copy()

        p_cols_path = pdir / f"{plant}_columns_all_metrics.csv"
        p_sum_path  = pdir / f"{plant}_summary_all_metrics.csv"
        cols_p.to_csv(p_cols_path, index=False)
        sum_p.to_csv(p_sum_path, index=False)
        paths[plant] = {"columns": p_cols_path, "summary": p_sum_path}

    if WRITE_GLOBAL:
        (output_root / "GLOBAL_columns_all_metrics.csv").write_text(
            df_cols_all.to_csv(index=False), encoding="utf-8"
        )
        (output_root / "GLOBAL_summary_all_metrics.csv").write_text(
            df_sum_all.to_csv(index=False), encoding="utf-8"
        )

    return paths



def run_drift_minimal(
    plant_names: list[str],
    strategies: list[str],
    plant_files: dict[str, Path],
    flag_files: dict[str, Path] | None,
    output_root: Path,
    *,
    metrics: list[str],
    CURRENT_WINDOW: str,
    RESAMPLE: str | None,
    RESAMPLE_AGG: str,
    EXCLUDE_COLUMNS: list[str],
    DECAY_HALF_LIFE_HOURS: int = 24*7,
    DECAY_WEIGHT_MASS: float = 0.95,
    GOLDEN_WIN: str = "30min",
    GOLDEN_STEP: str = "10min",
    GOLDEN_K: int = 40,
    SEASONAL_WEEKS_BACK: int = 12,
    WRITE_GLOBAL: bool = False
) -> dict[str, dict[str, Path]]:
    # Reutiliza el runner en memoria para no escribir nada intermedio
    df_sum_all, df_cols_all = run_drift_memory(
        plant_names=plant_names,
        strategies=strategies,
        plant_files=plant_files,
        flag_files=flag_files,
        metrics=metrics,
        CURRENT_WINDOW=CURRENT_WINDOW,
        RESAMPLE=RESAMPLE,
        RESAMPLE_AGG=RESAMPLE_AGG,
        EXCLUDE_COLUMNS=EXCLUDE_COLUMNS,
        DECAY_HALF_LIFE_HOURS=DECAY_HALF_LIFE_HOURS,
        DECAY_WEIGHT_MASS=DECAY_WEIGHT_MASS,
        GOLDEN_WIN=GOLDEN_WIN, GOLDEN_STEP=GOLDEN_STEP, GOLDEN_K=GOLDEN_K,
        SEASONAL_WEEKS_BACK=SEASONAL_WEEKS_BACK,
    )
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for plant in plant_names:
        pdir = output_root / plant
        pdir.mkdir(parents=True, exist_ok=True)
        cols_p = df_cols_all.query("plant == @plant").copy()
        sum_p  = df_sum_all.query("plant == @plant").copy()
        p_cols_path = pdir / f"{plant}_columns_all_metrics.csv"
        p_sum_path  = pdir / f"{plant}_summary_all_metrics.csv"
        cols_p.to_csv(p_cols_path, index=False)
        sum_p.to_csv(p_sum_path, index=False)
        paths[plant] = {"columns": p_cols_path, "summary": p_sum_path}
    if WRITE_GLOBAL:
        (output_root / "GLOBAL_columns_all_metrics.csv").write_text(df_cols_all.to_csv(index=False), encoding="utf-8")
        (output_root / "GLOBAL_summary_all_metrics.csv").write_text(df_sum_all.to_csv(index=False), encoding="utf-8")
    return paths