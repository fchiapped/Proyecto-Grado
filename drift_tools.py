import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Dict
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

Mode = Literal["adjacent", "baseline"]

# -----------------------------
# Utilidades internas
# -----------------------------
@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp

def _ensure_datetime(df: pd.DataFrame, fecha_col: str) -> pd.DataFrame:
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    return df

def _window_iter(df: pd.DataFrame, fecha_col: str,
                 window_days: int = 7,
                 step_days: int = 1) -> List[Window]:
    tmin = df[fecha_col].min().normalize()
    tmax = df[fecha_col].max().normalize()
    if pd.isna(tmin) or pd.isna(tmax):
        return []
    start = tmin
    windows = []
    one_day = pd.Timedelta(days=1)
    w = pd.Timedelta(days=window_days) - pd.Timedelta(seconds=1)
    step = pd.Timedelta(days=step_days)
    while start <= tmax:
        end = start + w
        if start > tmax:
            break
        windows.append(Window(start, min(end, tmax + (one_day - pd.Timedelta(seconds=1)))))
        start = start + step
    return windows

def _extract_series(df: pd.DataFrame, fecha_col: str, col: str, win: Window) -> pd.Series:
    mask = (df[fecha_col] >= win.start) & (df[fecha_col] <= win.end)
    return df.loc[mask, col].dropna()

def _window_coverage_days(df: pd.DataFrame, fecha_col: str, col: str, win: Window) -> int:
    mask = (df[fecha_col] >= win.start) & (df[fecha_col] <= win.end) & (df[col].notna())
    return df.loc[mask, fecha_col].dt.date.nunique()

# -----------------------------
# KS en ventanas deslizantes
# -----------------------------
def ks_sliding_windows(
    df: pd.DataFrame,
    fecha_col: str = "date_time",
    cols: Optional[List[str]] = None,
    window_days: int = 7,
    step_days: int = 1,
    min_days_coverage: int = 4,
    min_points: int = 1000,
    compare: Mode = "adjacent",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Aplica KS (ks_2samp) en ventanas deslizantes para múltiples columnas.

    Retorna un DataFrame con:
      variable, window_start, window_end, ref_start, ref_end,
      n_ref, n_new, stat, pvalue, drift_detectado, detalle
    """
    df = _ensure_datetime(df, fecha_col)
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    wins = _window_iter(df, fecha_col, window_days, step_days)
    results = []

    # Pre-cálculo por (col, window)
    coverage_days: Dict[Tuple[str, int], int] = {}
    counts: Dict[Tuple[str, int], int] = {}
    series_cache: Dict[Tuple[str, int], pd.Series] = {}
    for i, w in enumerate(wins):
        for col in cols:
            cd = _window_coverage_days(df, fecha_col, col, w)
            coverage_days[(col, i)] = cd
            s = _extract_series(df, fecha_col, col, w)
            counts[(col, i)] = len(s)
            series_cache[(col, i)] = s

    for col in cols:
        # Ventanas con suficientes datos
        valid_idx = [i for i, _ in enumerate(wins)
                     if (coverage_days[(col, i)] >= min_days_coverage and
                         counts[(col, i)] >= min_points)]

        if len(valid_idx) < 2:
            for i, w in enumerate(wins):
                results.append({
                    "variable": col,
                    "window_start": w.start,
                    "window_end": w.end,
                    "ref_start": None,
                    "ref_end": None,
                    "n_ref": None,
                    "n_new": counts[(col, i)],
                    "stat": np.nan,
                    "pvalue": np.nan,
                    "drift_detectado": None,
                    "detalle": "pocos datos en ventanas válidas",
                })
            continue

        baseline_i = valid_idx[0]
        for k, i in enumerate(valid_idx[1:]):
            cur_i = i
            ref_i = valid_idx[k] if compare == "adjacent" else baseline_i

            ref_w, cur_w = wins[ref_i], wins[cur_i]
            ref_s = series_cache[(col, ref_i)]
            cur_s = series_cache[(col, cur_i)]

            if len(ref_s) < min_points or len(cur_s) < min_points:
                results.append({
                    "variable": col,
                    "window_start": cur_w.start,
                    "window_end": cur_w.end,
                    "ref_start": ref_w.start,
                    "ref_end": ref_w.end,
                    "n_ref": len(ref_s),
                    "n_new": len(cur_s),
                    "stat": np.nan,
                    "pvalue": np.nan,
                    "drift_detectado": None,
                    "detalle": "pocos datos (min_points)",
                })
                continue

            stat, p = ks_2samp(ref_s, cur_s)
            results.append({
                "variable": col,
                "window_start": cur_w.start,
                "window_end": cur_w.end,
                "ref_start": ref_w.start,
                "ref_end": ref_w.end,
                "n_ref": len(ref_s),
                "n_new": len(cur_s),
                "stat": float(stat),
                "pvalue": float(p),
                "drift_detectado": bool(p < alpha),
                "detalle": "ok",
            })

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values(["variable", "window_start"]).reset_index(drop=True)
    return out

# -----------------------------
# Visualizaciones y utilidades
# -----------------------------
def plot_ks_series(res_df: pd.DataFrame, variable: str, alpha: float = 0.05):
    """Serie temporal de la estadística KS para una variable."""
    if variable not in res_df["variable"].unique():
        raise ValueError(f"'{variable}' no está en results")
    sub = res_df[(res_df["variable"] == variable) & (res_df["detalle"] == "ok")]
    if sub.empty:
        print("No hay resultados para graficar (detalle != ok)")
        return
    x = sub["window_end"]
    y = sub["stat"]

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o")
    plt.axhline(0.1, linestyle="--", alpha=0.3)  # guía opcional
    plt.title(f"KS statistic (sliding) – {variable}")
    plt.xlabel("Fin de ventana")
    plt.ylabel("KS stat")
    plt.tight_layout()
    plt.show()

def plot_ecdf_two_windows(
    df: pd.DataFrame,
    fecha_col: str,
    col: str,
    ref_range: Tuple[pd.Timestamp, pd.Timestamp],
    new_range: Tuple[pd.Timestamp, pd.Timestamp],
    bins: int = 200,
):
    """Grafica CDFs empíricas de dos ventanas específicas (intuición del KS)."""
    df = _ensure_datetime(df, fecha_col)
    ref = df[(df[fecha_col] >= ref_range[0]) & (df[fecha_col] <= ref_range[1])][col].dropna().values
    new = df[(df[fecha_col] >= new_range[0]) & (df[fecha_col] <= new_range[1])][col].dropna().values

    if len(ref) == 0 or len(new) == 0:
        print("Ventanas sin datos suficientes para CDF")
        return

    grid = np.linspace(np.nanmin([ref.min(), new.min()]),
                       np.nanmax([ref.max(), new.max()]), bins)

    def cdf(x, v): return (v[:, None] <= x[None, :]).mean(axis=0)
    cdf_ref = cdf(grid, ref)
    cdf_new = cdf(grid, new)

    plt.figure(figsize=(10, 4))
    plt.plot(grid, cdf_ref, label=f"Ref {ref_range[0].date()}–{ref_range[1].date()}")
    plt.plot(grid, cdf_new, label=f"New {new_range[0].date()}–{new_range[1].date()}")
    plt.legend()
    plt.title(f"CDF – {col}")
    plt.xlabel(col)
    plt.ylabel("F(x)")
    plt.tight_layout()
    plt.show()

def plot_timeseries_with_drift(
    df: pd.DataFrame,
    fecha_col: str,
    col: str,
    res_df: pd.DataFrame,
    resample: Optional[str] = "15min",
    drift_color: str = "tab:red",
    drift_alpha: float = 0.20,
    show_points: bool = False,
):
    """
    Grafica la serie completa y sombrea las ventanas donde 'drift_detectado' es True.
    """
    if col not in df.columns:
        raise ValueError(f"'{col}' no está en el DataFrame")

    df2 = _ensure_datetime(df, fecha_col).sort_values(fecha_col)
    s = df2[[fecha_col, col]].dropna()

    if resample is not None:
        s = s.set_index(fecha_col).resample(resample)[col].median().dropna().reset_index()

    sub = res_df[(res_df["variable"] == col) & (res_df["detalle"] == "ok")]
    if not sub.empty:
        sub = sub.sort_values("window_start")

    plt.figure(figsize=(12, 4))
    if show_points:
        plt.plot(s[fecha_col], s[col], marker='.', linestyle='None', markersize=2)
    else:
        plt.plot(s[fecha_col], s[col])

    if not sub.empty:
        for _, row in sub.iterrows():
            if bool(row.get("drift_detectado", False)):
                ws = pd.Timestamp(row["window_start"])
                we = pd.Timestamp(row["window_end"])
                plt.axvspan(ws, we, color=drift_color, alpha=drift_alpha)

    plt.title(f"Serie de tiempo – {col} (ventanas con drift resaltadas)")
    plt.xlabel("Tiempo")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

def summarize_drift_windows(res_df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Tabla compacta de drift: window_start, window_end, stat, pvalue, drift_detectado.
    """
    sub = res_df[(res_df["variable"] == variable) & (res_df["detalle"] == "ok")].copy()
    if sub.empty:
        return pd.DataFrame(columns=["window_start", "window_end", "stat", "pvalue", "drift_detectado"])
    sub = sub[["window_start", "window_end", "stat", "pvalue", "drift_detectado"]].sort_values("window_start").reset_index(drop=True)
    return sub

# -----------------------------
# Endurecimiento (FDR + tamaño de efecto + persistencia)
# -----------------------------
def _bh_fdr(pvals, alpha=0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR: devuelve máscara booleana de significativos."""
    p = np.asarray(pvals)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = np.arange(1, n + 1)
    thresh = alpha * ranked / n
    passed_sorted = p[order] <= thresh
    if not np.any(passed_sorted):
        out = np.zeros(n, dtype=bool)
        out[order] = passed_sorted
        return out
    kmax = np.where(passed_sorted)[0].max()
    out = np.zeros(n, dtype=bool)
    out[order[:kmax + 1]] = True
    return out

def harden_ks_results(res_df: pd.DataFrame, variable: str,
                      alpha: float = 0.01, ks_min: float = 0.10,
                      min_consecutive: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Endurece la señal de drift combinando:
      - FDR (Benjamini–Hochberg) sobre p-valores,
      - Umbral mínimo de magnitud KS (ks_min),
      - Persistencia: ≥ min_consecutive ventanas consecutivas.

    Retorna:
      res_hard: resultados del variable con columnas extra (p_fdr, ks_ok, drift_raw, drift_hard)
      blocks:  ventanas fusionadas (start, end) para sombreos limpios
    """
    sub = res_df[(res_df["variable"] == variable) & (res_df["detalle"] == "ok")].copy()
    if sub.empty:
        return sub.assign(p_fdr=np.nan, ks_ok=False, drift_raw=False, drift_hard=False), \
               pd.DataFrame(columns=["start", "end"])

    sub = sub.sort_values("window_start").reset_index(drop=True)

    # 1) FDR por ventanas
    sub["p_fdr"] = _bh_fdr(sub["pvalue"].values, alpha=alpha)

    # 2) Tamaño de efecto KS
    sub["ks_ok"] = sub["stat"] >= ks_min

    # 3) Señal cruda
    sub["drift_raw"] = sub["p_fdr"] & sub["ks_ok"]

    # 4) Persistencia (≥ N consecutivas)
    run = 0
    hard_flags = []
    raw_vals = sub["drift_raw"].values.tolist()
    for v in raw_vals:
        run = run + 1 if v else 0
        hard_flags.append(run >= min_consecutive)

    # Propaga hacia atrás dentro de cada corrida
    if min_consecutive > 1:
        i = len(hard_flags) - 1
        while i >= 0:
            if hard_flags[i]:
                j = i
                while j >= 0 and raw_vals[j]:
                    hard_flags[j] = True
                    j -= 1
                i = j
            i -= 1

    sub["drift_hard"] = hard_flags

    # 5) Fusiona ventanas contiguas/solapadas para sombreo
    blocks = []
    open_s, open_e = None, None
    for _, r in sub.iterrows():
        if r["drift_hard"]:
            s, e = pd.Timestamp(r["window_start"]), pd.Timestamp(r["window_end"])
            if open_s is None:
                open_s, open_e = s, e
            else:
                if s <= open_e + pd.Timedelta(seconds=1):
                    open_e = max(open_e, e)
                else:
                    blocks.append({"start": open_s, "end": open_e})
                    open_s, open_e = s, e
        else:
            if open_s is not None:
                blocks.append({"start": open_s, "end": open_e})
                open_s, open_e = None, None
    if open_s is not None:
        blocks.append({"start": open_s, "end": open_e})

    return sub, pd.DataFrame(blocks)

# -----------------------------
# API pública
# -----------------------------
__all__ = [
    "ks_sliding_windows",
    "plot_ks_series",
    "plot_ecdf_two_windows",
    "plot_timeseries_with_drift",
    "summarize_drift_windows",
    "harden_ks_results",
]
