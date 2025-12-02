# evaluacion_drift.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# 1) Utilidad: eventos (start/end) → intervalos manuales
# -------------------------------------------------------------------

#Convierte eventos por variable en intervalos
def events_to_intervals(ev: pd.DataFrame) -> pd.DataFrame:
    has_type = "drift_type" in ev.columns
    rows = []
    for var, g in ev.groupby("variable", sort=True):
        open_t = None
        open_type = None
        for _, r in g.iterrows():
            evt = str(r["event"]).lower()
            dt_val = r["drift_type"] if has_type else "unknown"

            if evt == "start":
                open_t = r["date_time"]
                open_type = dt_val
            elif evt == "end" and open_t is not None and r["date_time"] > open_t:
                rows.append({
                    "variable": var,
                    "manual_start": open_t,
                    "manual_end": r["date_time"],
                    "drift_type": open_type,
                })
                open_t = None
                open_type = None
    return pd.DataFrame(rows)


# Evaluación episodios automáticos vs manuales
def evaluate_episodes_vs_manual_multi_metric(
    df_episodes_auto: pd.DataFrame,
    intervals_manual: pd.DataFrame,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    detection_delay_hours: float = 6.0,
):
    """
    - TP/FP/FN y F1 por (window, strategy, metric)
    - Métricas de tiempo: Precision_time, Recall_time, F1_time
    - Delay medio y mediano (shift de +detection_delay_hours)
    - extra_ratio_auto, false_alarms_per_day, coverage_mean/median

    Devuelve:
      eval_df, manual_marked, auto_marked
    """
    if df_episodes_auto.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    man = intervals_manual.copy()
    total_days = max((t_max - t_min).total_seconds() / (3600 * 24), 1e-9)

    results = []
    marks_manual_all = []
    marks_auto_all   = []

    for (win, strat, metric), auto_sub in df_episodes_auto.groupby(
        ["window", "strategy", "metric"], dropna=False
    ):
        vars_in_auto = sorted(auto_sub["variable"].unique())
        man_sub = man[man["variable"].isin(vars_in_auto)].copy()
        if man_sub.empty and auto_sub.empty:
            continue

        manual_matches = []
        coverage_vals  = []
        delay_vals     = []

        for _, mrow in man_sub.iterrows():
            v  = mrow["variable"]
            ms = mrow["manual_start"]
            me = mrow["manual_end"]

            rel_auto = auto_sub[auto_sub["variable"] == v]

            total_overlap = pd.Timedelta(0)
            first_det = None

            for _, arow in rel_auto.iterrows():
                as_ = arow["seg_start"]
                ae  = arow["seg_end"]

                start = max(ms, as_)
                end   = min(me, ae)
                if end > start:
                    total_overlap += (end - start)

                    det_candidate = as_
                    if det_candidate < ms:
                        det_candidate = ms
                    if first_det is None or det_candidate < first_det:
                        first_det = det_candidate

            matched = total_overlap > pd.Timedelta(0)
            manual_matches.append(bool(matched))

            dur = me - ms
            if dur.total_seconds() > 0:
                cov_val = total_overlap.total_seconds() / dur.total_seconds()
            else:
                cov_val = 0.0
            coverage_vals.append(cov_val)

            if first_det is None:
                delay_vals.append(np.nan)
            else:
                first_det_shifted = first_det + pd.Timedelta(hours=detection_delay_hours)
                delay = (first_det_shifted - ms).total_seconds() / 3600.0
                if delay < 0:
                    delay = 0.0
                delay_vals.append(delay)

        man_sub["matched_auto"] = manual_matches
        man_sub["coverage"]     = coverage_vals
        man_sub["delay_hours"]  = delay_vals

        TP = int(man_sub["matched_auto"].sum()) if not man_sub.empty else 0
        FN = int((~man_sub["matched_auto"]).sum()) if not man_sub.empty else 0

        auto_sub = auto_sub.copy()
        auto_matches = []
        for _, arow in auto_sub.iterrows():
            v  = arow["variable"]
            as_ = arow["seg_start"]
            ae  = arow["seg_end"]

            overlap = (
                (man_sub["variable"] == v) &
                ~(man_sub["manual_end"] < as_) &
                ~(man_sub["manual_start"] > ae)
            ).any()
            auto_matches.append(overlap)

        auto_sub["matched_manual"] = auto_matches
        FP = int((~auto_sub["matched_manual"]).sum()) if not auto_sub.empty else 0

        prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        rec  = TP / (TP + FN) if (TP + FN) > 0 else np.nan

        if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
            f1 = np.nan
        else:
            f1 = 2 * prec * rec / (prec + rec)

        coverage_mean   = float(np.nanmean(coverage_vals))  if coverage_vals else np.nan
        coverage_median = float(np.nanmedian(coverage_vals)) if coverage_vals else np.nan

        delay_valid = [d for d in delay_vals if not np.isnan(d)]
        delay_mean_hours   = float(np.mean(delay_valid))   if delay_valid else np.nan
        delay_median_hours = float(np.median(delay_valid)) if delay_valid else np.nan

        false_alarms_per_day = FP / total_days

        if not man_sub.empty:
            man_sub["manual_len_sec"] = (
                man_sub["manual_end"] - man_sub["manual_start"]
            ).dt.total_seconds()
            manual_len_total_sec = man_sub["manual_len_sec"].sum()
        else:
            manual_len_total_sec = 0.0

        if not auto_sub.empty:
            auto_sub["auto_len_sec"] = (
                auto_sub["seg_end"] - auto_sub["seg_start"]
            ).dt.total_seconds()
            auto_len_total_sec = auto_sub["auto_len_sec"].sum()
        else:
            auto_len_total_sec = 0.0

        overlap_total_sec = 0.0
        if manual_len_total_sec > 0 and auto_len_total_sec > 0:
            for _, mrow in man_sub.iterrows():
                v  = mrow["variable"]
                ms = mrow["manual_start"]
                me = mrow["manual_end"]

                rel_auto = auto_sub[auto_sub["variable"] == v]
                for _, arow in rel_auto.iterrows():
                    as_ = arow["seg_start"]
                    ae  = arow["seg_end"]
                    start = max(ms, as_)
                    end   = min(me, ae)
                    if end > start:
                        overlap_total_sec += (end - start).total_seconds()

        if auto_len_total_sec > 0:
            prec_time = overlap_total_sec / auto_len_total_sec
        else:
            prec_time = np.nan

        if manual_len_total_sec > 0:
            rec_time = overlap_total_sec / manual_len_total_sec
        else:
            rec_time = np.nan

        if np.isnan(prec_time) or np.isnan(rec_time) or (prec_time + rec_time) == 0:
            f1_time = np.nan
        else:
            f1_time = 2 * prec_time * rec_time / (prec_time + rec_time)

        extra_time_sec = max(auto_len_total_sec - overlap_total_sec, 0.0)
        extra_hours = extra_time_sec / 3600.0 if extra_time_sec > 0 else 0.0

        if auto_len_total_sec > 0:
            extra_ratio_auto = extra_time_sec / auto_len_total_sec
        else:
            extra_ratio_auto = np.nan

        results.append({
            "window": win,
            "strategy": strat,
            "metric": metric,
            "TP_episodes": TP,
            "FP_episodes": FP,
            "FN_episodes": FN,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "manual_total_hours": manual_len_total_sec / 3600.0 if manual_len_total_sec > 0 else 0.0,
            "auto_total_hours": auto_len_total_sec / 3600.0 if auto_len_total_sec > 0 else 0.0,
            "overlap_hours": overlap_total_sec / 3600.0 if overlap_total_sec > 0 else 0.0,
            "Precision_time": prec_time,
            "Recall_time": rec_time,
            "F1_time": f1_time,
            "coverage_mean": coverage_mean,
            "coverage_median": coverage_median,
            "delay_mean_hours": delay_mean_hours,
            "delay_median_hours": delay_median_hours,
            "false_alarms_per_day": false_alarms_per_day,
            "extra_hours": extra_hours,
            "extra_ratio_auto": extra_ratio_auto,
        })

        man_sub["window"]   = win
        man_sub["strategy"] = strat
        man_sub["metric"]   = metric

        auto_sub["window"]   = win
        auto_sub["strategy"] = strat
        auto_sub["metric"]   = metric

        marks_manual_all.append(man_sub)
        marks_auto_all.append(auto_sub)

    eval_df_out = pd.DataFrame(results)
    manual_marked_out = (
        pd.concat(marks_manual_all, ignore_index=True)
        if marks_manual_all else pd.DataFrame()
    )
    auto_marked_out = (
        pd.concat(marks_auto_all, ignore_index=True)
        if marks_auto_all else pd.DataFrame()
    )

    return eval_df_out, manual_marked_out, auto_marked_out


# Refinar episodios con 1H
def refine_episodes_with_1h(episodes_all: pd.DataFrame,
                            episodes_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta episodios de ventanas grandes usando episodios con ventana 1H
    - Si hay solapamiento con un episodio 1H, mueve el seg_start al mínimo seg_start de las 1H solapadas
    - De lo contrario, deja el episodio como está
    """
    eps_1h = episodes_1h.copy()
    eps_big = episodes_all.copy()

    if "window" in eps_big.columns:
        eps_big = eps_big[eps_big["window"] != "1H"].copy()

    if eps_1h.empty:
        return episodes_all

    refined_rows = []

    for _, row in eps_big.iterrows():
        var    = row["variable"]
        strat  = row.get("strategy", None)
        metric = row.get("metric", None)
        start_b = row["seg_start"]
        end_b   = row["seg_end"]

        cand = eps_1h[
            (eps_1h["variable"] == var) &
            (eps_1h["strategy"] == strat) &
            (eps_1h["metric"]   == metric)
        ]

        if cand.empty:
            row_ref = row.copy()
            row_ref["seg_start_refined"]  = row["seg_start"]
            row_ref["seg_length_refined"] = row.get("seg_length", row["seg_end"] - row["seg_start"])
            refined_rows.append(row_ref)
            continue

        mask_overlap = ~(
            (cand["seg_end"]   < start_b) |
            (cand["seg_start"] > end_b)
        )
        overlap_1h = cand[mask_overlap]

        if overlap_1h.empty:
            row_ref = row.copy()
            row_ref["seg_start_refined"]  = row["seg_start"]
            row_ref["seg_length_refined"] = row.get("seg_length", row["seg_end"] - row["seg_start"])
            refined_rows.append(row_ref)
            continue

        new_start = overlap_1h["seg_start"].min()

        row_ref = row.copy()
        row_ref["seg_start_refined"]  = new_start
        row_ref["seg_length_refined"] = row["seg_end"] - new_start
        refined_rows.append(row_ref)

    eps_big_refined = pd.DataFrame(refined_rows)
    combined = pd.concat([eps_big_refined, eps_1h], ignore_index=True)
    return combined
