import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import changefinder


def AR_outlier_series(x, lag_ar=2, alpha=0.01, quantile=0.9995,
                      factor_olvido=0.01, lag_cambio=1, suavizado=5, change_quantile=0.999):
    """基于 notebook 中逻辑的序列版本：输入 numpy array，输出 dict of arrays"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= lag_ar:
        # insufficient data
        return {
            'outlier_score': np.full(n, np.nan),
            'change_score': np.full(n, np.nan),
            'label': np.array(['normal'] * n, dtype=object)
        }

    ar = AutoReg(x, lags=lag_ar, old_names=False)
    mod = ar.fit()
    pred = mod.predict(start=lag_ar, end=n - 1, dynamic=False)
    residuos = np.full(n, np.nan, dtype=float)
    residuos[lag_ar:] = x[lag_ar:] - pred

    serie_residuos = pd.Series(residuos)
    var_inicial = np.nanvar(serie_residuos.values)
    if not np.isfinite(var_inicial) or var_inicial <= 0:
        var_inicial = 1.0

    s2 = []
    s2_curr = var_inicial
    for r in serie_residuos.values:
        if np.isfinite(r):
            s2_curr = (1 - alpha) * s2_curr + alpha * (r ** 2)
        s2.append(max(s2_curr, 1e-8))
    s2 = np.array(s2, dtype=float)

    outlier_score = 0.5 * (np.log(2 * np.pi * s2) + (residuos ** 2) / s2)

    valid_out = np.isfinite(outlier_score)
    thr_out = np.quantile(outlier_score[valid_out], quantile) if valid_out.any() else np.inf
    is_out = outlier_score >= thr_out

    cf = changefinder.ChangeFinder(r=factor_olvido, order=lag_cambio, smooth=suavizado)
    score_cambio = np.array([cf.update(float(v)) for v in x], dtype=float)

    valid_ch = np.isfinite(score_cambio)
    thr_ch = np.quantile(score_cambio[valid_ch], change_quantile) if valid_ch.any() else np.inf
    is_ch = score_cambio >= thr_ch

    labels = np.full(n, 'normal', dtype=object)
    labels[is_out] = 'outlier'
    labels[is_ch] = 'change'

    return {
        'outlier_score': outlier_score,
        'change_score': score_cambio,
        'label': labels,
        'model': mod
    }


def detect_chunk(df_chunk, col, **kwargs):
    """对 df_chunk 的指定列进行检测，返回带 score/label 的 DataFrame 行。"""
    if df_chunk.empty:
        return pd.DataFrame()
    series = df_chunk[col].astype(float).values
    res = AR_outlier_series(series, **kwargs)
    out = df_chunk.copy().reset_index(drop=True)
    out['outlier_score'] = res['outlier_score']
    out['change_score'] = res['change_score']
    out['label'] = res['label']
    return out
