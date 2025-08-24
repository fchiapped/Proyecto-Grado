import matplotlib.pyplot as plt  
import seaborn as sns 
import pandas as pd
import numpy as np

# Drift
from scipy.stats import ks_2samp, chi2_contingency
#--------------------------------------------------------------------------------------------------------#

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

# Graficar todas las variables numéricas como series de tiempo
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


# Función auxiliar para extraer límites de una variable desde el DataFrame de descripción
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

# Drift y tendencia

def detectar_drift_unificado(
    df,
    ref_inicio=None,
    ref_fin=None,
    test_inicio=None,
    test_fin=None,
    frac=0.2,
    bins=10,
    aplicar_psi=True
):

    df = df.copy()
    df = df.dropna(subset=["date_time"])
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    # --- Selección de periodos ---
    if ref_inicio and ref_fin and test_inicio and test_fin:
        ref = df[(df['date_time'] >= ref_inicio) & (df['date_time'] <= ref_fin)]
        test = df[(df['date_time'] >= test_inicio) & (df['date_time'] <= test_fin)]
    else:
        # partición automática
        n = len(df)
        corte = int(n * frac)
        ref = df.iloc[:corte]
        test = df.iloc[-corte:]

    resultados = []

    for col in df.columns:
        if col == "date_time":
            continue

        serie_ref = ref[col].dropna()
        serie_test = test[col].dropna()

        # --- Validación de datos suficientes ---
        if len(serie_ref) < 10 or len(serie_test) < 10:
            resultados.append({
                "variable": col,
                "tipo": str(df[col].dtype),
                "metodo": None,
                "stat": None,
                "pvalue": None,
                "psi": None,
                "drift_detectado": None,
                "detalle": "pocos datos"
            })
            continue

        # --- Numéricas ---
        if np.issubdtype(df[col].dtype, np.number):
            # KS
            stat, pval = ks_2samp(serie_ref, serie_test)
            drift = pval < 0.05
            resultados.append({
                "variable": col,
                "tipo": "numerica",
                "metodo": "KS",
                "stat": float(stat),
                "pvalue": float(pval),
                "psi": None,
                "drift_detectado": drift,
                "detalle": None
            })

            # PSI opcional
            if aplicar_psi:
                base_counts, bin_edges = np.histogram(serie_ref, bins=bins)
                comp_counts, _ = np.histogram(serie_test, bins=bin_edges)
                base_perc = base_counts / (base_counts.sum() + 1e-8)
                comp_perc = comp_counts / (comp_counts.sum() + 1e-8)
                psi = np.sum((base_perc - comp_perc) * np.log((base_perc + 1e-8) / (comp_perc + 1e-8)))
                resultados.append({
                    "variable": col,
                    "tipo": "numerica",
                    "metodo": "PSI",
                    "stat": None,
                    "pvalue": None,
                    "psi": float(psi),
                    "drift_detectado": psi > 0.2,
                    "detalle": None
                })

        else:
            # --- Categóricas ---
            cont = pd.crosstab(serie_ref, serie_test)
            if cont.shape[0] > 1 and cont.shape[1] > 1:
                chi2, pval, _, _ = chi2_contingency(cont)
                drift = pval < 0.05
                resultados.append({
                    "variable": col,
                    "tipo": "categorica",
                    "metodo": "Chi2",
                    "stat": float(chi2),
                    "pvalue": float(pval),
                    "psi": None,
                    "drift_detectado": drift,
                    "detalle": None
                })
            else:
                resultados.append({
                    "variable": col,
                    "tipo": "categorica",
                    "metodo": "Chi2",
                    "stat": None,
                    "pvalue": None,
                    "psi": None,
                    "drift_detectado": None,
                    "detalle": "sin variación suficiente"
                })

    return pd.DataFrame(resultados)