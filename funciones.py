import matplotlib.pyplot as plt  
import seaborn as sns 
import pandas as pd
import numpy as np

def plot_temporal(df, columna: str, color: str='blue', marker: str='o', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    d = df.copy()
    d['date_time'] = pd.to_datetime(d['date_time'], errors='coerce')
    ax.plot(d['date_time'], d[columna], color=color, marker=marker, markersize=3, linewidth=1)
    ax.set_title(f'{columna} vs fecha')
    ax.set_xlabel('fecha')
    ax.set_ylabel(columna)
    ax.grid(True)
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
                          agg_heatmap='mean'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    plot_temporal(df, columna, color_temporal, marker_temporal, ax=axes[0, 0])

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

