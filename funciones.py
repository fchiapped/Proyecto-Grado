import matplotlib.pyplot as plt  
import seaborn as sns 

def plot_temporal(df, columna: str, color: str, marker: str):

    plt.figure(figsize=(12, 6))
    plt.plot(df['date_time'], df[columna], color=color, marker=marker)

    plt.title(f'{columna} vs fecha')
    plt.xlabel('fecha')
    plt.ylabel(columna)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_avg_hora(df, columna: str, color: str):

    df['hour'] = df['date_time'].dt.hour
    avg_by_hour = df.groupby('hour')[columna].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='hour', y=columna, data=avg_by_hour, color=color)
    plt.title(f'Promedio de {columna} por Hora')
    plt.xlabel('Hora del Día')
    plt.ylabel(f'Promedio {columna}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_densidad(df, columna: str, color: str):
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[columna], color=color, fill=True, linewidth=2)
    plt.title(f'Densidad de {columna}')
    plt.xlabel(columna)
    plt.ylabel('Densidad')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_corr(df):
    corr = df.select_dtypes(include='number').corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

    plt.title('Mapa de Correlación entre Variables Numéricas')
    plt.show()

def heatmap_hour(df, columna: str, agg: str):
    d = df.copy()
    d['date_time'] = pd.to_datetime(d['date_time'], errors='coerce')
    d = d.dropna(subset=['date_time'])

    d['hour'] = d['date_time'].dt.hour
    d['dow_num'] = d['date_time'].dt.dayofweek 
    dow_labels = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']

    pivot = d.pivot_table(index='dow_num', columns='hour', values=columna, aggfunc=agg)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=False, cbar=True)  
    plt.title(f'{agg.capitalize()} de {columna} por Hora y Día de la semana')
    plt.xlabel('Hora del día')
    plt.ylabel('Día de la semana')
    plt.yticks(ticks=range(7), labels=dow_labels, rotation=0)
    plt.tight_layout()
    plt.show()