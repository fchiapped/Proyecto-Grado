# Análisis de Diferencias Centradas - Guía de Uso

## 📖 Descripción

El análisis de diferencias centradas es un método de detección de valores atípicos implementado por tu equipo en `Modelos/DIFF_model.ipynb`. Este método:

1. **Calcula diferencias centradas**: Para cada punto, toma la diferencia mínima con sus vecinos anterior y posterior
2. **Análisis adaptativo por día**: Compara la desviación estándar diaria con la global
3. **Doble criterio de detección**:
   - Diferencia centrada > std_usado × multiplicador (default: 12)
   - Diferencia simple absoluta > umbral (default: 4)

## 🔧 Uso desde Pipeline Manager

### Ejemplo básico

```python
from pipeline_manager import PipelineManager

# Crear pipeline y cargar datos
pm = PipelineManager()
pm.load_data("df_procesados/df_planta_1.csv")

# Ejecutar análisis de diferencias
resultado = pm.run_diff_analysis(
    fecha_inicio='2025-06-01',
    fecha_fin='2025-08-09'
)

# Ver resultados
print(f"Columnas analizadas: {resultado['n_columnas_analizadas']}")
for col, datos in resultado['resultados'].items():
    print(f"{col}: {datos['n_outliers']} outliers detectados")
```

### Parámetros configurables

```python
resultado = pm.run_diff_analysis(
    columnas=['Conductividad DAF', 'pH DAF'],  # Específicas o None para todas
    fecha_inicio='2025-06-01',                 # Fecha inicial (YYYY-MM-DD)
    fecha_fin='2025-08-09',                    # Fecha final (YYYY-MM-DD)
    null_threshold=0.5,                        # 50% de datos faltantes = día insuficiente
    std_multiplier=12.0,                       # Multiplicador de std (más alto = menos sensible)
    diff_absolute_threshold=4.0                # Umbral de diferencia absoluta
)
```

## 📊 Estructura de resultados

```python
{
    'status': 'success',
    'message': 'Análisis de diferencias completado exitosamente',
    'n_columnas_analizadas': 15,
    'resultados': {
        'Conductividad DAF': {
            'resultados': DataFrame,  # Índice: datetime, Columnas: ['Datos', 'Etiqueta', 'Diff_centrada']
            'n_outliers': 16,
            'n_normales': 69512,
            'n_insuficientes': 0,
            'std_global': 0.1729,
            'parametros': {
                'null_threshold': 0.5,
                'std_multiplier': 12.0,
                'diff_absolute_threshold': 4.0
            }
        },
        # ... otras columnas
    }
}
```

### Etiquetas en el DataFrame resultante

- `"Outlier"`: Valor atípico detectado
- `"Normal"`: Valor dentro del rango esperado
- `"No hay data suf"`: Día con datos insuficientes (>50% nulos)
- `None`: Valor faltante (NaN)

## 🎯 Uso desde Auto Analyzer (menú interactivo)

Cuando ejecutas `python auto_analyzer.py`:

1. Menú principal → `[2] Configuración de análisis`
2. Activa `[4] Análisis de diferencias`
3. Volver → `[1] Analizar archivo`
4. Selecciona el archivo de planta

Los resultados se guardarán en:
- `resultados_analisis/planta_X_TIMESTAMP/5_reporte_diferencias.txt`: Resumen
- `resultados_analisis/planta_X_TIMESTAMP/5_diff_COLUMNA.csv`: Detalles por columna

## 🔬 Algoritmo interno

### 1. Cálculo de diferencia centrada

```python
def calcular_diff_centrada(serie):
    val_prev = serie.shift(1)   # Valor anterior
    val_sig = serie.shift(-1)   # Valor siguiente
    
    diff_prev = abs(val_prev - serie)
    diff_sig = abs(val_sig - serie)
    
    # Tomar el mínimo de las dos diferencias
    diff_centrada = min(diff_prev, diff_sig)
    return diff_centrada
```

### 2. Detección adaptativa por día

Para cada día:
```python
std_dia = diff_centrada[fecha].std()
std_global = diff_centrada.std()

# Usar el mayor de los dos
std_usado = max(std_dia, std_global)

# Marcar como outlier si:
es_outlier = (diff_centrada >= std_usado * 12) OR (diff_simple >= 4)
```

## ⚙️ Ajuste de parámetros

### `std_multiplier` (default: 12)
- **Más alto** (ej. 15): Menos sensible, detecta solo cambios muy bruscos
- **Más bajo** (ej. 8): Más sensible, detecta cambios moderados
- Recomendado: 10-15 para variables estables, 5-8 para variables volátiles

### `diff_absolute_threshold` (default: 4)
- Umbral de diferencia simple entre puntos consecutivos
- Útil para detectar saltos abruptos independientemente de la variabilidad
- Ajustar según la escala de tus variables

### `null_threshold` (default: 0.5)
- Porcentaje máximo de nulos permitidos por día
- 0.5 = si más del 50% del día tiene datos faltantes, marcar todo el día como insuficiente

## 📝 Notas importantes

1. **Requiere índice temporal**: El DataFrame debe tener columna `date_time`
2. **Análisis por día**: La detección se hace día por día, no sobre toda la serie
3. **Manejo de bordes**: Los primeros/últimos puntos usan el segundo vecino si el primero falta
4. **Compatibilidad**: Funciona con todas las columnas numéricas del DataFrame

## 🚀 Próximos pasos

Para extender el análisis:
- Modifica `Analisis/funciones_analisis.py::analizar_diferencias()` para ajustar la lógica
- Llama `pm.reload_analysis_functions()` para recargar sin reiniciar el programa
- Los cambios se reflejan automáticamente en `auto_analyzer.py`

## 📚 Referencia

Implementación original: `Modelos/DIFF_model.ipynb` (celdas 2-10)
Función wrapper: `Analisis/funciones_analisis.py::analizar_diferencias()`
Integración pipeline: `pipeline/pipeline_manager.py::run_diff_analysis()`
