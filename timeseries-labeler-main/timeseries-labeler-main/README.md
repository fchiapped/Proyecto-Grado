# Etiquetador Avanzado de Series Temporales

Esta es una aplicación web interactiva construida con Dash y Plotly, diseñada para facilitar el etiquetado manual de puntos de datos en series temporales. Es una herramienta ideal para científicos de datos que necesitan crear datasets de entrenamiento para modelos de detección de anomalías, clasificación o imputación de datos.

## Funcionalidades Principales

-   **Carga de Datos Flexible**: Soporta archivos CSV en formato largo (long) y ancho (wide).
-   **Visualización Interactiva**: Gráficos dinámicos que permiten hacer zoom, paneo y selección de rangos de datos.
-   **Gestión de Categorías**: Permite crear y eliminar categorías de etiquetas sobre la marcha.
-   **Etiquetado Rápido**: Selecciona puntos directamente en el gráfico y asígnales una categoría con un solo clic.
-   **Eliminación de Etiquetas**: Corrige errores seleccionando puntos ya etiquetados y eliminando su etiqueta.
-   **Filtrado Visual**: Oculta o muestra categorías en el gráfico para facilitar la inspección.
-   **Tabla de Resumen**: Visualiza en tiempo real todos los puntos que han sido etiquetados.
-   **Exportación de Resultados**: Descarga los datos etiquetados en dos formatos útiles para análisis posteriores.

---

## Requisitos Previos

-   Python (se recomienda 3.8 o superior)
-   `pip` (el gestor de paquetes de Python)

---

## Instalación

1.  **Clonar el repositorio**
    ```bash
    git clone <URL-del-repositorio>
    cd <nombre-del-directorio>
    ```

2.  **(Recomendado) Crear un entorno virtual**
    ```bash
    python -m venv venv
    # En Windows
    .\venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```

3.  **Instalar las dependencias**
    Crea un archivo `requirements.txt` con el siguiente contenido:
    ```
    dash
    pandas
    plotly
    ```
    Luego, instálalo con pip:
    ```bash
    pip install -r requirements.txt
    ```

---

## Cómo Ejecutar la Aplicación

Una vez instaladas las dependencias, inicia la aplicación con el siguiente comando:

```bash
python etiquetador_series_avanzado.py
```

Abre tu navegador web y ve a la siguiente dirección: **http://127.0.0.1:8050**

---

## Guía de Uso (Flujo de Trabajo)

1.  **Paso 1: Cargar Datos**
    -   Arrastra y suelta tu archivo CSV en el área designada.
    -   La aplicación detectará automáticamente si el formato es largo o ancho.

2.  **Paso 2: Seleccionar Variable**
    -   Usa el menú desplegable para elegir la serie temporal que deseas visualizar y etiquetar.

3.  **Paso 3: Etiquetar Datos**
    -   En la barra de herramientas del gráfico, selecciona una herramienta de selección como **"Box Select"** (selección rectangular) o **"Lasso Select"** (selección libre).
    -   Dibuja una selección alrededor de los puntos que deseas etiquetar.
    -   Haz clic en el botón de la categoría correspondiente (ej. "Marcar como anómalo"). Los puntos seleccionados cambiarán de color en el gráfico y aparecerán en la tabla inferior.

4.  **Paso 4: Gestionar Categorías**
    -   Si necesitas una nueva etiqueta, escríbela en el campo "Nueva categoría" y haz clic en "Añadir".
    -   Para eliminar una categoría, selecciónala del menú desplegable y haz clic en "Eliminar".

5.  **Paso 5: Eliminar Etiquetas**
    -   Si cometiste un error, selecciona los puntos ya etiquetados en el gráfico (los marcadores de colores).
    -   Haz clic en el botón **"Eliminar etiquetas seleccionadas"**.

6.  **Paso 6: Exportar Resultados**
    -   **Dataset Etiquetado (long)**: Descarga un CSV (`data_etiquetada_long.csv`) que contiene *únicamente* los puntos que fueron etiquetados, con una columna adicional `etiqueta`.
    -   **Dataset (wide imputado)**: Descarga un CSV (`data_etiquetada_wide.csv`) con la estructura del archivo original en formato ancho, pero donde el valor de los puntos etiquetados ha sido reemplazado por un valor nulo. Ideal para tareas de imputación.

---

## Formato de los Datos de Entrada

La columna de tiempo es obligatoria y debe llamarse `timestamp`.

#### Formato Largo (Recomendado)

| timestamp                   | variable                    | valor |
| --------------------------- | --------------------------- | ----- |
| `2025-08-23T10:00:00-04:00` | `NÍVEL DO TANQUE "A" (%)`   | 95.32 |
| `2025-08-23T10:01:00-04:00` | `NÍVEL DO TANQUE "A" (%)`   | 95.33 |
| `2025-08-23T10:00:00-04:00` | `OD01 - TEMPERATURA`        | 22.5  |

#### Formato Ancho

| timestamp                   | NÍVEL DO TANQUE "A" (%) | OD01 - TEMPERATURA |
| --------------------------- | ----------------------- | ------------------ |
| `2025-08-23T10:00:00-04:00` | 95.32                   | 22.5               |
| `2025-08-23T10:01:00-04:00` | 95.33                   | 22.6               |
