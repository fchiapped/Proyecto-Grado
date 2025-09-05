import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table, ALL
import base64
import io
import dash

app = Dash(__name__)
app.title = "Etiquetador Avanzado de Series Temporales"

DEFAULT_CATEGORIES = ["anómalo", "outlier"]


def render_category_buttons(categorias):
    return [
        html.Div([
            html.Button(
                f"Marcar como {c}",
                id={'type': 'label-btn', 'index': c},
                n_clicks=0,
                style={"marginRight": "5px"}
            ),
            html.Button(
                "Eliminar",
                id={'type': 'delete-category-btn', 'index': c},
                n_clicks=0,
                style={"marginLeft": "5px", "color": "red"}
            )
        ]) for c in categorias
    ]

# Layout
app.layout = html.Div([
    html.H2("Etiquetador Avanzado de Series Temporales"),

    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arrastra un archivo CSV aquí o haz click para cargar']),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),

    dcc.Dropdown(
        id='var-selector',
        placeholder="Selecciona una variable",
        style={"margin": "10px"}
    ),

    dcc.Graph(
        id='time-series',
        config={'scrollZoom': True}
    ),

    html.Div([
        dcc.Input(
            id='new-category',
            type='text',
            placeholder='Nueva categoría',
            style={"marginRight": "10px"}
        ),
        html.Button(
            "Agregar categoría",
            id='add-category',
            n_clicks=0
        )
    ], style={"margin": "10px"}),

    html.Div(
        id='category-buttons',
        children=render_category_buttons(DEFAULT_CATEGORIES),
        style={"margin": "10px"}
    ),

    html.Div([
        html.Button(
            "Eliminar etiquetas seleccionadas",
            id='remove-labels',
            n_clicks=0
        ),
        html.Button(
            "Descargar dataset etiquetado (long)",
            id='download-long',
            n_clicks=0
        ),
        html.Button(
            "Descargar dataset (wide imputado)",
            id='download-wide',
            n_clicks=0
        )
    ], style={"margin": "10px"}),

    html.Label("Filtrar por categoría:", style={"marginTop": "10px"}),
    dcc.Checklist(
        id='filter-categories',
        options=[{'label': c, 'value': c} for c in DEFAULT_CATEGORIES],
        value=[]
    ),

    dash_table.DataTable(
        id='labeled-table',
        columns=[
            {"name": i, "id": i} for i in [
                "timestamp", "variable", "valor", "etiqueta"
            ]
        ],
        data=[],
        style_table={"overflowX": "auto"},
        page_size=10
    ),
    html.Div(
        id="feedback",
        style={"color": "red", "margin": "10px"}
    ),

    # Agregar un componente para mostrar el estado de la carga
    html.Div(id='upload-status', style={"color": "blue", "margin": "10px"}),
    dcc.Store(id='store-original'),
    dcc.Store(id='store-labeled'),
    dcc.Store(id='store-categorias', data=DEFAULT_CATEGORIES),
])

# Cargar y detectar formato
@app.callback(
    [Output('store-original', 'data'),
     Output('store-labeled', 'data'),
     Output('var-selector', 'options'),
     Output('var-selector', 'value')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_data(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate
        
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    if 'timestamp' not in df.columns:
        raise dash.exceptions.PreventUpdate

    # CORRECCIÓN: Estandarizar todo a UTC al cargar
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    if df['timestamp'].dt.tz is not None:
        # Si ya tiene zona horaria, convertir a UTC
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
    else:
        # Si es "naive", asumimos que es UTC para evitar ambigüedad
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

    # Detectar formato wide o long
    if 'variable' in df.columns and 'valor' in df.columns:
        df_long = df
    else:
        # Limpiar nombres de columnas para evitar problemas
        df.columns = [col.strip().replace('"', '') for col in df.columns]
        df_long = df.melt(id_vars='timestamp', var_name='variable', value_name='valor')

    opciones = [{'label': v, 'value': v} for v in df_long['variable'].unique()]
    valor_default = opciones[0]['value'] if opciones else None
    return df_long.to_dict('records'), None, opciones, valor_default

# Unificar lógica de agregar/eliminar categorías
@app.callback(
    [Output('category-buttons', 'children'),
     Output('filter-categories', 'options'),
     Output('store-categorias', 'data')],
    [Input('add-category', 'n_clicks'),
     Input({'type': 'delete-category-btn', 'index': ALL}, 'n_clicks')],
    [State('new-category', 'value'),
     State('store-categorias', 'data')],
    prevent_initial_call=True
)
def manage_categories(add_clicks, delete_clicks, nueva, categorias):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered_id
    categorias = categorias or DEFAULT_CATEGORIES.copy()

    if triggered_id == 'add-category' and nueva and nueva not in categorias:
        categorias.append(nueva)
    elif isinstance(triggered_id, dict) and triggered_id.get('type') == 'delete-category-btn':
        categorias = [c for c in categorias if c != triggered_id.get('index')]

    botones = render_category_buttons(categorias)
    opciones = [{'label': c, 'value': c} for c in categorias]
    return botones, opciones, categorias

# Mostrar gráfica
@app.callback(
    Output('time-series', 'figure'),
    [Input('var-selector', 'value'),
     Input('filter-categories', 'value'),
     Input('store-labeled', 'data'),
     Input('store-original', 'data'),
     State('store-categorias', 'data')]
)
def update_graph(variable, filtros, labeled_data, original_data, categorias):
    if not original_data or variable is None:
        return {}

    df_original = pd.DataFrame(original_data)
    df_v = df_original[df_original['variable'] == variable]
    fig = px.line(
        df_v, x='timestamp', y='valor', title=f'Serie: {variable}'
    )
    fig.update_traces(mode='lines+markers', name="serie")

    if labeled_data:
        df_labeled_local = pd.DataFrame(labeled_data)
        df_etiquetas = df_labeled_local[
            df_labeled_local['variable'] == variable
        ]
        if filtros:
            df_etiquetas = df_etiquetas[
                ~df_etiquetas['etiqueta'].isin(filtros)
            ]

        # CORRECCIÓN 1: Omitir el primer color para que no coincida con la serie principal
        color_sequence = px.colors.qualitative.Plotly[1:]
        color_map = {
            cat: color_sequence[i % len(color_sequence)]
            for i, cat in enumerate(categorias or [])
        }

        for cat in df_etiquetas['etiqueta'].unique():
            df_cat = df_etiquetas[df_etiquetas['etiqueta'] == cat]
            fig.add_scattergl(
                x=df_cat['timestamp'], y=df_cat['valor'],
                mode='markers', name=cat,
                marker=dict(size=10, color=color_map.get(cat)), legendgroup=cat
            )
    return fig


# Callback combinado para etiquetar y eliminar puntos
@app.callback(
    [Output('store-labeled', 'data', allow_duplicate=True),
     Output('labeled-table', 'data')],
    [Input({'type': 'label-btn', 'index': ALL}, 'n_clicks'),
     Input('remove-labels', 'n_clicks')],
    [State('time-series', 'selectedData'),
     State('var-selector', 'value'),
     State('store-labeled', 'data')],
    prevent_initial_call=True
)
def actualizar_labeled_table(
    n_clicks_label, n_clicks_remove, selected, variable, labeled_data
):
    df_labeled_local = (
        pd.DataFrame(labeled_data) if labeled_data else pd.DataFrame(
            columns=['timestamp', 'variable', 'valor', 'etiqueta']
        )
    )

    # Si el dataframe no está vacío, convertir la columna timestamp a datetime UTC para trabajar con ella
    if not df_labeled_local.empty:
        df_labeled_local['timestamp'] = pd.to_datetime(df_labeled_local['timestamp'])

    triggered = ctx.triggered_id

    # Si se presionó un botón de etiqueta
    if isinstance(triggered, dict) and triggered.get('type') == 'label-btn':
        if not any(n_clicks_label) or selected is None or variable is None:
            raise dash.exceptions.PreventUpdate
        categoria = triggered['index']
        nuevos = []
        for punto in selected['points']:
            # Convertir a datetime y localizar a UTC inmediatamente
            ts_utc = pd.to_datetime(punto['x']).tz_localize('UTC')
            nuevos.append({
                'timestamp': ts_utc,
                'variable': variable,
                'valor': punto['y'],
                'etiqueta': categoria
            })
        if not nuevos:
            raise dash.exceptions.PreventUpdate

        nuevos_df = pd.DataFrame(nuevos)
        
        if df_labeled_local.empty:
            df_labeled_local = nuevos_df
        else:
            df_labeled_local = pd.concat([df_labeled_local, nuevos_df])

        df_labeled_local = df_labeled_local.drop_duplicates(
            subset=['timestamp', 'variable', 'valor'], keep='last'
        )
        
    # Si se presionó el botón de eliminar
    elif triggered == 'remove-labels':
        if not n_clicks_remove or selected is None or df_labeled_local.empty:
            raise dash.exceptions.PreventUpdate
        
        puntos_a_eliminar = [
            (p['x'], p['y']) for p in selected.get('points', [])
        ]
        
        if not puntos_a_eliminar:
            raise dash.exceptions.PreventUpdate

        indices_a_eliminar = []
        for ts, val in puntos_a_eliminar:
            # Convertir el timestamp seleccionado (que es naive) a UTC para una comparación precisa
            ts_utc = pd.to_datetime(ts).tz_localize('UTC')
            idx = df_labeled_local[
                (df_labeled_local['timestamp'] == ts_utc) &
                (df_labeled_local['variable'] == variable) &
                (df_labeled_local['valor'] == val)
            ].index
            indices_a_eliminar.extend(idx)

        df_labeled_local = df_labeled_local.drop(indices_a_eliminar)
        
    else:
        # Si no hay trigger válido, no hacer nada
        raise dash.exceptions.PreventUpdate

    # CORRECCIÓN FINAL: Convertir a string UTC solo al final, antes de guardar
    if not df_labeled_local.empty:
        df_labeled_local['timestamp'] = df_labeled_local['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    records = df_labeled_local.to_dict('records')
    return records, records


# Callback unificado para descargas
@app.callback(
    Output("feedback", "children"),
    [Input("download-long", "n_clicks"),
     Input("download-wide", "n_clicks")],
    [State('store-original', 'data'),
     State('store-labeled', 'data')],
    prevent_initial_call=True
)
def descargar_archivos(n_long, n_wide, original_data, labeled_data):
    triggered_id = ctx.triggered_id
    
    if triggered_id == 'download-long':
        if not labeled_data:
            return "No hay etiquetas para exportar."
        df_labeled_local = pd.DataFrame(labeled_data)
        # No se necesita conversión, ya está en formato string UTC
        df_labeled_local.to_csv("data_etiquetada_long.csv", index=False)
        return "Archivo guardado como data_etiquetada_long.csv."

    elif triggered_id == 'download-wide':
        if not original_data:
            return "No hay datos cargados."
        
        df_comb = pd.DataFrame(original_data)
        df_comb['timestamp'] = pd.to_datetime(df_comb['timestamp'])

        if not labeled_data:
            df_wide = df_comb.pivot(
                index="timestamp", columns="variable", values="valor"
            ).reset_index()
            df_wide.to_csv("data_etiquetada_wide.csv", index=False)
            return "Archivo guardado como data_etiquetada_wide.csv."

        df_labeled_local = pd.DataFrame(labeled_data)
        df_labeled_local['timestamp'] = pd.to_datetime(df_labeled_local['timestamp'])
        df_labeled_local['valor'] = pd.to_numeric(df_labeled_local['valor'], errors='coerce')

        merged = df_comb.merge(
            df_labeled_local[['timestamp', 'variable', 'valor', 'etiqueta']], 
            how="left", 
            on=["timestamp", "variable", "valor"]
        )
        merged.loc[merged["etiqueta"].notna(), "valor"] = None
        df_wide = merged.pivot(
            index="timestamp", columns="variable", values="valor"
        ).reset_index()
        df_wide.to_csv("data_etiquetada_wide.csv", index=False)
        return "Archivo guardado como data_etiquetada_wide.csv."
    
    return dash.no_update


@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def mostrar_estado_carga(contents, filename):
    if contents is None:
        return "No se ha cargado ningún archivo."
    return f"Archivo '{filename}' cargado correctamente."


if __name__ == "__main__":
    app.run(debug=True, port=8050)
