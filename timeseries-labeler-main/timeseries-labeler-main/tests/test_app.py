import pytest
from pathlib import Path
from etiquetador_series_avanzado import app, manage_categories, DEFAULT_CATEGORIES


@pytest.fixture
def dash_test_app():
    app.config['TESTING'] = True
    return app


def test_upload_file(dash_duo, dash_test_app):
    dash_duo.start_server(dash_test_app)

    assert dash_duo.find_element("h2").text == (
        "Etiquetador Avanzado de Series Temporales"
    )

    sample_file = Path(__file__).parent / "sample.csv"
    upload = dash_duo.find_element("#upload-data input[type='file']")
    upload.send_keys(str(sample_file))

    dash_duo.wait_for_text_to_equal(
        "#upload-status",
        f"Archivo '{sample_file.name}' cargado correctamente."
    )


def test_add_category(dash_duo, dash_test_app):
    dash_duo.start_server(dash_test_app)

    input_box = dash_duo.find_element("#new-category")
    input_box.send_keys("nueva_categoria")
    dash_duo.find_element("#add-category").click()

    dash_duo.wait_for_element("#category-buttons")
    buttons = dash_duo.find_elements("#category-buttons button")
    assert any(b.text == "Marcar como nueva_categoria" for b in buttons)


def test_add_new_category():
    """
    Verifica que se puede añadir una nueva categoría a la lista existente.
    """
    # 1. Preparación (simulando los inputs y states del callback)
    add_clicks = 1
    delete_clicks = [0] * len(DEFAULT_CATEGORIES)
    new_category_value = "nueva_etiqueta"
    current_categories = DEFAULT_CATEGORIES.copy()

    # Simular el contexto de Dash para saber qué botón se presionó
    class MockContext:
        triggered_id = 'add-category'

    # Monkeypatching ctx para el test
    import dash
    dash.callback_context = MockContext()

    # 2. Ejecución (llamar a la función directamente)
    botones, opciones, updated_categories = manage_categories(
        add_clicks, delete_clicks, new_category_value, current_categories
    )

    # 3. Verificación (comprobar que el resultado es el esperado)
    expected_categories = DEFAULT_CATEGORIES + ["nueva_etiqueta"]
    assert updated_categories == expected_categories
    assert len(opciones) == len(expected_categories)
    assert any(opt['value'] == "nueva_etiqueta" for opt in opciones)


def test_delete_existing_category():
    """
    Verifica que se puede eliminar una categoría existente.
    """
    # 1. Preparación
    add_clicks = 0
    # Simular que se hizo clic en el botón de eliminar de la categoría "outlier"
    delete_clicks = [0, 1]  # [anómalo, outlier]
    new_category_value = ""
    current_categories = DEFAULT_CATEGORIES.copy()

    class MockContext:
        triggered_id = {'type': 'delete-category-btn', 'index': 'outlier'}

    import dash
    dash.callback_context = MockContext()

    # 2. Ejecución
    botones, opciones, updated_categories = manage_categories(
        add_clicks, delete_clicks, new_category_value, current_categories
    )

    # 3. Verificación
    assert "outlier" not in updated_categories
    assert updated_categories == ["anómalo"]
    assert len(opciones) == 1


def test_do_nothing_if_no_trigger():
    """
    Verifica que la función no hace nada si no hay un trigger válido.
    """
    class MockContext:
        triggered = []  # No hay trigger

    import dash
    dash.callback_context = MockContext()

    with pytest.raises(dash.exceptions.PreventUpdate):
        manage_categories(0, [0], "", DEFAULT_CATEGORIES)
