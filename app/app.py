
# -------------------------------------------------------------
# Interfaz de Streamlit 
# -------------------------------------------------------------

# Compatibilidad y tipado estático
from __future__ import annotations
from typing import Dict, Any, Tuple

# Análisis y manipulación de datos
import pandas as pd
import numpy as np

# Interfaz web interactiva con Streamlit
import streamlit as st

# Utilidades varias: manejo de JSON, mapas y normalización de texto
import json
import pydeck as pdk
import unicodedata

# ---------------------- CONFIGURACIÓN BÁSICA ----------------------

# Configuración de la página principal en Streamlit
st.set_page_config(page_title="Vivienda Madrid – Análisis por distrito", page_icon="🏠", layout="wide")

# Título principal de la aplicación
st.title("Análisis por distrito y factores determinantes del precio – Madrid")

# Texto introductorio breve bajo el título
st.caption("Explora datos inmobiliarios, visualiza KPIs, mapas y gráficos comparativos por distrito.")

# ---------------------- CARGA DE DATOS ----------------------------

# Ruta por defecto al archivo CSV de viviendas
DEFAULT_PATH = "data/final/viviendas_2011_2024.csv"

# Función con caché para cargar el CSV como DataFrame
@st.cache_data(show_spinner=True)
def load_csv_fixed(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8")

# Intento de carga de datos y verificación de contenido
try:
    df = load_csv_fixed(DEFAULT_PATH)
    if df.empty:
        st.warning("El CSV se cargó pero está vacío.")
        st.stop()
    st.success(f"✓ Datos cargados: {df.shape[0]:,} registros")

# Manejo de errores en caso de fallo al cargar el CSV
except Exception as e:
    st.error(f"No se pudo cargar el CSV desde la ruta fija. Detalle: {type(e).__name__}: {e}")
    st.stop()

df['Distrito'].unique()

# ---------------------- UTILIDADES -------------------------------------------

# Mapeo de nombres del CSV a nombres normalizados para el GeoJSON (los demás distritos ya vienen correctamente)
DISTRITO_MAPPING = {
    'FUENCARRALELPARDO': 'FUENCARRAL EL PARDO', 'MONCLOAARAVACA': 'MONCLOA ARAVACA',
    'PUENTEDEVALLECAS': 'PUENTE DE VALLECAS', 'CIUDADLINEAL': 'CIUDAD LINEAL',
    'VILLADEVALLECAS': 'VILLA DE VALLECAS', 'SANBLASCANILLEJAS': 'SAN BLAS CANILLEJAS',
}

# Normaliza texto para emparejar nombres (quita tildes, símbolos, mayúsculas, espacios)
def _norm(x: str) -> str:

    # Devuelve cadena vacía si es None y normaliza el valor a texto en mayúsculas sin espacios
    if x is None:
        return ""
    x = str(x).strip().upper()

    # Aplicar mapeo específico si el nombre está en nuestro diccionario
    x = DISTRITO_MAPPING.get(x, x)

    # Normalización Unicode (elimina tildes pero preserva Ñ)
    x = x.replace("Ñ", "###TEMP_N###")
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    x = x.replace("###TEMP_N###", "Ñ")

    # Reemplazar símbolos por espacios
    for a in ["-", "_", "/", ",", ".", "'", "`", "(", ")", "[", "]", "{", "}"]:
        x = x.replace(a, " ")

    # Limpiar espacios múltiples
    x = " ".join(x.split())

    # Devuelve el texto ya normalizado y listo para comparaciones
    return x

# ---------------------- COLUMNAS DE FILTRO ---------------------

# Definición de columnas críticas y opcionales para el análisis
critical_cols = ["Distrito", "Precio_ajustado"]
optional_cols = ["Ano", "Tipo_vivienda", "Operacion", "Habitaciones", "Tamano"]

# Verificar que todas las columnas críticas existen en el DataFrame
missing_critical = [c for c in critical_cols if c not in df.columns]
if missing_critical:
    st.error(f"❌ Columnas críticas faltantes: {', '.join(missing_critical)}")
    st.stop()

# Verificar qué columnas opcionales no están disponibles
missing_optional = [c for c in optional_cols if c not in df.columns]
if missing_optional:
    st.info(f"ℹ️ Filtros no disponibles: {', '.join(missing_optional)}")

# Construcción de la lista final de columnas a usar (críticas + opcionales disponibles)
requested_cols = critical_cols + [c for c in optional_cols if c in df.columns]

# ---------------------- TIPOS Y CASTS SUAVES ----------------------

# Definir todas las conversiones numéricas
NUMERIC_COLUMNS = {

    # Críticas para el funcionamiento
    "Ano": "Int64",
    "Precio_ajustado": "float",

    # Características de la vivienda
    "Habitaciones": "Int64",
    "Banos": "Int64",
    "Tamano": "float",
    "Ano_construccion": "Int64",
    "Planta": "Int64",

    # Datos del distrito
    "Superficie_distrito_ha": "float",
    "Densidad_poblacion": "float",
    "Renta_neta_hogar": "float",
    "Apartamentos_turisticos": "Int64"
}

# ! Aplicar conversiones y acumular warnings

# Lista para registrar advertencias de conversión numérica
conversion_warnings = []

# Iterar por cada columna numérica definida en NUMERIC_COLUMNS
for col, dtype in NUMERIC_COLUMNS.items():

    # Verificar si la columna existe en el DataFrame
    if col in df.columns:

        # Contar valores nulos antes de la conversión
        before = df[col].isna().sum()

        try:
            # Intentar convertir la columna a numérica con control de errore s
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if dtype == "Int64":
                df[col] = df[col].astype("Int64")

            # Calcular cuántos valores se perdieron tras la conversión
            lost = df[col].isna().sum() - before
            if lost > 0:
                conversion_warnings.append(f"{col}: {lost:,} valores no válidos")

        except:

            # Ignorar errores de conversión si ocurren
            continue

# Mostrar resumen de conversiones si hubo problemas
if conversion_warnings:
    with st.expander("⚠️ Advertencias de conversión de datos", expanded=False):
        for warning in conversion_warnings:
            st.text(f"• {warning}")

# ---------------------- GENERADOR DE FILTROS ----------------------
def build_filters_improved(_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    """
    Genera filtros interactivos manteniendo todas las opciones visibles
    y aplicando los filtros de forma conjunta al final
    """

    # Diccionario para almacenar el estado de los filtros aplicados
    filters_state: Dict[str, Any] = {}

    # Configuración de la barra lateral de Streamlit
    with st.sidebar:

        # Título de la sección de filtros
        st.subheader("🔍 Filtros")

        # Botón para resetear todos los filtros y recargar la app
        if st.button("🔄 Resetear filtros"):
            st.rerun()

        # Mostrar número total de registros antes de aplicar filtros
        st.metric("Total registros originales", f"{len(_df):,}")
        st.divider()

    # Copias del DataFrame original: una para opciones y otra para filtrado
    df_original = _df.copy()
    df_filtered = _df.copy()

    # Lista para acumular las condiciones de filtrado activas
    conditions = []

    # ? 1) Año (multiselect con formato mejorado)

    # Verificar que la columna Año existe y tiene valores no nulos
    if "Ano" in df_original.columns and df_original["Ano"].notna().any():

        # Obtener lista de años únicos como enteros y ordenados
        years = sorted(df_original["Ano"].dropna().astype(int).unique().tolist())

        # Por defecto, mostrar últimos 5 años si hay más de cinco
        default_years = years[-10:] if len(years) > 10 else years

        # Crear multiselect en la barra lateral para elegir años
        sel_years = st.sidebar.multiselect("📅 Año", options=years,default=default_years, help="Selecciona uno o varios años")

        # Guardar selección de años en el estado de filtros
        filters_state["Ano"] = sel_years

        # Aplicar condición de filtrado si se seleccionaron años
        if sel_years:
            conditions.append(df_filtered["Ano"].isin(sel_years))

    # ? 2) Distrito (multiselect con búsqueda)

    # Verificar que la columna Distrito existe y contiene datos válidos
    if "Distrito" in df_original.columns and df_original["Distrito"].notna().any():

        # Obtener lista de distritos únicos como texto y ordenados alfabéticamente
        distritos = sorted(df_original["Distrito"].dropna().astype(str).unique().tolist())

        # Seleccionar todos por defecto si hay pocos, ninguno si son muchos
        default_dist = distritos if len(distritos) <= 10 else []

        # Crear multiselect en la barra lateral para elegir distritos con búsqueda
        sel_dist = st.sidebar.multiselect(
            "📍 Distrito", options=distritos, default=default_dist, placeholder="Selecciona distritos...", help="Puedes buscar escribiendo"
        )

        # Guardar selección de distritos en el estado de filtros
        filters_state["Distrito"] = sel_dist

        # Aplicar condición de filtrado si se seleccionaron distritos
        if sel_dist:
            conditions.append(df_filtered["Distrito"].isin(sel_dist))

    # --- Precio_ajustado (slider con estado controlado) ---
    if "Precio_ajustado" in df_original.columns and df_original["Precio_ajustado"].notna().any():
        serie = pd.to_numeric(df_original["Precio_ajustado"], errors="coerce").dropna()
        pmin, pmax = float(serie.min()), float(serie.max())

        # Inicializa estado la primera vez o cuando cambie el rango base
        if "precio_range" not in st.session_state:
            st.session_state["precio_range"] = (pmin, pmax)
        else:
            # Si el estado guardado se sale de los nuevos límites, lo reencuadramos
            lo, hi = st.session_state["precio_range"]
            lo = max(pmin, float(lo))
            hi = min(pmax, float(hi))
            if lo >= hi:  # si quedó inválido
                lo, hi = pmin, pmax
            st.session_state["precio_range"] = (lo, hi)

        # Paso dinámico
        span = pmax - pmin
        if span > 100_000:
            step = 5_000.0
        elif span > 10_000:
            step = 1_000.0
        else:
            step = 100.0

        # Slider ligado al estado
        st.session_state["precio_range"] = st.sidebar.slider(
            "💰 Precio ajustado (€)",
            min_value=pmin,
            max_value=pmax,
            value=st.session_state["precio_range"],  # 👈 siempre usa el estado
            step=step,
            format="%d €",
            help=f"Rango total: {pmin:,.0f}€ - {pmax:,.0f}€",
            key="precio_slider",
        )

        # Aplica el filtro con el rango actual del estado
        p0, p1 = st.session_state["precio_range"]
        filters_state["Precio_ajustado"] = (p0, p1)
        conditions.append((df_filtered["Precio_ajustado"] >= p0) & (df_filtered["Precio_ajustado"] <= p1))
    else:
        st.sidebar.info("No se encontró 'Precio_ajustado' o no tiene datos válidos.")



    # ? 4) Tipo_vivienda (radio buttons si son pocos, multiselect si son muchos)

    # Verificar que la columna Tipo_vivienda existe y contiene datos válidos
    if "Tipo_vivienda" in df_original.columns and df_original["Tipo_vivienda"].notna().any():

        # Obtener lista de tipos de vivienda únicos como texto y ordenados
        tipos = sorted(df_original["Tipo_vivienda"].dropna().astype(str).unique().tolist())

        # Si hay pocas opciones, usar radio buttons para selección única
        if len(tipos) <= 3:

            # Radio buttons para elegir un tipo de vivienda o todos
            sel_tipo = st.sidebar.radio("🏠 Tipo de vivienda", options=["Todos"] + tipos, index=0)

            # Aplicar filtro solo si no se seleccionó "Todos"
            if sel_tipo != "Todos":
                filters_state["Tipo_vivienda"] = [sel_tipo]
                conditions.append(df_filtered["Tipo_vivienda"] == sel_tipo)
            else:
                filters_state["Tipo_vivienda"] = tipos

        # Si hay muchas opciones, usar multiselect para selección múltiple
        else:

            # Multiselect en barra lateral para elegir varios tipos de vivienda
            sel_tipos = st.sidebar.multiselect("🏠 Tipo de vivienda", options=tipos, default=tipos)

            # Guardar selección en el estado de filtros
            filters_state["Tipo_vivienda"] = sel_tipos

            # Aplicar condición de filtrado si hay tipos seleccionados
            if sel_tipos:
                conditions.append(df_filtered["Tipo_vivienda"].isin(sel_tipos))

    # ? 5) Operación selección única

    if "Operacion" in df_original.columns and df_original["Operacion"].notna().any():
        ops = sorted(df_original["Operacion"].dropna().astype(str).unique().tolist())

        if ops:  # hay opciones disponibles
            sel_op = st.sidebar.selectbox("💼 Operación (opción única)", options=ops, index=0)
            filters_state["Operacion"] = sel_op
            conditions.append(df_filtered["Operacion"] == sel_op)
        else:
            st.sidebar.info("No hay valores válidos en 'Operacion'.")
    else:
        st.sidebar.info("No se encontró la columna 'Operacion' o no tiene datos válidos.")


    # Aplicar TODAS las condiciones de una vez
    if conditions:

        # Inicializar condición final con la primera de la lista
        final_condition = conditions[0]

        # Combinar el resto de condiciones con AND lógico
        for condition in conditions[1:]:
            final_condition = final_condition & condition

        # Filtrar el DataFrame con la condición final acumulada
        df_filtered = df_filtered[final_condition]

    # Mostrar contador de registros DESPUÉS de filtrar
    with st.sidebar:

        # Separador visual en la barra lateral
        st.divider()

        # Dividir en dos columnas para mostrar métricas
        col1, col2 = st.columns(2)

        # Métrica con número total de registros filtrados
        with col1:
            st.metric("Registros filtrados", f"{len(df_filtered):,}")

        # Métrica con porcentaje respecto al total original
        with col2:
            pct = (len(df_filtered) / len(_df) * 100) if len(_df) > 0 else 0
            st.metric("Porcentaje", f"{pct:.1f}%")

        # Advertencia si los filtros dejan muy pocos resultados
        if 0 < len(df_filtered) < 10:
            st.warning("⚠️ Pocos registros con estos filtros")
        elif len(df_filtered) == 0:
            st.error("❌ Sin resultados para estos filtros")

    # Devolver el DataFrame filtrado y el estado de filtros aplicados
    return df_filtered, filters_state


# Ejecutar la función de construcción de filtros y obtener resultados
filtered_df, state = build_filters_improved(df)

# Verificar si el DataFrame filtrado quedó vacío y detener la app si es así
if filtered_df.empty:
    st.error("No hay datos que coincidan con los filtros seleccionados. Por favor, ajusta los criterios.")
    st.stop()

# ---------------------- MAPA POR DISTRITO -------------------------------

GEO_PATH = "app/madrid_distritos_geo.json"

# ---------------------- KPIs PRINCIPALES ----------------------
st.subheader("📊 Métricas Principales")

# Calcular KPIs solo si hay datos filtrados
if not filtered_df.empty and "Precio_ajustado" in filtered_df.columns:
    # Crear columnas para KPIs
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    # Calcular estadísticas
    precio_serie = filtered_df["Precio_ajustado"].dropna()

    with kpi1:
        mediana_global = precio_serie.median()
        st.metric(
            "Mediana General",
            f"{mediana_global:,.0f}€" if pd.notna(mediana_global) else "N/D",
            help="Mediana del precio ajustado en todos los distritos filtrados"
        )

    with kpi2:
        media_global = precio_serie.mean()
        delta_media = ((media_global - mediana_global) / mediana_global * 100) if pd.notna(
            mediana_global) and mediana_global != 0 else 0
        st.metric(
            "Media General",
            f"{media_global:,.0f}€" if pd.notna(media_global) else "N/D",
            f"{delta_media:+.1f}% vs mediana",
            help="Media del precio ajustado"
        )

    with kpi3:
        precio_min = precio_serie.min()
        st.metric(
            "Precio Mínimo",
            f"{precio_min:,.0f}€" if pd.notna(precio_min) else "N/D",
            help="Precio más bajo encontrado"
        )

    with kpi4:
        precio_max = precio_serie.max()
        st.metric(
            "Precio Máximo",
            f"{precio_max:,.0f}€" if pd.notna(precio_max) else "N/D",
            help="Precio más alto encontrado"
        )

    with kpi5:
        num_distritos = filtered_df["Distrito"].nunique() if "Distrito" in filtered_df.columns else 0
        st.metric(
            "Distritos",
            f"{num_distritos}/21",
            help="Distritos con datos tras aplicar filtros"
        )

    st.divider()

# ---------------------- MAPA INTERACTIVO MEJORADO ----------------------
st.subheader("🗺️ Mapa de Precios por Distrito")

if "Distrito" not in filtered_df.columns or "Precio_ajustado" not in filtered_df.columns:
    st.warning("⚠️ No se encuentran las columnas necesarias para el mapa.")

else:
    # Calcular mediana por distrito
    agg = (filtered_df
           .dropna(subset=["Distrito", "Precio_ajustado"])
           .groupby("Distrito", as_index=True)["Precio_ajustado"]
           .agg(['median', 'count', 'min', 'max'])
           .round(0))

    # Normalizar nombres para emparejar con GeoJSON
    agg_dict = {}
    for distrito, row in agg.iterrows():
        key = _norm(distrito)
        agg_dict[key] = {
            'median': float(row['median']) if pd.notna(row['median']) else None,
            'count': int(row['count']) if pd.notna(row['count']) else 0,
            'min': float(row['min']) if pd.notna(row['min']) else None,
            'max': float(row['max']) if pd.notna(row['max']) else None,
        }

    try:
        # Cargar GeoJSON
        with open(GEO_PATH, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        # Recopilar valores para calcular cuartiles
        valores_validos = []

        # Asignar valores a cada distrito en el GeoJSON
        for feat in geojson.get("features", []):
            nombre = feat.get("properties", {}).get("NOMBRE")
            key = _norm(nombre)

            distrito_data = agg_dict.get(key, {})
            mediana = distrito_data.get('median')

            # Guardar todos los datos en properties
            feat["properties"]["precio_mediana"] = mediana
            feat["properties"]["num_viviendas"] = distrito_data.get('count', 0)
            feat["properties"]["precio_min"] = distrito_data.get('min')
            feat["properties"]["precio_max"] = distrito_data.get('max')

            if mediana is not None:
                valores_validos.append(mediana)

        # Definir esquema de colores si hay valores
        if valores_validos:
            # Cuartiles (Q1, Q2/mediana, Q3)
            q1, q2, q3 = np.percentile(valores_validos, [25, 50, 75])

            # Paleta en azules (HEX → RGBA aprox) con transparencia suave
            # blues = ["#0D1B2A", "#274C77", "#468FAF", "#89C2D9", "#FFFFFF"]
            color_scale = {
                'muy_barato': [137, 194, 217, 150],  # #89C2D9 azul claro
                'barato':     [ 70, 143, 175, 160],  # #468FAF azul medio-claro
                'medio':      [ 39,  76, 119, 170],  # #274C77 azul intermedio
                'muy_caro':   [ 13,  27,  42, 180],  # #0D1B2A azul muy oscuro
                'sin_datos':  [255, 255, 255, 110],  # #FFFFFF blanco translúcido
            }

            def get_color_for_price(precio):
                if precio is None or (isinstance(precio, float) and np.isnan(precio)):
                    return color_scale['sin_datos']
                elif precio <= q1:
                    return color_scale['muy_barato']
                elif precio <= q2:
                    return color_scale['barato']
                elif precio <= q3:
                    return color_scale['medio']
                else:
                    return color_scale['muy_caro']

            # Asignar colores a cada feature
            for feat in geojson.get("features", []):
                precio = feat["properties"].get("precio_mediana")
                feat["properties"]["_fill_color"] = get_color_for_price(precio)

            # Leyenda (coherente con la paleta azul)
            col1, col2 = st.columns([3, 1])
            with col2:
                st.markdown("### Leyenda de Precios")
                st.markdown(
                    f"""
                    <div style="line-height:1.9; font-size: 0.95rem;">
                    <span style="display:inline-block;width:14px;height:14px;background:#89C2D9;border:1px solid #bbb;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>
                    <b>Muy barato</b>: &lt; {q1:,.0f}€<br>

                    <span style="display:inline-block;width:14px;height:14px;background:#468FAF;border:1px solid #bbb;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>
                    <b>Barato</b>: {q1:,.0f}€ – {q2:,.0f}€<br>

                    <span style="display:inline-block;width:14px;height:14px;background:#274C77;border:1px solid #bbb;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>
                    <b>Medio</b>: {q2:,.0f}€ – {q3:,.0f}€<br>

                    <span style="display:inline-block;width:14px;height:14px;background:#0D1B2A;border:1px solid #bbb;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>
                    <b>Muy caro</b>: &gt; {q3:,.0f}€<br>

                    <span style="display:inline-block;width:14px;height:14px;background:#FFFFFF;border:1px solid #bbb;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>
                    <b>Sin datos</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Añadir estadística adicional
                st.divider()
                st.markdown("### 📈 Top 3 Más Caros")
                top_3 = agg.nlargest(3, 'median')
                for i, (distrito, row) in enumerate(top_3.iterrows(), 1):
                    st.markdown(f"{i}. **{distrito}**: {row['median']:,.0f}€")

                st.markdown("### 📉 Top 3 Más Baratos")
                bottom_3 = agg.nsmallest(3, 'median')
                for i, (distrito, row) in enumerate(bottom_3.iterrows(), 1):
                    st.markdown(f"{i}. **{distrito}**: {row['median']:,.0f}€")

            with col1:
                # Crear capa del mapa con mejor tooltip
                layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=geojson,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    get_fill_color="properties._fill_color",
                    get_line_color=[60, 60, 60, 255],
                    line_width_min_pixels=2,
                    auto_highlight=True,
                    highlight_color=[255, 255, 0, 128],
                )

                # Vista inicial centrada en Madrid
                view_state = pdk.ViewState(
                    latitude=40.4168,
                    longitude=-3.7038,
                    zoom=10.8,
                    bearing=0,
                    pitch=0
                )

                # Tooltip mejorado con formato
                tooltip = {
                    "html": """
                    <div style="font-family: Arial; padding: 10px;">
                        <h3 style="margin: 0; color: #333;">{NOMBRE}</h3>
                        <hr style="margin: 5px 0;">
                        <p style="margin: 5px 0;"><b>Mediana:</b> {precio_mediana}</p>
                        <p style="margin: 5px 0;"><b>Rango:</b> {precio_min} - {precio_max}</p>
                        <p style="margin: 5px 0;"><b>Nº Viviendas:</b> {num_viviendas}</p>
                    </div>
                    """,
                    "style": {
                        "backgroundColor": "white",
                        "color": "black",
                        "fontSize": "12px",
                        "borderRadius": "5px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.2)"
                    }
                }

                # Renderizar mapa
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        tooltip=tooltip,
                        map_style="mapbox://styles/mapbox/light-v10"
                    ),
                    use_container_width=True,
                    height=600
                )

        else:
            st.warning("⚠️ No hay datos de precio para mostrar en el mapa con los filtros actuales.")

    except FileNotFoundError:
        st.error(
            f"❌ No se encontró el archivo GeoJSON '{GEO_PATH}'. Asegúrate de que esté en el directorio del proyecto.")
    except json.JSONDecodeError as e:
        st.error(f"❌ Error al leer el archivo GeoJSON: {e}")
    except Exception as e:
        st.error(f"❌ Error inesperado al construir el mapa: {type(e).__name__}: {e}")
        st.exception(e)  # Para debugging

# ---------------------- TABLA DE RESULTADOS Y DESCARGA ----------------------
st.divider()
st.subheader("📋 Datos Filtrados")

# Crear tabs para diferentes vistas
tab1, tab2, tab3 = st.tabs(["🔍 Vista Completa", "📊 Resumen por Distrito", "💾 Descargar"])

with tab1:
    # Vista completa de datos
    st.info(
        f"Mostrando {filtered_df.shape[0]:,} registros de {df.shape[0]:,} totales ({filtered_df.shape[0] / df.shape[0] * 100:.1f}%)")

    # Opción para limitar filas mostradas para mejorar rendimiento
    if filtered_df.shape[0] > 1000:
        show_sample = st.checkbox("Mostrar solo primeras 1000 filas (mejora rendimiento)", value=True)
        display_df = filtered_df.head(1000) if show_sample else filtered_df
    else:
        display_df = filtered_df

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        column_config={
            "Precio_ajustado": st.column_config.NumberColumn(
                "Precio Ajustado",
                format="%.0f €",
            ),
            "Tamano": st.column_config.NumberColumn(
                "Tamaño",
                format="%.0f m²",
            ),
        }
    )

with tab2:
    # Resumen agregado por distrito
    if "Distrito" in filtered_df.columns and "Precio_ajustado" in filtered_df.columns:
        resumen = (filtered_df
                   .groupby("Distrito")["Precio_ajustado"]
                   .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
                   .round(0)
                   .sort_values('median', ascending=False))

        st.dataframe(
            resumen,
            use_container_width=True,
            column_config={
                "count": st.column_config.NumberColumn("Nº Viviendas", format="%d"),
                "mean": st.column_config.NumberColumn("Media", format="%.0f €"),
                "median": st.column_config.NumberColumn("Mediana", format="%.0f €"),
                "std": st.column_config.NumberColumn("Desv. Estándar", format="%.0f €"),
                "min": st.column_config.NumberColumn("Mínimo", format="%.0f €"),
                "max": st.column_config.NumberColumn("Máximo", format="%.0f €"),
            }
        )
    else:
        st.info("No hay suficientes datos para generar el resumen por distrito")

with tab3:
    # Opciones de descarga
    st.markdown("### 📥 Opciones de Descarga")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Datos Filtrados Completos**")
        csv_filtered = filtered_df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="⬇️ Descargar CSV Filtrado",
            data=csv_filtered,
            file_name=f"madrid_viviendas_filtrado_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Descarga todos los datos con los filtros aplicados"
        )

    with col2:
        st.markdown("**Resumen por Distrito**")
        if 'resumen' in locals():
            csv_resumen = resumen.to_csv(encoding='utf-8')
            st.download_button(
                label="⬇️ Descargar Resumen",
                data=csv_resumen,
                file_name=f"madrid_resumen_distrito_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Descarga el resumen estadístico por distrito"
            )
        else:
            st.info("Genera primero el resumen en la pestaña anterior")
