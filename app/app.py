# -------------------------------------------------------------
# Interfaz de Streamlit 
# -------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any, Tuple

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import json
import unicodedata
import bz2
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.cleaning import normaliza_distrito

# ---------------------- CONFIGURACIÓN BÁSICA ----------------------

st.set_page_config(page_title="Vivienda Madrid – Análisis por distrito", page_icon="🏠", layout="wide")
st.title("Análisis por distrito y factores determinantes del precio – Madrid")
st.caption("Explora datos inmobiliarios, visualiza KPIs, mapas y gráficos comparativos por distrito.")

# ---------------------- CARGA DE DATOS ----------------------------

DEFAULT_PATH = "data/final/viviendas_2011_2024_IAV.csv"
GEO_PATH = "app/madrid_distritos_geo.json"

@st.cache_data(show_spinner=True)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8")

try:
    df = load_csv(DEFAULT_PATH)
    if df.empty:
        st.warning("El CSV se cargó pero está vacío.")
        st.stop()
except Exception as e:
    st.error(f"No se pudo cargar el CSV. Detalle: {type(e).__name__}: {e}")
    st.stop()

# Normaliza nombres de columna (espacios)
df.columns = [c.strip() for c in df.columns]

# Renombra posibles aliases (ajusta si tus nombres son otros)
alias = {
    "IAV compra": "IAV_compra",
    "Esfuerzo compra": "Esfuerzo_compra",
    "IAV alquiler": "IAV_alquiler",
    "Esfuerzo alquiler": "Esfuerzo_alquiler",
}
df = df.rename(columns={a:b for a,b in alias.items() if a in df.columns})

# Tipos numéricos
for c in ["Precio_ajustado","IAV_compra","Esfuerzo_compra","IAV_alquiler","Esfuerzo_alquiler","Tamano","Renta_neta_hogar"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

st.success(f"✓ Datos cargados: {df.shape[0]:,} registros")

# ---------------------- UTILIDADES -------------------------------------------

DISTRITO_MAPPING = {
    'FUENCARRALELPARDO': 'FUENCARRAL EL PARDO',
    'MONCLOAARAVACA': 'MONCLOA ARAVACA',
    'PUENTEDEVALLECAS': 'PUENTE DE VALLECAS',
    'CIUDADLINEAL': 'CIUDAD LINEAL',
    'VILLADEVALLECAS': 'VILLA DE VALLECAS',
    'SANBLASCANILLEJAS': 'SAN BLAS CANILLEJAS',
}
def _norm(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip().upper()
    x = DISTRITO_MAPPING.get(x, x)
    x = x.replace("Ñ", "###TEMP_N###")
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    x = x.replace("###TEMP_N###", "Ñ")
    for a in ["-", "_", "/", ",", ".", "'", "`", "(", ")", "[", "]", "{", "}"]:
        x = x.replace(a, " ")
    x = " ".join(x.split())
    return x

# ---------------------- FILTROS ----------------------------------------------

critical_cols = ["Distrito"]
missing_critical = [c for c in critical_cols if c not in df.columns]
if missing_critical:
    st.error(f"❌ Faltan columnas críticas: {', '.join(missing_critical)}")
    st.stop()

optional_cols = ["Ano", "Operacion", "Tipo_vivienda", "Precio_ajustado"]
missing_optional = [c for c in optional_cols if c not in df.columns]
if missing_optional:
    st.info(f"ℹ️ Filtros no disponibles: {', '.join(missing_optional)}")

def build_filters(_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    filters_state: Dict[str, Any] = {}
    with st.sidebar:
        st.subheader("🔍 Filtros")
        if st.button("🔄 Resetear filtros"):
            st.rerun()
        st.metric("Total registros originales", f"{len(_df):,}")
        st.divider()

    df_f = _df.copy()
    conditions = []

    if "Ano" in _df.columns and _df["Ano"].notna().any():
        years = sorted(_df["Ano"].dropna().astype(int).unique().tolist())
        default_years = years[-10:] if len(years) > 10 else years
        sel_years = st.sidebar.multiselect("📅 Año", options=years, default=default_years)
        filters_state["Ano"] = sel_years
        if sel_years:
            conditions.append(df_f["Ano"].isin(sel_years))

    if "Distrito" in _df.columns and _df["Distrito"].notna().any():
        distritos = sorted(_df["Distrito"].dropna().astype(str).unique().tolist())
        default_dist = distritos if len(distritos) <= 10 else []
        sel_dist = st.sidebar.multiselect("📍 Distrito", options=distritos, default=default_dist)
        filters_state["Distrito"] = sel_dist
        if sel_dist:
            conditions.append(df_f["Distrito"].isin(sel_dist))

    if "Precio_ajustado" in _df.columns and _df["Precio_ajustado"].notna().any():
        serie = pd.to_numeric(_df["Precio_ajustado"], errors="coerce").dropna()
        pmin, pmax = float(serie.min()), float(serie.max())
        span = pmax - pmin
        step = 5_000.0 if span > 100_000 else (1_000.0 if span > 10_000 else 100.0)
        p0, p1 = st.sidebar.slider("💰 Precio ajustado (€)", min_value=pmin, max_value=pmax,
                                   value=(pmin, pmax), step=step, format="%d €")
        filters_state["Precio_ajustado"] = (p0, p1)
        conditions.append((df_f["Precio_ajustado"] >= p0) & (df_f["Precio_ajustado"] <= p1))

    if "Operacion" in _df.columns and _df["Operacion"].notna().any():
        ops = sorted(_df["Operacion"].dropna().astype(str).unique().tolist())
        sel_op = st.sidebar.selectbox("💼 Operación (opción única)", options=["(Todas)"] + ops, index=0)
        if sel_op != "(Todas)":
            filters_state["Operacion"] = sel_op
            conditions.append(df_f["Operacion"] == sel_op)

    if conditions:
        cond = conditions[0]
        for c in conditions[1:]:
            cond = cond & c
        df_f = df_f[cond]

    with st.sidebar:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Registros filtrados", f"{len(df_f):,}")
        with col2:
            pct = (len(df_f) / len(_df) * 100) if len(_df) > 0 else 0
            st.metric("Porcentaje", f"{pct:.1f}%")
        if 0 < len(df_f) < 10:
            st.warning("⚠️ Pocos registros con estos filtros")
        elif len(df_f) == 0:
            st.error("❌ Sin resultados para estos filtros")

    return df_f, filters_state

filtered_df, state = build_filters(df)
if filtered_df.empty:
    st.error("No hay datos con los filtros actuales.")
    st.stop()

# ---------------------- KPIs ----------------------
st.subheader("📊 Métricas Principales")
if "Precio_ajustado" in filtered_df.columns:
    k1, k2, k3, k4, k5 = st.columns(5)
    precio = filtered_df["Precio_ajustado"].dropna()
    mediana = precio.median()
    media = precio.mean()
    minimo = precio.min()
    maximo = precio.max()
    with k1: st.metric("Mediana General", f"{mediana:,.0f}€" if pd.notna(mediana) else "N/D")
    with k2:
        delta = ((media - mediana) / mediana * 100) if pd.notna(mediana) and mediana != 0 else 0
        st.metric("Media General", f"{media:,.0f}€" if pd.notna(media) else "N/D", f"{delta:+.1f}% vs mediana")
    with k3: st.metric("Precio Mínimo", f"{minimo:,.0f}€" if pd.notna(minimo) else "N/D")
    with k4: st.metric("Precio Máximo", f"{maximo:,.0f}€" if pd.notna(maximo) else "N/D")
    with k5:
        nd = filtered_df["Distrito"].nunique() if "Distrito" in filtered_df.columns else 0
        st.metric("Distritos", f"{nd}/21")

st.divider()

# ===================== Diccionario de datos =====================
with st.expander("📚 Diccionario de datos"):
    st.markdown("### Indicadores de accesibilidad y esfuerzo")

    st.markdown("**IAV compra**: Índice de accesibilidad a la vivienda en compra.")
    st.latex(r"""
    IAV_{compra} = 
    \frac{\text{Renta media del hogar}}
    {\text{Precio de venta (€/m}^{2}) \times 40}
    """)
    st.markdown("≥ 1 → accesible; < 1 → no accesible")

    st.markdown("**Esfuerzo compra**: Años de renta bruta necesarios para comprar.")
    st.latex(r"""
    Esfuerzo_{compra} = 
    \frac{\text{Precio de venta}}{\text{Renta bruta del hogar}}
    """)
    st.markdown("≤ 5 → poco esfuerzo; > 5 → mucho esfuerzo")

    st.markdown("**IAV alquiler**: Capacidad de cubrir el alquiler con la renta media del hogar.")
    st.latex(r"""
    IAV_{alquiler} = 
    \frac{\text{Renta media del hogar}}
    {\text{Precio de alquiler (€/m}^{2}) \times 40 \times 12}
    """)
    st.markdown("≥ 1 → cubre alquiler; < 1 → no cubre")

    st.markdown("**Esfuerzo alquiler (%)**: Porcentaje de la renta destinado al alquiler.")
    st.latex(r"""
    Esfuerzo_{alquiler}(\%) = 
    \frac{\text{Precio de alquiler (€/m}^{2}) \times 40 \times 12}
    {\text{Renta media del hogar}} \times 100
    """)
    st.markdown("≤ 30% → sostenible; 30–35% → tensión; > 35% → sobreesfuerzo")

    st.markdown("**Precio ajustado**: Precio corregido por distrito (método híbrido).")
    st.latex(r"""
    Precio_{ajustado} = 
    \alpha \pm (\text{15\% venta / 5\% alquiler}) + \text{ruido controlado}
    """)
    st.markdown("Alinea la proyección con el mercado.")

# ---------------------- MAPA INTERACTIVO ----------------------
st.subheader("🗺️ Mapa por distrito (elige la variable)")

if "Distrito" not in filtered_df.columns:
    st.warning("⚠️ Falta la columna 'Distrito'.")
else:
    # ---------- Agregaciones robustas por operación ----------
    # Precio (no depende de la operación)
    have_price = "Precio_ajustado" in filtered_df.columns
    if have_price:
        precio_agg = (filtered_df.dropna(subset=["Distrito", "Precio_ajustado"])
                      .groupby("Distrito")["Precio_ajustado"]
                      .agg(median="median", count="count", min="min", max="max"))
    else:
        precio_agg = pd.DataFrame()

    # IAV de compra -> solo sobre filas de venta
    if {"IAV_compra","Operacion"}.issubset(filtered_df.columns):
        venta = filtered_df[filtered_df["Operacion"].astype(str).str.lower() == "venta"]
        iavc_agg = (venta.dropna(subset=["Distrito", "IAV_compra"])
                    .groupby("Distrito")["IAV_compra"]
                    .agg(IAV_compra_mediana="median"))
    else:
        iavc_agg = pd.DataFrame()

    # Esfuerzo de compra -> solo sobre filas de venta  <-- NUEVO
    if {"Esfuerzo_compra","Operacion"}.issubset(filtered_df.columns):
        esf_c_agg = (venta.dropna(subset=["Distrito","Esfuerzo_compra"])
                    .groupby("Distrito")["Esfuerzo_compra"]
                    .agg(Esfuerzo_compra_mediana="median"))
    else:
        esf_c_agg = pd.DataFrame()
        
    # IAV de alquiler -> solo sobre filas de alquiler
    if {"IAV_alquiler","Operacion"}.issubset(filtered_df.columns):
        alqu = filtered_df[filtered_df["Operacion"].astype(str).str.lower() == "alquiler"]
        iava_agg = (alqu.dropna(subset=["Distrito", "IAV_alquiler"])
                    .groupby("Distrito")["IAV_alquiler"]
                    .agg(IAV_alquiler_mediana="median"))
    else:
        iava_agg = pd.DataFrame()

    # Esfuerzo de alquiler -> solo sobre filas de alquiler
    if {"Esfuerzo_alquiler","Operacion"}.issubset(filtered_df.columns):
        esf_agg = (alqu.dropna(subset=["Distrito", "Esfuerzo_alquiler"])
                   .groupby("Distrito")["Esfuerzo_alquiler"]
                   .agg(Esfuerzo_alquiler_mediana="median"))
    else:
        esf_agg = pd.DataFrame()

    # ---------- Unificación de todo en un único DataFrame por distrito ----------
    parts = []
    if not precio_agg.empty:
        parts.append(precio_agg.rename(columns={
            "median": "Precio_mediana",
            "count":  "Num_viviendas",
            "min":    "Precio_min",
            "max":    "Precio_max"
        }))
    if not iavc_agg.empty: parts.append(iavc_agg)
    if not iava_agg.empty: parts.append(iava_agg)
    if not esf_agg.empty:  parts.append(esf_agg)
    if not esf_c_agg.empty: parts.append(esf_c_agg) 

    if not parts:
        st.warning("No hay métricas con datos para los filtros actuales. Cambia Año u Operación.")
        st.stop()

    agg = pd.concat(parts, axis=1)

    # ---------- Normaliza índice para cruzar con GeoJSON ----------
    agg_norm = {}
    for d, row in agg.iterrows():
        key = _norm(d)
        agg_norm[key] = {k: (None if pd.isna(v) else float(v)) for k, v in row.to_dict().items()}

    # ---------- Selector: solo métricas que tienen datos no nulos ----------
    def nn(col):  # nº distritos con dato
        return int(agg[col].notna().sum()) if col in agg.columns else 0

    options = []
    if "Precio_mediana" in agg.columns and nn("Precio_mediana") > 0:
        options.append(("Precio mediano (€)", "Precio_mediana", nn("Precio_mediana")))
    if "IAV_compra_mediana" in agg.columns and nn("IAV_compra_mediana") > 0:
        options.append(("IAV compra (mediana)", "IAV_compra_mediana", nn("IAV_compra_mediana")))
    if "IAV_alquiler_mediana" in agg.columns and nn("IAV_alquiler_mediana") > 0:
        options.append(("IAV alquiler (mediana)", "IAV_alquiler_mediana", nn("IAV_alquiler_mediana")))
    if "Esfuerzo_alquiler_mediana" in agg.columns and nn("Esfuerzo_alquiler_mediana") > 0:
        options.append(("Esfuerzo alquiler % (mediana)", "Esfuerzo_alquiler_mediana", nn("Esfuerzo_alquiler_mediana")))
    if "Esfuerzo_compra_mediana" in agg.columns and nn("Esfuerzo_compra_mediana") > 0:      # <-- NUEVO
        options.append(("Esfuerzo compra (mediana)", "Esfuerzo_compra_mediana", nn("Esfuerzo_compra_mediana")))

    if not options:
        st.warning("No hay métricas con datos para los filtros actuales. Cambia Año u Operación.")
        st.stop()

    # por defecto, la que tenga más distritos con datos
    options.sort(key=lambda x: x[2], reverse=True)
    labels = [lbl for (lbl, _, _) in options]
    label_to_col = {lbl: col for (lbl, col, _) in options}

    sel_label = st.selectbox("Variable para colorear el mapa", labels, index=0)
    sel_col = label_to_col[sel_label]

    # Mensajes de contexto si la operación elegida deja sin datos
    if state.get("Operacion") == "venta" and sel_col in ["IAV_alquiler_mediana","Esfuerzo_alquiler_mediana"]:
        st.info("Has filtrado por operación **venta**. Las métricas de **alquiler** pueden quedar sin datos.")
    if state.get("Operacion") == "alquiler" and sel_col in ["IAV_compra_mediana","Esfuerzo_compra_mediana"]:
        st.info("Has filtrado por operación **alquiler**. Las métricas de **compra** pueden quedar sin datos.")

    # ---------- Construcción del GeoJSON con propiedades ----------
    try:
        with open(GEO_PATH, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        valores_validos = []

        def fmt_n(x, suf=""):
            if x is None or (isinstance(x, float) and np.isnan(x)): return "N/D"
            return f"{x:,.2f}{suf}"

        for feat in geojson.get("features", []):
            nombre = feat.get("properties", {}).get("NOMBRE")
            key = _norm(nombre)
            data = agg_norm.get(key, {})

            # Precio
            feat["properties"]["Precio_mediana"] = data.get("Precio_mediana")
            feat["properties"]["Num_viviendas"]  = int(data.get("Num_viviendas", 0)) if data.get("Num_viviendas") is not None else 0
            feat["properties"]["Precio_min"]     = data.get("Precio_min")
            feat["properties"]["Precio_max"]     = data.get("Precio_max")
            # IAVs / Esfuerzo
            feat["properties"]["IAV_compra_mediana"]        = data.get("IAV_compra_mediana")
            feat["properties"]["IAV_alquiler_mediana"]      = data.get("IAV_alquiler_mediana")
            feat["properties"]["Esfuerzo_alquiler_mediana"] = data.get("Esfuerzo_alquiler_mediana")
            feat["properties"]["Esfuerzo_compra_mediana"]   = data.get("Esfuerzo_compra_mediana")

            v = data.get(sel_col)
            if v is not None:
                valores_validos.append(v)

            # Formatos
            feat["properties"]["_precio_mediana_fmt"] = fmt_n(feat["properties"]["Precio_mediana"], "€")
            feat["properties"]["_precio_min_fmt"]     = fmt_n(feat["properties"]["Precio_min"], "€")
            feat["properties"]["_precio_max_fmt"]     = fmt_n(feat["properties"]["Precio_max"], "€")
            feat["properties"]["_iavc_fmt"]           = fmt_n(feat["properties"]["IAV_compra_mediana"])
            feat["properties"]["_iava_fmt"]           = fmt_n(feat["properties"]["IAV_alquiler_mediana"])
            feat["properties"]["_esf_fmt"]            = fmt_n(feat["properties"]["Esfuerzo_alquiler_mediana"], "%")
            feat["properties"]["_esf_c_fmt"]          = fmt_n(feat["properties"]["Esfuerzo_compra_mediana"])

        if not valores_validos:
            st.warning("⚠️ No hay valores numéricos para colorear el mapa con los filtros actuales.")
            st.stop()

        # ---------- Cuartiles y colores ----------
        q1, q2, q3 = np.percentile(valores_validos, [25, 50, 75])
        if sel_col in ["IAV_compra_mediana", "IAV_alquiler_mediana"]:
            color_scale = {
                'q1': [215, 25, 28, 180], # rojo
                'q2': [253, 174, 97, 180], # naranja
                'q3': [255, 255, 191, 180], # amarillo verdoso
                'q4': [166, 217, 106, 180], # verde
                'na': [255, 255, 255, 110],
            }
        elif sel_col in ["Esfuerzo_compra_mediana", "Esfuerzo_alquiler_mediana"]:
            color_scale = {
                'q1': [166, 217, 106, 180], # verde
                'q2': [255, 255, 191, 180], # amarillo verdoso
                'q3': [253, 174, 97, 180], # naranja
                'q4': [215, 25, 28, 180], # rojo
                'na': [255, 255, 255, 110], # gris
            }
        else: # Colores azules por defecto
            color_scale = {
                'q1': [137, 194, 217, 150],
                'q2': [ 70, 143, 175, 160],
                'q3': [ 39,  76, 119, 170],
                'q4': [ 13,  27,  42, 180],
                'na': [255, 255, 255, 110],
            }

        def color_for(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return color_scale['na']
            if v <= q1: return color_scale['q1']
            if v <= q2: return color_scale['q2']
            if v <= q3: return color_scale['q3']
            return color_scale['q4']

        for feat in geojson.get("features", []):
            v = feat["properties"].get(sel_col)
            feat["properties"]["_fill_color"] = color_for(v)

        st.caption(f"Coloreando por: **{sel_label}**")
        
        def rgba_to_hex(rgba): # Convertir RGB a HEX
            r, g, b = rgba[:3]
            return f'#{r:02X}{g:02X}{b:02X}'

        hex_q1 = rgba_to_hex(color_scale['q1'])
        hex_q2 = rgba_to_hex(color_scale['q2'])
        hex_q3 = rgba_to_hex(color_scale['q3'])
        hex_q4 = rgba_to_hex(color_scale['q4'])

        # Leyenda compacta
        st.markdown(
            f"""
            <div style="line-height:1.6; font-size:0.9rem">
            <span style="display:inline-block;width:12px;height:12px;background:{hex_q1};border:1px solid #bbb;border-radius:2px;margin-right:6px;"></span>
            ≤ {q1:,.2f} &nbsp;&nbsp;
            <span style="display:inline-block;width:12px;height:12px;background:{hex_q2};border:1px solid #bbb;border-radius:2px;margin:0 6px;"></span>
            {q1:,.2f} – {q2:,.2f} &nbsp;&nbsp;
            <span style="display:inline-block;width:12px;height:12px;background:{hex_q3};border:1px solid #bbb;border-radius:2px;margin:0 6px;"></span>
            {q2:,.2f} – {q3:,.2f} &nbsp;&nbsp;
            <span style="display:inline-block;width:12px;height:12px;background:{hex_q4};border:1px solid #bbb;border-radius:2px;margin-left:6px;"></span>
            ≥ {q3:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------- Capa y render ----------
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
        view_state = pdk.ViewState(latitude=40.4168, longitude=-3.7038, zoom=9.5, bearing=0, pitch=0)
        tooltip = {
            "html": """
            <div style="font-family: Arial; padding: 10px;">
                <h3 style="margin: 0; color: #333;">{NOMBRE}</h3>
                <hr style="margin: 5px 0;">
                <p style="margin: 5px 0;"><b>Precio mediano:</b> {_precio_mediana_fmt}</p>
                <p style="margin: 5px 0;"><b>Rango:</b> {_precio_min_fmt} - {_precio_max_fmt}</p>
                <p style="margin: 5px 0;"><b>Nº Viviendas:</b> {Num_viviendas}</p>
                <p style="margin: 5px 0;"><b>IAV compra (mediana):</b> {_iavc_fmt}</p>
                <p style="margin: 5px 0;"><b>IAV alquiler (mediana):</b> {_iava_fmt}</p>
                <p style="margin: 5px 0;"><b>Esfuerzo alquiler (mediana):</b> {_esf_fmt}</p>
                <p style="margin: 5px 0;"><b>Esfuerzo compra (mediana):</b> {_esf_c_fmt}</p>
            </div>
            """,
            "style": {"backgroundColor": "white", "color": "black", "fontSize": "12px",
                      "borderRadius": "5px", "boxShadow": "0 2px 4px rgba(0,0,0,0.2)"}
        }
        
        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=view_state,
                     tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v10"),
            use_container_width=True, height=600
        )
        
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo GeoJSON '{GEO_PATH}'.")
    except json.JSONDecodeError as e:
        st.error(f"❌ Error al leer el GeoJSON: {e}")
    except Exception as e:
        st.error(f"❌ Error inesperado: {type(e).__name__}: {e}")
        st.exception(e)


# ---------------------- TABLA & DESCARGAS ----------------------
st.divider()
st.subheader("📋 Datos Filtrados")

tab1, tab2, tab3 = st.tabs(["🔍 Vista Completa", "📊 Resumen por Distrito", "💾 Descargar"])

with tab1:
    st.info(f"Mostrando {filtered_df.shape[0]:,} registros de {df.shape[0]:,} totales "
            f"({filtered_df.shape[0] / df.shape[0] * 100:.1f}%)")
    display_df = filtered_df.head(1000) if filtered_df.shape[0] > 1000 and st.checkbox(
        "Mostrar solo primeras 1000 filas (mejora rendimiento)", value=True) else filtered_df
    st.dataframe(display_df, width="stretch", height=500)

with tab2:
    if "Distrito" in filtered_df.columns and "Precio_ajustado" in filtered_df.columns:
        resumen = (filtered_df.groupby("Distrito")["Precio_ajustado"]
                   .agg(['count','mean','median','std','min','max']).round(0)
                   .sort_values('median', ascending=False))
        st.dataframe(resumen, width="stretch")
    else:
        st.info("No hay suficientes datos para generar el resumen por distrito")

with tab3:
    st.markdown("### 📥 Opciones de Descarga")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Datos Filtrados Completos**")
        st.download_button("⬇️ Descargar CSV Filtrado",
                           data=filtered_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"madrid_viviendas_filtrado_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv")
    with c2:
        if "resumen" in locals():
            st.markdown("**Resumen por Distrito**")
            st.download_button("⬇️ Descargar Resumen",
                               data=resumen.to_csv().encode("utf-8"),
                               file_name=f"madrid_resumen_distrito_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                               mime="text/csv")
        else:
            st.info("Genera primero el resumen en la pestaña anterior")

# ===================== Pestaña: Evolución temporal =====================
ts_tab, = st.tabs(["📈 Evolución temporal"])

# Función para cambiar nombre visualizado de métricas para aplicarlo más adelante
metricas = ["Precio_ajustado", "IAV_alquiler","Esfuerzo_alquiler", "IAV_compra", "Esfuerzo_compra"]
def nombres_metricas(met):   
    return met.replace("_", " ")

with ts_tab:
    if "Ano" not in filtered_df.columns:
        st.info("No hay columna 'Ano' para series temporales.")
    else:
        # Métrica a visualizar
        metrica = st.radio(
            "Métrica",
            metricas,
            format_func=nombres_metricas,
            horizontal=True,
            key="ts_metric"
        )

        # Usa los distritos del estado de filtros (sidebar)
        sel_distritos = state.get("Distrito") or []

        df_ts = filtered_df.copy()
        if sel_distritos:
            df_ts = df_ts[df_ts["Distrito"].isin(sel_distritos)]

        if metrica not in df_ts.columns:
            st.info("La métrica seleccionada no existe en el dataset filtrado.")
        else:
            # Serie: mediana anual por distrito
            ts = (
                df_ts.dropna(subset=["Ano", "Distrito", metrica])
                     .groupby(["Ano", "Distrito"])[metrica]
                     .median()
                     .reset_index()
            )

            if ts.empty:
                st.warning("No hay datos para la métrica con los filtros actuales.")
            else:
                # Pivot a formato ancho para graficar varias líneas (una por distrito)
                wide = (
                    ts.pivot(index="Ano", columns="Distrito", values=metrica)
                      .sort_index()
                )

                # Si hay demasiados distritos y el usuario no eligió ninguno,
                # muestra solo los 8 con más registros para que el gráfico sea legible
                if not sel_distritos and wide.shape[1] > 8:
                    # contamos presencia por distrito
                    top_cols = (
                        ts["Distrito"].value_counts()
                        .index.tolist()[:8]
                    )
                    wide = wide[top_cols]
                    st.caption("Mostrando los 8 distritos con más datos (ajusta los filtros para afinar).")

                st.line_chart(wide, width="stretch")
                st.caption("Mediana anual por distrito y métrica seleccionada (según los filtros de la barra lateral).")


# ===================== Pestaña: Visualización 2D con UMAP =====================

umap_tab, = st.tabs(["VISUALIZACIÓN"])
with umap_tab:
    st.caption("Proyección bidimensional de las observaciones mediante UMAP y distribución de los clusters por distrito.")

    # Pestañas internas: UMAP y Evolución de clusters
    tab_umap, tab_evo = st.tabs(["📌 UMAP 2D", "✳️ Distribución de los clusters"])

    # ---- Tab 1: UMAP 2D ----
    with tab_umap:
        umap_path = "reports/figures/umap.png"
        if umap_path:
            st.image(str(umap_path), caption="Visualización 2D con UMAP", width="stretch")
            with open(umap_path, "rb") as f:
                st.download_button("Descargar imagen (PNG)", f, file_name="umap.png")
        else:
            st.warning("No se encuentra **umap.png**. Colocar en reports/figures.")

    # ---- Tab 2: Evolución temporal de los clusters ----
    with tab_evo:
        evo_path = "reports/figures/porcentaje_viviendas_distrito_cluster.png"
        if evo_path:
            st.image(str(evo_path), caption="Distribución de los clusters", width="stretch")
            with open(evo_path, "rb") as f:
                st.download_button(
                    "Descargar imagen (PNG)",
                    f,
                    file_name="porcentaje_viviendas_distrito_cluster.png"
                )
        else:
            st.warning("No se encuentra **porcentaje_viviendas_distrito_cluster.png**. Colocar en reports/figures.")



# ===================== Tabla de clusters (UMAP) =====================
st.markdown("### Tabla de clusters (interpretación UMAP)")

clusters_data = [
    {
        "CLUSTER": "C0",
        "DESCRIPCIÓN GENERAL": "Apartamentos familiares en zonas consolidadas",
        "CARACTERÍSTICAS PRINCIPALES": "Tamaño medio (~104 m²), 2–3 habitaciones, precios medios-altos, alto porcentaje exterior y ascensor."
    },
    {
        "CLUSTER": "C1",
        "DESCRIPCIÓN GENERAL": "Vivienda mediana en alquiler en zonas centrales",
        "CARACTERÍSTICAS PRINCIPALES": "Viviendas de ~80 m², 2 habitaciones, bajo precio, alto porcentaje de alquiler."
    },
    {
        "CLUSTER": "C2",
        "DESCRIPCIÓN GENERAL": "Viviendas de lujo o gran superficie",
        "CARACTERÍSTICAS PRINCIPALES": "Chalets y tipologías unifamiliares, ~300 m² promedio, precios muy altos."
    },
    {
        "CLUSTER": "C3",
        "DESCRIPCIÓN GENERAL": "Vivienda económica y compacta",
        "CARACTERÍSTICAS PRINCIPALES": "~65 m² promedio, 2 habitaciones, precios bajos, baja dotación de equipamiento."
    },
    {
        "CLUSTER": "C4",
        "DESCRIPCIÓN GENERAL": "Vivienda moderna y bien equipada en zonas centrales",
        "CARACTERÍSTICAS PRINCIPALES": "~100 m², alto equipamiento (terraza, ascensor), precios medios-altos."
    },
    {
        "CLUSTER": "C5",
        "DESCRIPCIÓN GENERAL": "Vivienda periférica estándar bien dotada",
        "CARACTERÍSTICAS PRINCIPALES": "Viviendas de ~90 m², garaje y trastero frecuentes, precio medio, ubicadas en expansión periférica."
    },
]

df_clusters = pd.DataFrame(clusters_data)

st.dataframe(
    df_clusters,
    width="stretch",
    hide_index=True,
)

# Descarga opcional
st.download_button(
    "Descargar tabla de clusters (CSV)",
    data=df_clusters.to_csv(index=False).encode("utf-8"),
    file_name="tabla_clusters_umap.csv",
    mime="text/csv"
)


# ===================== Mapa: Tasa de gentrificación (Cluster_KMeans) =====================
st.subheader("🗺️ Tasa de gentrificación por distrito (%) — KMeans")

# 1) Cargar clusters.csv 
CLUSTERS_PATH = "data/final/clusters.csv"

@st.cache_data(show_spinner=True)
def load_clusters_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8")

try:
    dclu = load_clusters_csv(CLUSTERS_PATH)
except Exception as e:
    st.error(f"No pude leer 'clusters.csv': {type(e).__name__}: {e}")
    st.stop()

# 2) Checks mínimos
if "Distrito" not in dclu.columns or "Cluster_KMeans" not in dclu.columns:
    st.error("El archivo 'clusters.csv' debe incluir las columnas 'Distrito' y 'Cluster_KMeans'.")
    st.stop()

# Normalizar cabeceras y tipos
dclu.columns = [c.strip() for c in dclu.columns]
dclu["Cluster_KMeans"] = dclu["Cluster_KMeans"].astype(str).str.strip()

# 3) Respetar filtros laterales (Distrito; y Año si existe)
sel_distritos = []
try:
    sel_distritos = state.get("Distrito") or []
except NameError:
    pass

if sel_distritos:
    dclu = dclu[dclu["Distrito"].isin(sel_distritos)]

if "Ano" in dclu.columns and "Ano" in filtered_df.columns:
    # Si el usuario seleccionó años en el sidebar, aplica el mismo filtro
    sel_years = state.get("Ano") or []
    if sel_years:
        dclu = dclu[dclu["Ano"].isin(sel_years)]

if dclu.empty:
    st.warning("No hay registros en 'clusters.csv' tras aplicar los filtros actuales.")
    st.stop()

# 4) Definir clusters 'gentrificadores' por defecto (etiq. 'C#' o números)
ALL_LABELS = sorted(dclu["Cluster_KMeans"].dropna().unique().tolist())
DEFAULT_G = [lab for lab in ["C0","C1","C3","C4","0","1","3","4"] if lab in ALL_LABELS]

# Si quieres que sea 100% fijo sin UI, deja esta línea; si prefieres permitir edición, usa un multiselect.
GENTRIFYING = set(DEFAULT_G) if DEFAULT_G else set(ALL_LABELS)  # fallback por si no coincide

# 5) Calcular tasa por distrito (porcentaje de observaciones en clusters gentrificadores)
tmp = dclu.dropna(subset=["Distrito", "Cluster_KMeans"]).copy()
tmp["_gflag"] = tmp["Cluster_KMeans"].isin(GENTRIFYING)

tasa = (tmp.groupby("Distrito")["_gflag"].mean() * 100).rename("tasa_gent_%")
n_obs = tmp.groupby("Distrito")["_gflag"].size().rename("n_obs")
res = pd.concat([tasa, n_obs], axis=1)

# 6) Cargar GeoJSON y volcar propiedades
try:
    with open(GEO_PATH, "r", encoding="utf-8") as f:
        geojson_g = json.load(f)

    # índice normalizado para machear distritos
    res_norm = { _norm(idx): vals for idx, vals in res.to_dict(orient="index").items() }

    valores = []
    for feat in geojson_g["features"]:
        key = _norm(feat["properties"].get("NOMBRE"))
        dat = res_norm.get(key, {})
        v = float(dat.get("tasa_gent_%", np.nan)) if dat else np.nan
        n = int(dat.get("n_obs", 0)) if dat else 0
        feat["properties"]["tasa_gent"] = v
        feat["properties"]["n_obs"] = n
        feat["properties"]["_tasa_fmt"] = "N/D" if np.isnan(v) else f"{v:.2f}%"
        if not np.isnan(v): valores.append(v)

    if not valores:
        st.warning("No hay valores para calcular la tasa con los filtros actuales.")
        st.stop()

    # 7) Colorear por cuartiles (escala azul)
    q1, q2, q3 = np.percentile(valores, [25, 50, 75])
    def col(v):
        if np.isnan(v): return [255,255,255,110]
        if v <= q1: return [137,194,217,150]
        if v <= q2: return [ 70,143,175,160]
        if v <= q3: return [ 39, 76,119,170]
        return [ 13, 27, 42,180]

    for f in geojson_g["features"]:
        f["properties"]["_fill_color"] = col(f["properties"].get("tasa_gent", np.nan))

    # 8) Render
    st.caption(f"Fuente: clusters.csv · Algoritmo: KMeans · Clusters gentrificadores: {', '.join(sorted(GENTRIFYING))}")
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_g,
        pickable=True,
        filled=True,
        stroked=True,
        get_fill_color="properties._fill_color",
        get_line_color=[60,60,60,255],
        line_width_min_pixels=2,
        auto_highlight=True,
        highlight_color=[255,255,0,128],
    )
    view = pdk.ViewState(latitude=40.4168, longitude=-3.7038, zoom=10.8)
    tooltip = {
    "html": """
    <div style="font-family: Arial; padding: 10px;">
        <h3 style="margin:0">{NOMBRE}</h3>
        <hr style="margin:6px 0">
        <p style="margin:2px 0"><b>Tasa gentrificación:</b> {_tasa_fmt}</p>   <!-- ← usar _tasa_fmt -->
        <p style="margin:2px 0"><b>Nº obs.:</b> {n_obs}</p>
    </div>
    """,
    "style": {"backgroundColor":"white","color":"black","fontSize":"12px",
              "borderRadius":"5px","boxShadow":"0 2px 4px rgba(0,0,0,.2)"}
    }

    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view,
                 tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v10"),
        use_container_width=True, height=520
    )

    # 9) Leyenda
    st.markdown(
        f"""
        <div style="line-height:1.6; font-size:0.9rem">
          <span style="display:inline-block;width:12px;height:12px;background:#89C2D9;border:1px solid #bbb;border-radius:2px;margin-right:6px;"></span>
          ≤ {q1:,.2f}% &nbsp;&nbsp;
          <span style="display:inline-block;width:12px;height:12px;background:#468FAF;border:1px solid #bbb;border-radius:2px;margin:0 6px;"></span>
          {q1:,.2f}% – {q2:,.2f}% &nbsp;&nbsp;
          <span style="display:inline-block;width:12px;height:12px;background:#274C77;border:1px solid #bbb;border-radius:2px;margin:0 6px;"></span>
          {q2:,.2f}% – {q3:,.2f}% &nbsp;&nbsp;
          <span style="display:inline-block;width:12px;height:12px;background:#0D1B2A;border:1px solid #bbb;border-radius:2px;margin-left:6px;"></span>
          ≥ {q3:,.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error(f"No se encontró el GeoJSON en '{GEO_PATH}'.")
except Exception as e:
    st.error(f"Error al construir el mapa: {type(e).__name__}: {e}")
    st.exception(e)

# ===================== Pestaña: Predicción de precio =====================
st.divider()
st.subheader("🔮 Predicción de precios de vivienda")


MODEL_PATH_VENTA = "models/rf_final_venta.pkl.bz2"
MODEL_PATH_ALQUILER = "models/rf_final_alquiler.pkl.bz2"

@st.cache_resource(show_spinner=True)
def load_models():
    with bz2.open(MODEL_PATH_VENTA, "rb") as f:
        model_rf_venta = pickle.load(f)
    with bz2.open(MODEL_PATH_ALQUILER, "rb") as f:
        model_rf_alquiler = pickle.load(f)
    return model_rf_venta, model_rf_alquiler

# --- Interfaz de inputs ---
st.markdown("### Introduce las características de la vivienda:")

tipos_vivienda = ["Ático", "Chalet", "Dúplex", "Estudio",
                  "Tríplex", "Apartamento", "Mansión", "Loft"]

distritos_visibles = [
    "Centro",
    "Chamberí",
    "Arganzuela",
    "Vicálvaro",
    "Moratalaz",
    "Barajas",
    "Puente de Vallecas",
    "Latina",
    "Villaverde",
    "Tetuán",
    "Usera",
    "Salamanca",
    "Villa de Vallecas",
    "Carabanchel",
    "Fuencarral-El Pardo",
    "Ciudad Lineal",
    "Moncloa-Aravaca",
    "Retiro",
    "Chamartín",
    "Hortaleza",
    "San Blas-Canillejas"
]

col1, col2, col3 = st.columns(3)
with col1:
    tipo_vivienda = st.selectbox("Tipo de vivienda", tipos_vivienda)
    planta = st.number_input("Planta", min_value=-1, max_value=20, value=2, step=1)
    tamano = st.slider("Tamaño (m²)", 30, 300, 80, step=5)
    habitaciones = st.number_input("Habitaciones", min_value=1, max_value=10, value=2, step=1)
    banos = st.number_input("Baños", min_value=1, max_value=4, value=1, step=1)
with col2:
    ascensor = st.checkbox("Ascensor", value=True)
    terraza = st.checkbox("Terraza", value=True)
    piscina = st.checkbox("Piscina", value=True)
    exterior = st.checkbox("Exterior", value=True)
    trastero = st.checkbox("Trastero", value=True)
    garaje = st.checkbox("Garaje", value=True)
with col3:
    distrito = st.selectbox("Distrito", distritos_visibles)
    ano = st.slider("Año", 2025, 2030, 2026, step=1)
    operacion = st.selectbox("Operacion", ["alquiler", "venta"])

# Distrito correcto
distrito = normaliza_distrito(distrito)

# --- Construir entrada para predicción ---
input_dict = {
    "Tipo_vivienda": tipo_vivienda,
    "Planta": planta,
    "Tamano": tamano,
    "Habitaciones": habitaciones,
    "Banos": banos,
    "Ascensor": ascensor,
    "Terraza": terraza,
    "Piscina": piscina,
    "Exterior": exterior,
    "Trastero": trastero,
    "Garaje": garaje,
    "Ano": ano,
    "Distrito": distrito,
    "Operacion": operacion
}

st.write("📋 **Características seleccionadas:**")
st.json(dict(list(input_dict.items())))

def predecir_precio_vivienda(vivienda:dict, df:pd.DataFrame, tendencias_socio:pd.DataFrame, modelo):
    """
    Predice el precio ajustado de una vivienda específica para un año futuro.

    Parameters
    ----------
    vivienda : dict
        Ejemplo:
        {"Distrito": "CHAMBERI", "Tipo_vivienda": "Chalet", "Habitaciones": 3,
         "Banos": 1, "Planta": 3, "Tamano": 90, "Ano": 2030, ...}

    df : DataFrame
        Dataset histórico con todas las variables hasta el último año real (ej. 2022).

    tendencias_socio : DataFrame
        CSV con columnas de tendencia por distrito (Esperanza_vida, etc.)

    mejor_modelo : modelo entrenado
        Modelo sklearn ya ajustado.
    """
    distrito = vivienda['Distrito']
    ano_objetivo = vivienda['Ano']
    ultimo_ano = df['Ano'].max()

    # --- 1️⃣ Obtener datos históricos del distrito ---
    df_dist = df.copy()
    df_dist = pd.get_dummies(df_dist, drop_first=True)
    df_dist = df_dist[df_dist[f'Distrito_{distrito}'] == 1]
    df_dist = df_dist.sort_values('Ano')
    fila_base = df_dist.iloc[-1]  # último año disponible

    # --- 2️⃣ Actualizar variables socioeconómicas según tendencia ---
    socio = tendencias_socio[tendencias_socio['Distrito'] == distrito].iloc[0]
    delta = ano_objetivo - ultimo_ano

    socio_vars = ['Esperanza_vida', 'Mayores_65anos%', 'Menores_18anos%',
                  'Paro_registrado%', 'Apartamentos_turisticos',
                  'Superficie_distrito_ha', 'Zonas_verdes%']

    for var in socio_vars:
        if var in socio:
            fila_base[var] = fila_base[var] + socio[var] * delta

    # --- 3️⃣ Calcular features temporales del distrito ---
    fila_base['precio_prom_ano_ant'] = df_dist['Precio_ajustado'].iloc[-1]
    fila_base['tendencia_distrito'] = df_dist['Precio_ajustado'].pct_change().iloc[-1]
    fila_base['precio_m2_distrito_ant'] = (df_dist['Precio_ajustado'].iloc[-1] / df_dist['Tamano'].iloc[-1])
    fila_base['ratio_vs_distrito'] = 1.0 # (no se puede calcular)  
    fila_base['volatilidad_distrito'] = df_dist['Precio_ajustado'].std()

    # --- 4️⃣ Rellenar características propias de la vivienda ---
    tipos_vivienda_dummies = ["Ático", "Chalet", "Dúplex", "Estudio",
                  "Tríplex", "Mansión", "Loft"]
    
    for var in vivienda.keys():
        if var not in ['Tipo_vivienda', 'Distrito', 'Operacion']: # que no sean dummies
            fila_base[var] = vivienda[var]
        else:
            if var == 'Tipo_vivienda':
                for tipo in tipos_vivienda_dummies:
                    fila_base[f'Tipo_vivienda_{tipo}'] = 0 # convertir todas a 0
                if vivienda[var] != 'Apartamento': # si existe el dummy, ponerlo en 1
                    fila_base[f'Tipo_vivienda_{vivienda[var]}'] = 1
            if var == 'Operacion':
                fila_base['Operacion_venta'] = 0 if vivienda[var] == 'alquiler' else 1
            if var == 'Distrito':
                for distrito in [normaliza_distrito(x) for x in distritos_visibles]:
                    if distrito != 'ARGANZUELA':
                        fila_base[f'Distrito_{distrito}'] = 0
                if vivienda[var] != 'ARGANZUELA':
                    fila_base[f'Distrito_{vivienda[var]}'] == 1
    
    # Variables is_missing (planta la damos nosotros, las otras son predichas)
    fila_base['Planta_is_missing'] = 0
    for var_missing in ['Mayores_65anos%_is_missing', 'Menores_18anos%_is_missing', 'Paro_registrado%_is_missing']:
        fila_base[var_missing] = 1

    # --- 5️⃣ Crear DataFrame ---
    df_input = pd.DataFrame([fila_base])
    
    # Alinear columnas con las del modelo
    FEATURES = ['Esperanza_vida', 'Mayores_65anos%', 'Menores_18anos%',
       'Paro_registrado%', 'Apartamentos_turisticos', 'Superficie_distrito_ha',
       'Zonas_verdes%', 'Habitaciones', 'Tamano', 'Garaje', 'Trastero',
       'Piscina', 'Terraza', 'Planta', 'Exterior', 'Ascensor', 'Banos',
       'Planta_is_missing', 'Mayores_65anos%_is_missing',
       'Menores_18anos%_is_missing', 'Paro_registrado%_is_missing',
       'precio_prom_ano_ant', 'tendencia_distrito', 'precio_m2_distrito_ant',
       'ratio_vs_distrito', 'volatilidad_distrito', 'Distrito_BARAJAS',
       'Distrito_CARABANCHEL', 'Distrito_CENTRO', 'Distrito_CHAMARTIN',
       'Distrito_CHAMBERI', 'Distrito_CIUDADLINEAL',
       'Distrito_FUENCARRALELPARDO', 'Distrito_HORTALEZA', 'Distrito_LATINA',
       'Distrito_MONCLOAARAVACA', 'Distrito_MORATALAZ',
       'Distrito_PUENTEDEVALLECAS', 'Distrito_RETIRO', 'Distrito_SALAMANCA',
       'Distrito_SANBLASCANILLEJAS', 'Distrito_TETUAN', 'Distrito_USERA',
       'Distrito_VICALVARO', 'Distrito_VILLADEVALLECAS', 'Distrito_VILLAVERDE',
       'Operacion_venta', 'Tipo_vivienda_chalet', 'Tipo_vivienda_dúplex',
       'Tipo_vivienda_estudio', 'Tipo_vivienda_loft', 'Tipo_vivienda_mansión',
       'Tipo_vivienda_tríplex', 'Tipo_vivienda_ático']

    # Asegura que todas las columnas de FEATURES existen en df_input
    for col in FEATURES:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reordena df_input para que tenga exactamente las columnas de FEATURES y en ese orden
    df_input = df_input[FEATURES]

    # --- 6️⃣ Predicción ---
    precio_predicho = modelo.predict(df_input)[0]

    return precio_predicho

# --- Cargar modelo ---
try:
    model_rf_venta, model_rf_alquiler = load_models()
    st.success("Modelos cargados correctamente ✅")
except FileNotFoundError:
    st.error(f"No se encontró el modelo en {MODEL_PATH_VENTA} o {MODEL_PATH_ALQUILER}.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar los modelos: {type(e).__name__}: {e}")
    st.stop()

# --- Botón de predicción ---
if st.button("🔮 Predecir precio"):
    try:
        df_modelado = load_csv("data/final/viviendas_2011_2024.csv")
        df_modelado = df_modelado[df_modelado['Ano'] == 2022].copy()
        df_tendencias = load_csv("data/final/tendencias.csv")

        if input_dict["Operacion"] == 'venta':
            precio_pred = predecir_precio_vivienda(input_dict, df_modelado, df_tendencias, model_rf_venta)
        else:
            precio_pred = predecir_precio_vivienda(input_dict, df_modelado, df_tendencias, model_rf_alquiler)

        st.success(f"💰 Precio estimado: **{precio_pred:,.0f} €**")
        st.caption("Predicción generada con el modelo entrenado de precios ajustados.")

    except Exception as e:
        st.error(f"Error durante la predicción: {type(e).__name__}: {e}")
        print(e)
