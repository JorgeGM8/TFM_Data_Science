import pandas as pd
import unicodedata
import re

# Función para eliminar comillas y convertir formato numérico europeo a americano
def convertir_formato_europeo(col):
    col = col.astype(str).str.replace(r'[\"\']', '', regex=True).str.strip()  # Quita comillas y espacios
    col = col.str.replace('.', '', regex=False)  # Quita puntos de miles
    col = col.str.replace(',', '.', regex=False)  # Coma decimal a punto
    return pd.to_numeric(col, errors='coerce')

# Función para normalizar los nombres de distritos
def normaliza_distrito(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).lower()
    # Quitar prefijo de numeración tipo "01. Centro"
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)
    # Quitar tildes
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    # Dejar solo letras (incluye Ñ/ñ); elimina números, espacios y símbolos
    s = re.sub(r"[^A-Za-zÑñ]", "", s)
    # Mapeo manual de casos especiales
    replace_map = {
        "barriodesalamanca": "salamanca",
        "fuencarral": "fuencarralelpardo",
        "moncloa": "moncloaaravaca",
        "sanblas": "sanblascanillejas"
    }
    if s in replace_map:
        s = replace_map[s]
    # Mayúsculas
    return s.upper()