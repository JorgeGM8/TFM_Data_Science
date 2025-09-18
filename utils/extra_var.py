from pathlib import Path
import pandas as pd
from utils.cleaning import normaliza_distrito
import re

# Leer con seguridad archivos csv (usado solo para unificacion variables extra)
def read_csv_safely(p: Path) -> pd.DataFrame:
    """
    Abrir y leer csv de forma segura.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc, thousands=".")
        except Exception:
            pass
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc, sep=";", thousands=".")
        except Exception:
            pass
    raise RuntimeError(f"No pude leer {p}")

# Normalizar columnas de dataframe
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalización de columnas de dataframe para cambiar los nombres adecuadamente.
    """
    cols_map = {
        "precio":"Precio",
        "localidad":"Localidad",
        "tamaño":"Tamano",
        "tamano":"Tamano",
        "tamanio":"Tamano",
        "habitaciones":"Habitaciones",
        "descripcion":"Descripcion",
        "link":"Link",
        "descripcion_larga":"Descripcion_larga",
        "distrito":"Distrito",
    }
    expected = ["Precio","Localidad","Tamano","Habitaciones","Descripcion","Link","Descripcion_larga","Distrito"]
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={c: cols_map.get(c, c) for c in df.columns})
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA
    return df[expected]

def unify_files(files, operacion):
    frames = []
    for p in files:
        df = read_csv_safely(p)
        df = normalize_columns(df)
        df["Distrito"] = df["Distrito"].apply(normaliza_distrito)
        df["Operacion"] = operacion
        frames.append(df)

    if not frames:
        print(f"⚠ No se encontraron archivos para {operacion}")
        return pd.DataFrame()  # DataFrame vacío
    return pd.concat(frames, ignore_index=True)

# Regex para año: busca un número de 4 cifras entre 1850 y 2029, y revisa si se trata de reforma o de construcción
def extract_years(text):
    if not isinstance(text, str):
        return None, None
    text_l = text.lower()
    # Buscar años
    years = re.findall(r"(18[5-9]\d|19[0-9]\d|20[0-2]\d)", text_l) # Fechas entre 1850 y 2029
    anio_construccion = None
    anio_reforma = None
    for y in years:
        # Se coge contexto +- 15 caracteres
        idx = text_l.find(y)
        context = text_l[max(0, idx-15): idx+15]
        # Si sale algo de esto, se añade a año de construcción
        if any(w in context for w in ["constru", "edifica", "obra nueva", "antig", "data de"]):
            anio_construccion = int(y)
        # Si sale algo de esto, se añade a año de reforma
        elif any(w in context for w in ["reforma", "rehabil"]):
            anio_reforma = int(y)
    return anio_construccion, anio_reforma

# Regex para tipo de vivienda: busca si aparece algún tipo y lo añade
def extract_tipo_vivienda(text):
    # Posibles tipos de vivienda
    TYPES = [
        "mansión", "tríplex", "dúplex",
        "ático", "apartamento", "chalet",
        "estudio", "loft", "piso"
    ]
    if not isinstance(text, str):
        return None
    text_l = text.lower()
    found = [t for t in TYPES if re.search(rf"\b{t}\b", text_l)]
    return found[0] if found else None

# Regex para baños: busca si sale información de número de baños y los añade
def extract_banos(text):
    if not isinstance(text, str):
        return 1
    text_l = text.lower()
    # Busca "1 baño", "2 baños", "un aseo", etc.
    match = re.search(r"(\d+)\s*(bañ?os?|aseos?|retretes?)", text_l)
    if match:
        return int(match.group(1)) # Añade el número de baños
    if re.search(r"\bun\s*(bañ?o|bano|aseo|retrete)\b", text_l):
        return 1 # Añade un baño
    return 1 # Si no pone nada, se da por hecho que tiene un baño

# Seleccionar planta, si es exterior y si tiene ascensor.
def extract_planta_ext_ascensor(text: str):
    """
    Función que detecta en la columna "descripción" el número de planta, si es exterior o interior y si tiene ascensor.
    """
    # Normalizar a minúsculas
    try:
        t = str(text).lower()
    except Exception:
        return None, False, False
    
    # --- Planta ---
    planta = None
    # Casos especiales
    if "semi-sótano" in t or "semisotano" in t or "sótano" in t or "sotano" in t:
        planta = -1
    elif "entreplanta" in t or "bajo" in t:
        planta = 0
    else:
        # Buscar "planta <n>ª" con regex
        m = re.search(r"planta\s*(-?\d+)", t)
        if m:
            num = int(m.group(1))
            if num == -1:
                planta = 0
            else:
                planta = num
    
    # --- Interior / Exterior ---
    if "exterior" in t:
        exterior = True
    elif "interior" in t:
        exterior = False
    else:
        exterior = False
    
    # --- Ascensor ---
    if "sin ascensor" in t:
        ascensor = False
    elif "con ascensor" in t:
        ascensor = True
    else:
        ascensor = False
    
    return planta, exterior, ascensor