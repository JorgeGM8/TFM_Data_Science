import pandas as pd
from pathlib import Path
try:
    from utils.extra_var import unify_files, extract_years, extract_tipo_vivienda, \
                                extract_banos, extract_planta_ext_ascensor
except ModuleNotFoundError:
    print('--> Usa "python -m src.unificacion_datos_idealista_2025" para importar correctamente las funciones.')
    quit()
except ImportError as e:
    print(f'--> Error de importación: {e}')
    quit()
except Exception as e:
    print(f'--> Error inesperado: {type(e).__name__}: {e}')

# ==============
# CONFIGURACIÓN
# ==============
BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw"
PROCESSED = BASE_DIR / "data" / "processed"
FINAL = BASE_DIR / "data" / "final"

# Crear carpetas si no existen
for folder in [RAW, PROCESSED, FINAL]:
    folder.mkdir(parents=True, exist_ok=True)

# ==============
# UNIFICAR ALQUILER Y VENTA Y LIMPIAR DATOS
# ==============
files_alq = sorted(RAW.glob("properties_all_alquiler*.csv"))
files_vta = sorted(RAW.glob("properties_all_venta*.csv"))

df_alq = unify_files(files_alq, "alquiler")
df_vta = unify_files(files_vta, "venta")

# Guardar intermedios
df_alq.to_csv(PROCESSED / "alquiler_unificado.csv", index=False)
df_vta.to_csv(PROCESSED / "venta_unificado.csv", index=False)

# Gestionar columnas
all_cols = sorted(set(df_alq.columns) | set(df_vta.columns))
for df in (df_alq, df_vta):
    for c in all_cols:
        if c not in df.columns:
            df[c] = pd.NA

# Unir venta y alquiler
df_all = pd.concat([df_alq[all_cols], df_vta[all_cols]], ignore_index=True)

# Eliminar duplicados
df_all = df_all.drop_duplicates(subset=["link", "operacion"], keep="first")

# Procesar datos erróneos de tamaño y eliminar las filas incorrectas
df_all['tamanio'] = df_all['tamanio'].str.replace(".", "", regex=False) # Elimina puntos en string
df_all["tamanio"] = pd.to_numeric(df_all["tamanio"], errors="coerce") # Pasa a numérico, si no puede, nulo
df_all = df_all[df_all["tamanio"].notna()] # Elimina filas no numéricas (errores)

# Guardar dataframe intermedio
df_all.to_csv(PROCESSED / "inmuebles_unificado_total.csv", index=False)

print('--> Datos de alquiler y venta unificados y limpiados.')
print(f'--> Datos intermedios guardados en "{PROCESSED}/inmuebles_unificado_total.csv"')

# ==============
# VARIABLES EXTRA
# ==============

# Revisar si aparecen los términos en la descripción y añadir True si sale y False si no sale.
df_all["garaje"] = df_all["descripcion_larga"].str.lower().str.contains(r"\b(?:garaje|cochera)\b", na=False)
df_all["trastero"] = df_all["descripcion_larga"].str.lower().str.contains(r"\btrastero\b", na=False)
df_all["piscina"] = df_all["descripcion_larga"].str.lower().str.contains(r"\b(?:piscina|pizcina)\b", na=False)
df_all["terraza"] = df_all["descripcion_larga"].str.lower().str.contains(r"\b(?:terraza|terrasa|balcon|balcón)\b", na=False)

# Extraer datos de planta, exterior/interior y con/sin ascensor
df_all[["planta", "exterior", "ascensor"]] = df_all["descripcion"].apply(
    lambda x: pd.Series(extract_planta_ext_ascensor(x))
)

# Extraer datos de año de construcción y/o reforma, si aparecen
df_all[["anio_construccion","anio_reforma"]] = df["descripcion_larga"].apply(
    lambda x: pd.Series(extract_years(x))
)

# Extraer datos de tipo de vivienda (mansión, dúplex, chalet, apartamento...)
df_all["tipo_vivienda"] = df_all["descripcion_larga"].apply(extract_tipo_vivienda)

# Extraer datos de número de baños
df_all["banos"] = df_all["descripcion_larga"].apply(extract_banos)

print('--> Variables extra añadidas (garaje, trastero, piscina, terraza,' \
'año construcción, año reforma, planta, exterior y ascensor).')

# ==============
# BORRADO DE VARIABLES INNECESARIAS
# ==============
col_innecesarias = ['descripcion', 'descripcion_larga', 'link', 'localidad']

df_all = df_all.drop(columns=col_innecesarias)

print(f'--> Variables sin uso eliminadas {col_innecesarias}.')

# ==============
# GUARDADO FINAL
# ==============
df_all.to_csv(FINAL / "inmuebles_unificado_total_final.csv", index=False)

print(f'--> Proceso completado. Archivo guardado en: {FINAL}/inmuebles_unificado_total_final.csv')