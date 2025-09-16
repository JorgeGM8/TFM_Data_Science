import pandas as pd
from pathlib import Path
try:
    from utils.extra_var import unify_files, extract_years, extract_tipo_vivienda, extract_banos
except ModuleNotFoundError:
    print('Usa "python -m src.unificacion_var_extra_idealista" para importar correctamente las funciones.')
    quit()
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
# UNIFICAR ALQUILER Y VENTA
# ==============

files_alq = sorted(RAW.glob("properties_all_alquiler*.csv"))
files_vta = sorted(RAW.glob("properties_all_venta*.csv"))

df_alq = unify_files(files_alq, "alquiler")
df_vta = unify_files(files_vta, "venta")

# Guardar intermedios
df_alq.to_csv(PROCESSED / "alquiler_unificado.csv", index=False)
df_vta.to_csv(PROCESSED / "venta_unificado.csv", index=False)

# ==============
# UNIFICAR_TODO
# ==============
all_cols = sorted(set(df_alq.columns) | set(df_vta.columns))
for df in (df_alq, df_vta):
    for c in all_cols:
        if c not in df.columns:
            df[c] = pd.NA

df_all = pd.concat([df_alq[all_cols], df_vta[all_cols]], ignore_index=True)
df_all = df_all.drop_duplicates(subset=["link", "operacion"], keep="first")
df_all.to_csv(PROCESSED / "inmuebles_unificado_total.csv", index=False)

# ==============
# VARIABLES EXTRA
# ==============

# Revisa si aparecen los términos en la descripción y añade True si sale y False si no sale.
df_all["garaje"] = df_all["descripcion_larga"].str.lower().str.contains(r"\b(?:garaje|cochera)\b", na=False)
df_all["trastero"] = df_all["descripcion_larga"].str.lower().str.contains(r"\btrastero\b", na=False)
df_all["piscina"] = df_all["descripcion_larga"].str.lower().str.contains(r"\b(?:piscina|pizcina)\b", na=False)
df_all["terraza"] = df_all["descripcion_larga"].str.lower().str.contains(r"\b(?:terraza|terrasa|balcon|balcón)\b", na=False)

# Aplicar al dataframe
df[["anio_construccion","anio_reforma"]] = df["descripcion_larga"].apply(
    lambda x: pd.Series(extract_years(x))
)
df_all["tipo_vivienda"] = df_all["descripcion_larga"].apply(extract_tipo_vivienda)
df_all["banos"] = df_all["descripcion_larga"].apply(extract_banos)

# ==============
# GUARDADO FINAL
# ==============
df_all.to_csv(FINAL / "inmuebles_unificado_total_final.csv", index=False)

print("Proceso completado. Archivos en:", FINAL)