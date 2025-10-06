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
# UNIFICAR ALQUILER Y VENTA Y LIMPIAR DATOS
# ==============
while True:
    try:
        capital = int(input("¿Datos de Madrid (1) o periféricos (2)?: "))
        capital = (capital == 1) # Convertir en True si se eligió 1
        break
    except ValueError:
        print('Introduce un número correcto.')

if capital:
    files_alq = sorted(RAW.glob("properties_all_alquiler*.csv"))
    files_vta = sorted(RAW.glob("properties_all_venta*.csv"))
else:
    files_alq = sorted(RAW.glob("properties_all_perifericos_alquiler*.csv"))
    files_vta = sorted(RAW.glob("properties_all_perifericos_venta*.csv"))

df_alq = unify_files(files_alq, "alquiler")
df_vta = unify_files(files_vta, "venta")

# Guardar intermedios
if capital:
    df_alq.to_csv(PROCESSED / "alquiler_unificado.csv", index=False)
    df_vta.to_csv(PROCESSED / "venta_unificado.csv", index=False)
else:
    df_alq.to_csv(PROCESSED / "alquiler_perifericos_unificado.csv", index=False)
    df_vta.to_csv(PROCESSED / "venta_perifericos_unificado.csv", index=False)

# Gestionar columnas
all_cols = sorted(set(df_alq.columns) | set(df_vta.columns))
for df in (df_alq, df_vta):
    for c in all_cols:
        if c not in df.columns:
            df[c] = pd.NA

# Unir venta y alquiler
df_all = pd.concat([df_alq[all_cols], df_vta[all_cols]], ignore_index=True)

# Eliminar duplicados
df_all = df_all.drop_duplicates(subset=["Link", "Operacion"], keep="first")

# Eliminar duplicados de viviendas que puedan tener más de una inmobiliaria (normalmente las más caras)
df_menores = df_all[df_all['Precio'] <= 3000000]  # Los que NO cumplen la condición
df_mayores_sin_duplicados = (df_all[df_all['Precio'] > 3000000]
                           .groupby(['Distrito', 'Precio'])
                           .first()
                           .reset_index())

# Combinar de nuevo
df_all = df_all.infer_objects()
df_all = pd.concat([df_menores, df_mayores_sin_duplicados], ignore_index=True)

# Procesar datos erróneos de tamaño y eliminar las filas incorrectas
df_all['Tamano'] = df_all['Tamano'].str.replace(".", "", regex=False) # Elimina puntos en string
df_all["Tamano"] = pd.to_numeric(df_all["Tamano"], errors="coerce") # Pasa a numérico, si no puede, nulo
df_all = df_all[df_all["Tamano"].notna()] # Elimina filas no numéricas (errores)

# Guardar dataframe intermedio
if capital:
    df_all.to_csv(PROCESSED / "inmuebles_unificado_total.csv", index=False)
    print('--> Datos de alquiler y venta unificados y limpiados.')
    print(f'--> Datos intermedios guardados en "{PROCESSED}/inmuebles_unificado_total.csv"')
else:
    df_all.to_csv(PROCESSED / "inmuebles_perifericos_unificado_total.csv", index=False)
    print('--> Datos de alquiler y venta unificados y limpiados.')
    print(f'--> Datos intermedios guardados en "{PROCESSED}/inmuebles_perifericos_unificado_total.csv"')


# ==============
# VARIABLES EXTRA
# ==============

# Revisar si aparecen los términos en la descripción y añadir True si sale y False si no sale.
df_all["Garaje"] = df_all["Descripcion_larga"].str.lower().str.contains(r"\b(?:garaje|cochera)\b", na=False)
df_all["Trastero"] = df_all["Descripcion_larga"].str.lower().str.contains(r"\btrastero\b", na=False)
df_all["Piscina"] = df_all["Descripcion_larga"].str.lower().str.contains(r"\b(?:piscina|pizcina)\b", na=False)
df_all["Terraza"] = df_all["Descripcion_larga"].str.lower().str.contains(r"\b(?:terraza|terrasa|balcon|balcón)\b", na=False)

# Extraer datos de planta, exterior/interior y con/sin ascensor
df_all[["Planta", "Exterior", "Ascensor"]] = df_all["Descripcion"].apply(
    lambda x: pd.Series(extract_planta_ext_ascensor(x))
)

# Extraer datos de año de construcción y/o reforma, si aparecen
df_all[["Ano_construccion","Ano_reforma"]] = df["Descripcion_larga"].apply(
    lambda x: pd.Series(extract_years(x))
)

# Extraer datos de tipo de vivienda (mansión, dúplex, chalet, apartamento...)
df_all["Tipo_vivienda"] = df_all["Descripcion_larga"].apply(extract_tipo_vivienda)

# Extraer datos de número de baños
df_all["Banos"] = df_all["Descripcion_larga"].apply(extract_banos)

print('--> Variables extra añadidas (garaje, trastero, piscina, terraza,' \
'año construcción, año reforma, planta, exterior y ascensor).')

# ==============
# BORRADO DE VARIABLES INNECESARIAS
# ==============
col_innecesarias = ['Descripcion', 'Descripcion_larga', 'Link', 'Localidad']

df_all = df_all.drop(columns=col_innecesarias)

print(f'--> Variables sin uso eliminadas {col_innecesarias}.')

# ==============
# GUARDADO FINAL
# ==============
if capital:
    df_all.to_csv(FINAL / "inmuebles_unificado_total_final.csv", index=False)
    print(f'--> Proceso completado. Archivo guardado en: {FINAL}/inmuebles_unificado_total_final.csv')
else:
    df_all.to_csv(FINAL / "inmuebles_perifericos_unificado_total_final.csv", index=False)
    print(f'--> Proceso completado. Archivo guardado en: {FINAL}/inmuebles_perifericos_unificado_total_final.csv')