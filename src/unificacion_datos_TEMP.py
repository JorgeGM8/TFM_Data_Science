import pandas as pd

#Carga de datasets
general = pd.read_csv("general_data.csv")  
barrios  = pd.read_excel("Barrios.xlsx") 

#Columnas necesarias
barrios_info = barrios[["NOMBRE", "NOMDIS"]].rename(columns={"NOMBRE":"Barrio","NOMDIS":"Distrito"})

# Columna de distrito incluida
general_distrito = general.merge(barrios_info, on="Barrio")

print(general_distrito.head())

general_distrito

# Poner Distrito antes de Barrio
cols = general_distrito.columns.tolist()

# Mover Distrito a la posición 0
cols.insert(cols.index("Barrio"), cols.pop(cols.index("Distrito")))

# Reordenar dataframe
general_distrito = general_distrito[cols]

print(general_distrito.head())

general_distrito.to_csv("general_con_distrito.csv", index=False)

general_distrito
# Estandarizar columnas en general_con_distrito

import pandas as pd

general = pd.read_csv("general_con_distrito.csv")

general

import pandas as pd
import unicodedata, re

#Cargar CSV
df = pd.read_csv("general_con_distrito.csv", encoding="utf-8")

# Normalizar encabezados
df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]

# Renombrar a 'Distrito'
for c in df.columns:
    if c.lower().strip() == "distrito":
        df = df.rename(columns={c: "Distrito"})

# Función de estandarización
def normaliza_valor_distrito(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x)
    # quitar numeración tipo "01. Centro"
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)
    # quitar tildes
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    # dejar solo letras (incluye Ñ)
    s = re.sub(r"[^A-Za-zÑñ]", "", s)
    # mayúsculas
    return s.upper()

# Crear columna estandarizada
df["Distrito"] = df["Distrito"].apply(normaliza_valor_distrito)

df.to_csv("general_con_distrito_std.csv", index=False, encoding="utf-8-sig")
print("Archivo limpio guardado como general_con_distrito_std.csv")


# Estandarizar columnas en alquiler_venta_procesado

import pandas as pd
import unicodedata
import re

# Cargar CSV
try:
    df = pd.read_csv("venta_alquiler_procesado.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("venta_alquiler_procesado.csv", encoding="cp1252")

# Normalizar encabezados
df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]

# Asegurar nombre 'Distrito'
for c in df.columns:
    if c.lower().strip() == "distrito" and c != "Distrito":
        df = df.rename(columns={c: "Distrito"})
        break

if "Distrito" not in df.columns:
    raise KeyError("No se encontró una columna llamada 'Distrito' (o equivalente).")

# Estandarización de valores de Distrito
def normaliza_distrito(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x)
    # Quitar prefijo de numeración tipo "01. Centro"
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)
    # Quitar tildes
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    # Dejar solo letras (incluye Ñ/ñ); elimina números, espacios y símbolos
    s = re.sub(r"[^A-Za-zÑñ]", "", s)
    # Mayúsculas
    return s.upper()

# Aplicar estandarización
df["Distrito"] = df["Distrito"].apply(normaliza_distrito)

# Unificar nombre de año 
df = df.rename(columns={"Año": "anyo", "año": "anyo"})

df.to_csv("venta_alquiler_procesado_std.csv", index=False, encoding="utf-8-sig")

print("Ejemplos de Distrito estandarizado:", df["Distrito"].dropna().unique()[:10])


# ## Unificacion de dataset general_con_distrito_std con venta_alquiler_procesado_std ##

import pandas as pd

# Cargar datasets ya estandarizados
general = pd.read_csv("general_con_distrito_std.csv", encoding="utf-8")
precios = pd.read_csv("venta_alquiler_procesado_std.csv", encoding="utf-8")

# Renombrar columnas de año 
general = general.rename(columns={"Año":"anyo", "año":"anyo"})
precios = precios.rename(columns={"Año":"anyo", "año":"anyo"})

# Replicar por cada barrio de su distrito
mapa_barrios = general[["Distrito","Barrio"]].drop_duplicates()
precios_expandido = precios.merge(mapa_barrios, on="Distrito", how="left")

base_2020_2024 = general.merge(
    precios_expandido[["anyo","Distrito","Barrio","Precio_venta","Precio_alquiler"]],
    on=["anyo","Distrito","Barrio"], how="left"
)

precios_2011_2019 = (
    precios_expandido.loc[precios_expandido["anyo"] < 2020,
                          ["anyo","Distrito","Barrio","Precio_venta","Precio_alquiler"]]
    .copy()
)

# Agregar las columnas de general como vacías
faltan_cols = [c for c in base_2020_2024.columns if c not in precios_2011_2019.columns]
for c in faltan_cols:
    precios_2011_2019[c] = pd.NA

# Reordenar columnas
precios_2011_2019 = precios_2011_2019[base_2020_2024.columns]

# Concatenar ambos bloques
dataset_final = pd.concat([precios_2011_2019, base_2020_2024], ignore_index=True)

dataset_final.to_csv("dataset_general_2011_2024.csv", index=False, encoding="utf-8-sig")

print("Años incluidos:", dataset_final["anyo"].min(), "-", dataset_final["anyo"].max())
print("Filas totales:", len(dataset_final))
print("Ejemplo:\n", dataset_final.head())


# ## Unificar con dataset de esperanza de vida ##

import pandas as pd
import unicodedata, re

# Normalizar nombres de distrito
def norm_distrito(s):
    if pd.isna(s):
        return s
    s = str(s)
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)  # quitar numeración tipo "01. "
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-zÑñ]", "", s)  # solo letras
    return s.upper()

general = pd.read_csv("dataset_general_2011_2024.csv", encoding="utf-8")
ev = pd.read_csv("esperanza_vida.csv", sep=";", encoding="utf-8-sig")

# Normalizar columnas de esperanza de vida, renombrando la columna año
ev = ev.rename(columns={"Año":"anyo", "año":"anyo"})

# Limpiar columna Distrito
ev["Distrito"] = ev["Distrito"].map(norm_distrito)

# Convertir a numérico la esperanza de vida
ev["esperanza_vida"] = ev["Esperanza de vida"].astype(str).str.replace(",", ".")
ev["esperanza_vida"] = pd.to_numeric(ev["esperanza_vida"], errors="coerce")

# MFiltrar columnas necesarias
ev = ev[["anyo","Distrito","esperanza_vida"]]

# Normalizar distritos en la base general 
general["Distrito"] = general["Distrito"].map(norm_distrito)

# Unir datasets (anyo + Distrito)
merged = general.merge(ev, on=["anyo","Distrito"], how="left")

merged.to_csv("dataset_general_2011_2024_ev.csv", index=False, encoding="utf-8-sig")

print("Dataset unificado guardado en dataset_general_2011_2024_ev.csv")
print("Columnas:", merged.columns.tolist()[:10], "...")
print("Ejemplo:\n", merged.head())


# Estandarización del dataset de renta_media_persona

import pandas as pd
import unicodedata, re

# Cargar  del archivo con tolerancia de encoding y separador ;
try:
    df = pd.read_csv("renta_media.csv", sep=";", encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv("renta_media.csv", sep=";", encoding="cp1252")

# Limpiar encabezados quitando ';', espacios raros
df.columns = [c.strip().replace(";", "") for c in df.columns]

# Unificar columna año
for cand in ("Año", "año", "Ano", "ano"):
    if cand in df.columns:
        df = df.rename(columns={cand: "anyo"})
        break

# Unificar columna 'Distrito'
for c in df.columns:
    if c.lower().strip() == "distrito" and c != "Distrito":
        df = df.rename(columns={c: "Distrito"})
        break
if "Distrito" not in df.columns:
    raise KeyError("No se encontró la columna 'Distrito'.")

# Normalizar valores de Distrito: sin números, sin espacios/símbolos, sin tildes, ni mayúsculas 
def normaliza_distrito(x: str) -> str:
    if pd.isna(x): return x
    s = str(x)
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)  # quita prefijo tipo '01. '
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))  # sin tildes
    s = re.sub(r"[^A-Za-zÑñ]", "", s)       # solo letras (quita espacios, guiones, números, etc.)
    return s.upper()

df["Distrito"] = df["Distrito"].map(normaliza_distrito)

df.to_csv("renta_media_procesado_std.csv", index=False, encoding="utf-8-sig")
print("Guardado renta_media_procesado_std.csv")
print(df[["anyo","Distrito"]].head())


# ## Unificación del dataset renta_media_persona con el dataset_general_2011_2024_ev ##

import pandas as pd
import unicodedata, re

# Normalizar columna 'Distrito' 
def norm_distrito(s):
    if pd.isna(s): return s
    s = str(s)
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)                     # quita "01. "
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(ch))               # sin tildes
    s = re.sub(r"[^A-Za-zÑñ]", "", s)                           # solo letras
    return s.upper()

base = pd.read_csv("dataset_general_2011_2024_ev.csv", encoding="utf-8")
renta = pd.read_csv("renta_media_procesado_std.csv", encoding="utf-8-sig")

# Unificar columna 'Año'
base = base.rename(columns={"Año":"anyo", "año":"anyo"})
renta = renta.rename(columns={"Año":"anyo", "año":"anyo"})

# normalizar distritos en ambos 
base["Distrito"]  = base["Distrito"].map(norm_distrito)
renta["Distrito"] = renta["Distrito"].map(norm_distrito)

# Convertir a numérico la columna renta
col_renta = next((c for c in renta.columns if "renta" in c.lower() and c not in ["anyo","Distrito"]), None)
if col_renta is None:
    # si no tiene "renta" en el nombre, toma cualquier columna distinta a claves
    col_renta = [c for c in renta.columns if c not in ["anyo","Distrito"]][0]

# limpiar formato de símbolos y pasar a float
renta_val = (renta[col_renta].astype(str)
             .str.replace(r"[^0-9\-,\.]", "", regex=True)
             .str.replace(".", "", regex=False)
             .str.replace(",", ".", regex=False))
renta["renta_media_persona"] = pd.to_numeric(renta_val, errors="coerce")

renta_std = renta[["anyo","Distrito","renta_media_persona"]].copy()

# Unir por anyo + Distrito
merged = base.merge(renta_std, on=["anyo","Distrito"], how="left")

merged.to_csv("general_11_24_ev_renta.csv", index=False, encoding="utf-8-sig")

print("Guardado: general_11_24_ev_renta.csv")
print("Filas:", len(merged))
print("Años:", merged["anyo"].min(), "-", merged["anyo"].max())
print("Rentas faltantes tras el merge:", merged["renta_media_persona"].isna().sum())
print(merged[["anyo","Distrito","Barrio","renta_media_persona"]].head())


# Estandarización y procesado de datos de datos demograficos INE 

import pandas as pd
import unicodedata, re

# Función para leer CSV aunque tengan distinto separador o codificación
def leer_csv_flexible(path):
    try:
        return pd.read_csv(path, engine="python", sep=None, encoding="utf-8-sig")
    except Exception:
        for enc in ["utf-8-sig", "utf-8", "cp1252"]:
            for sep in [";", ",", "\t", "|"]:
                try:
                    return pd.read_csv(path, sep=sep, encoding=enc)
                except Exception:
                    pass
    raise RuntimeError(f"No pude leer {path} con separadores/codificaciones comunes")

df = leer_csv_flexible("datos_demograficos_ine_procesado.csv")

# limpiar encabezados
df.columns = [c.replace(";", "").strip() for c in df.columns]

# unificar la columna 'Año' por anyo
for cand in ("Año", "año", "Ano", "ano", "year"):
    if cand in df.columns:
        df = df.rename(columns={cand: "anyo"})
        break

# Renombrar columna Distrito
cand_distrito = [c for c in df.columns if c.lower().strip() == "distrito"]
if cand_distrito:
    if cand_distrito[0] != "Distrito":
        df = df.rename(columns={cand_distrito[0]: "Distrito"})
else:
    raise KeyError(f"No se encontró la columna 'Distrito'. Columnas: {df.columns.tolist()}")

# Normalizar valores de la columna 'Distrito'
def normaliza_distrito(x):
    if pd.isna(x): return x
    s = str(x)
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)                         # quita prefijo tipo '01. '
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(ch))                   # sin tildes
    s = re.sub(r"[^A-Za-zÑñ]", "", s)                               # solo letras (sin espacios, números, guiones…)
    return s.upper()

df["Distrito"] = df["Distrito"].map(normaliza_distrito)

# normalizar 'anyo' a numérico
if "anyo" in df.columns:
    df["anyo"] = pd.to_numeric(df["anyo"], errors="coerce").astype("Int64")

df.to_csv("datos_demograficos_ine_std.csv", index=False, encoding="utf-8-sig")
print("Guardado: datos_demograficos_ine_std.csv")
print(df[["anyo","Distrito"]].head())


# ## Unificar con dataset general_11_24_ev_renta ##

import pandas as pd
import unicodedata, re

general = pd.read_csv("general_11_24_ev_renta.csv", encoding="utf-8-sig")
demog   = pd.read_csv("datos_demograficos_ine_std.csv", encoding="utf-8-sig")

# Asegurar columnas claves
general = general.rename(columns={"Año": "anyo", "año": "anyo"})
demog   = demog.rename(columns={"Año": "anyo", "año": "anyo"})

# Normalizar valores de distrito 
def norm_distrito(s):
    if pd.isna(s): return s
    s = str(s)
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)  # quitar numeración "01. "
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(ch))  # quitar tildes
    s = re.sub(r"[^A-Za-zÑñ]", "", s)       # solo letras
    return s.upper()

general["Distrito"] = general["Distrito"].map(norm_distrito)
demog["Distrito"]   = demog["Distrito"].map(norm_distrito)

# Merge por anyo + Distrito
merged = general.merge(demog, on=["anyo","Distrito"], how="left")

merged.to_csv("general_11_24_ev_renta_demog.csv", index=False, encoding="utf-8-sig")

print("Guardado: general_11_24_ev_renta_demog.csv")
print("Filas:", len(merged))
print("Columnas:", len(merged.columns))
print(merged.head())


# Estandarizacion del dataset paro_registrado_2012_2024_total
# 

import pandas as pd
import unicodedata, re

# Cargar CSV
try:
    df = pd.read_csv("paro_registrado_2012_2024_total.csv", sep=";", encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv("paro_registrado_2012_2024_total.csv", sep=";", encoding="cp1252")

# Limpiar encabezados
df.columns = [c.replace(";", "").strip() for c in df.columns]

# Unificar columna año -> anyo
for cand in ("Año", "año", "Ano", "ano"):
    if cand in df.columns:
        df = df.rename(columns={cand: "anyo"})
        break

# Asegurar columna Distrito
for c in df.columns:
    if c.lower().strip() == "distrito" and c != "Distrito":
        df = df.rename(columns={c: "Distrito"})
        break
if "Distrito" not in df.columns:
    raise KeyError(f"No se encontró columna 'Distrito'. Columnas: {df.columns.tolist()}")

# Normalizar valores de Distrito
def normaliza_distrito(x: str) -> str:
    if pd.isna(x): return x
    s = str(x)
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)  # quita prefijos "01. "
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))  # sin tildes
    s = re.sub(r"[^A-Za-zÑñ]", "", s)       # solo letras (quita espacios, guiones, números…)
    return s.upper()

df["Distrito"] = df["Distrito"].map(normaliza_distrito)

# Normalizar año a numérico
if "anyo" in df.columns:
    df["anyo"] = pd.to_numeric(df["anyo"], errors="coerce").astype("Int64")

df.to_csv("paro_registrado_2012_2024_std.csv", index=False, encoding="utf-8-sig")

print("Guardado: paro_registrado_2012_2024_std.csv")
print(df[["anyo","Distrito"]].head())


# Cálculo de promedio anual de paro registrado

import pandas as pd
import re

# Carga de CSV
paro = pd.read_csv("paro_registrado_2012_2024_std.csv", encoding="utf-8-sig")

# Asegurar nombres clave
paro = paro.rename(columns={"Año":"anyo", "año":"anyo", "Mes":"mes", "mes":"mes"})
paro_cols = [c for c in paro.columns if "paro" in c.lower() and c not in ["anyo","Distrito","mes"]]
if not paro_cols:
    raise KeyError(f"No se encontró columna de paro. Columnas disponibles: {paro.columns.tolist()}")

col_paro = paro_cols[0]

# Convertir la columna a numérico
def to_float_eu(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(r"[^0-9\-,\.]", "", regex=True)  # quitar símbolos extra
    s = s.str.replace(".", "", regex=False)            # quitar separador de miles
    s = s.str.replace(",", ".", regex=False)           # coma decimal → punto
    return pd.to_numeric(s, errors="coerce")

paro[col_paro] = to_float_eu(paro[col_paro])

# Calcular promedio anual
paro_anual = (
    paro.groupby(["anyo","Distrito"], as_index=False)[col_paro]
        .mean(numeric_only=True)
        .rename(columns={col_paro: "paro_registrado_anual"})
)

paro_anual.to_csv("paro_registrado_anual.csv", index=False, encoding="utf-8-sig")

print("Archivo guardado como paro_registrado_anual.csv")
print(paro_anual.info())
print(paro_anual.head())


# ## Unificacion con dataset paro_registrado_anual con el dataset general_11_24_ev_renta_demog ##
# 

import pandas as pd
import unicodedata, re

# Leer datsets
general = pd.read_csv("general_11_24_ev_renta_demog.csv", encoding="utf-8-sig")
paro    = pd.read_csv("paro_registrado_anual.csv", encoding="utf-8-sig")

# Asegurar columna anyo
general = general.rename(columns={"Año":"anyo", "año":"anyo"})
paro    = paro.rename(columns={"Año":"anyo", "año":"anyo"})

# Normalizar columna 'Distrito'
def norm_distrito(s):
    if pd.isna(s): return s
    s = str(s)
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)                           # quitar numeración tipo 01.
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(ch))                     # quitar tildes
    s = re.sub(r"[^A-Za-zÑñ]", "", s)                                 # solo letras
    return s.upper()

general["Distrito"] = general["Distrito"].map(norm_distrito)
paro["Distrito"]    = paro["Distrito"].map(norm_distrito)

# Merge por anyo + Distrito
merged = general.merge(paro, on=["anyo","Distrito"], how="left")

merged.to_csv("general_11_24_ev_renta_demog_paro.csv", index=False, encoding="utf-8-sig")

print("Guardado: general_11_24_ev_renta_demog_paro.csv")
print("Columnas totales:", len(merged.columns))
print("Ejemplo:\n", merged[["anyo","Distrito","paro_registrado_anual"]].head())


# Estandarización de columnas de poblacion del dataset poblacion_2011_2022_std

import pandas as pd

pobl = pd.read_csv("poblacion_2011_2022_std.csv", encoding="utf-8-sig")

col = "poblacion_anual"

# limpiar y convertir a float
pobl[col] = (pobl[col].astype(str)
             .str.replace(r"[^0-9\-\.,]", "", regex=True)  # quita símbolos
             .str.replace(".", "", regex=False)            # miles
             .str.replace(",", ".", regex=False))          # coma->punto
pobl[col] = pd.to_numeric(pobl[col], errors="coerce")

# Pasar a entero
pobl[col] = pobl[col].round(0).astype("Int64")

pobl.to_csv("poblacion_2011_2022_std.csv", index=False, encoding="utf-8-sig")

print("Actualizado sin decimales en poblacion_anual")
print(pobl.dtypes)
print(pobl.head())


# ## Unificacion con el dataset poblacion_2011_2022_std con general_11_24_ev_renta_demog_paro ##
# 

import pandas as pd
import re, unicodedata

# helper para normalizar distrito 
def norm_distrito(s):
    if pd.isna(s): return s
    s = str(s)
    s = re.sub(r"^\s*\d{1,2}\.\s*", "", s)                     # quita "01. "
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(ch))               # sin tildes
    s = re.sub(r"[^A-Za-zÑñ]", "", s)                           # solo letras
    return s.upper()

# Cargar CSV
base = pd.read_csv("general_11_24_ev_renta_demog_paro.csv", encoding="utf-8-sig")
pobl = pd.read_csv("poblacion_2011_2022_std.csv", encoding="utf-8-sig")

# Asegurar nombres clave y formato coherente
base = base.rename(columns={"Año":"anyo", "año":"anyo"})
pobl = pobl.rename(columns={"Año":"anyo", "año":"anyo"})

base["Distrito"] = base["Distrito"].map(norm_distrito)
pobl["Distrito"] = pobl["Distrito"].map(norm_distrito)

# Asegurar que el nombre de la columna 'poblacion_anual' 
if "poblacion_anual" not in pobl.columns:
    cand = next(c for c in pobl.columns if "pobl" in c.lower() or "total" in c.lower())
    pobl = pobl.rename(columns={cand: "poblacion_anual"})

# Asegurar que sea numérica
pobl["poblacion_anual"] = pd.to_numeric(pobl["poblacion_anual"], errors="coerce")

# si hay duplicados los consolidamos con media
pobl = (pobl.groupby(["anyo","Distrito"], as_index=False)["poblacion_anual"]
            .mean(numeric_only=True))

# Merge
merged = base.merge(pobl, on=["anyo","Distrito"], how="left")

merged.to_csv("general_11_24_ev_renta_demog_paro_pobl.csv", index=False, encoding="utf-8-sig")

print("Guardado: general_11_24_ev_renta_demog_paro_pobl.csv")
print("Filas:", len(merged), "| Columnas:", len(merged.columns))
print("Años:", merged["anyo"].min(), "-", merged["anyo"].max())
print("Faltantes en poblacion_anual:", merged["poblacion_anual"].isna().sum())
print(merged[["anyo","Distrito","poblacion_anual"]].head())


general_11_24_ev_renta_demog_paro_pobl = pd.read_csv("general_11_24_ev_renta_demog_paro_pobl.csv")

general_11_24_ev_renta_demog_paro_pobl

# ## Unificar dataset SuperficieZonasVerdesDistritosCalles_std con dataset general_11_24_ev_renta_demog_paro_pobl ##

import pandas as pd

# Cargar CSV (con ;)
try:
    zv = pd.read_csv("SuperficieZonasVerdesDistritosCalles_std.csv", sep=";", encoding="utf-8-sig")
except UnicodeDecodeError:
    zv = pd.read_csv("SuperficieZonasVerdesDistritosCalles_std.csv", sep=";", encoding="cp1252")

# limpiar encabezados
zv.columns = [c.replace("\ufeff","").strip() for c in zv.columns]

# asegurar claves estándar
for cand in ("Año","año","Ano","ano"):
    if cand in zv.columns: 
        zv = zv.rename(columns={cand:"anyo"})
        break

# Asegurar que la columna 'Distrito' se llame así
for c in list(zv.columns):
    if c.lower().strip() == "distrito" and c != "Distrito":
        zv = zv.rename(columns={c:"Distrito"})
        break

print("ZV cols:", zv.columns.tolist())  # Asegurar que la columna 'Distrito' y 'anyo' se llamen así

# Cargar CSV
general = pd.read_csv("general_11_24_ev_renta_demog_paro_pobl.csv", encoding="utf-8-sig")
general.columns = [c.replace("\ufeff","").strip() for c in general.columns]
general = general.rename(columns={"Año":"anyo", "año":"anyo"})

# Merge anyo + Distrito
merged = general.merge(zv, on=["anyo","Distrito"], how="left")

merged.to_csv("general_11_24_ev_renta_demog_paro_pobl_zv.csv", index=False, encoding="utf-8-sig")
print("Unificado y guardado en general_11_24_ev_renta_demog_paro_pobl_zv.csv")


# Correción y estandarización de columnas del datset de inversión

import pandas as pd
import unicodedata, re

# Funciones auxiliares para limpiar y estandarizar texto
# slug(s):   convierte un string en un identificador simple (solo letras minúsculas y guiones)
# normalize_col(c): prepara nombres de columnas (sin acentos, minúsculas, con guiones bajos)
def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().strip()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

def normalize_col(c: str) -> str:
    s = unicodedata.normalize("NFKD", str(c))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

# Cargar .xlsx
df = pd.read_excel("inversiones_obra.xlsx", sheet_name=0)
df.columns = [normalize_col(c) for c in df.columns]

# estandarizar la columna 'Distrito'
df["distrito_std"] = df["distrito"].map(slug)

# construir base SIN duplicar 'distrito'
base = df.loc[:, ["distrito_std", "ano_inicio", "codigo_pep", "ano_de_finalizacion"]].copy()
base = base.rename(columns={
    "distrito_std": "distrito",  # ya estandarizado (minúsculas, sin acentos ni espacios)
    "ano_inicio": "anyo",        # renombramos columna
    "codigo_pep": "codigo"
})

base["anyo"] = pd.to_numeric(base["anyo"], errors="coerce").astype("Int64")
base["ano_de_finalizacion"] = pd.to_numeric(base["ano_de_finalizacion"], errors="coerce").astype("Int64")

# Conteos por AÑO DE INICIO 
iniciadas = (
    base.groupby(["distrito", "anyo"], as_index=False)["codigo"]
        .nunique()
        .rename(columns={"codigo": "obras_iniciadas"})
)

finalizadas = (
    base[base["ano_de_finalizacion"].notna()]
        .groupby(["distrito", "anyo"], as_index=False)["codigo"]
        .nunique()
        .rename(columns={"codigo": "obras_finalizadas"})
)

resumen = (
    iniciadas.merge(finalizadas, on=["distrito", "anyo"], how="left")
             .fillna({"obras_finalizadas": 0})
)
resumen["obras_finalizadas"] = resumen["obras_finalizadas"].astype(int)

resumen.to_csv("resumen_inversiones_por_inicio_std.csv", index=False, encoding="utf-8-sig")

print("OK -> resumen_inversiones_por_inicio_std.csv")


# Renombrar columnas del resumen_inversiones_por_inicio_std

import pandas as pd
import unicodedata
import re

# Lista estándar de distritos definidos
distritos_std = [
    "CENTRO","ARGANZUELA","RETIRO","SALAMANCA","CHAMARTIN","TETUAN","CHAMBERI",
    "FUENCARRALELPARDO","MONCLOAARAVACA","LATINA","CARABANCHEL","USERA",
    "PUENTEDEVALLECAS","MORATALAZ","CIUDADLINEAL","HORTALEZA","VILLAVERDE",
    "VILLADEVALLECAS","VICALVARO","SANBLASCANILLEJAS","BARAJAS"
]

# función para limpiar texto
def limpiar(s):
    s = str(s).upper().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z0-9]", "", s)  # quitar espacios y símbolos
    return s

# cargar dataset
df = pd.read_csv("resumen_inversiones_por_inicio_std.csv", encoding="utf-8")

# limpiar valores de distrito
df["distrito"] = df["distrito"].map(limpiar)

# filtrar solo los distritos de la lista
df = df[df["distrito"].isin(distritos_std)].copy()

# renombrar columnas
df = df.rename(columns={
    "distrito": "Distrito",
    "anyo": "anyo",
    "obras_iniciadas": "obras_iniciadas",
    "obras_finalizadas": "obras_finalizadas"
})

df.to_csv("resumen_inversiones_por_inicio_std_clean.csv", index=False, encoding="utf-8-sig")

print(df.head())


# ## Unificación de dataset general_11_24_ev_renta_demog_paro_pobl_zv con el dataset resumen_inversiones_por_inicio_std_clean ##

import pandas as pd

# cargar datasets
resumen = pd.read_csv("resumen_inversiones_por_inicio_std_clean.csv", encoding="utf-8")
general = pd.read_csv("general_11_24_ev_renta_demog_paro_pobl_zv.csv", encoding="utf-8")

# Aseguramos nombres sin espacios 
resumen.columns = resumen.columns.str.strip()
general.columns = general.columns.str.strip()

# LEFT JOIN para mantener todas las filas de general
df_final = pd.merge(
    general,
    resumen,
    on=["Distrito", "anyo"],
    how="left"
)

df_final.to_csv("general_11_24_ev_renta_demog_paro_pobl_zv_obras.csv", index=False, encoding="utf-8-sig")

print("Archivo generado: general_11_24_ev_renta_demog_paro_pobl_zv_obras.[csv]")
print(df_final.head())


# Corrección de Columna Población para garantizar que sea promedio mensual de población

import pandas as pd
import unicodedata, re

def normaliza_distrito(s):
    if pd.isna(s): return s
    s = str(s).strip().upper()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))  # sin acentos
    return re.sub(r"\s+", "", s)  # sin espacios

# Leer el archivo con separador ;
df = pd.read_csv("poblacion_2011_2022_sin_decimales.csv", encoding="utf-8-sig", sep=";")
print("Columnas detectadas:", df.columns.tolist())

# Estandarizar columna 'Distrito'
df["Distrito"] = df["Distrito"].apply(normaliza_distrito)

# Calcular el promedio anual de la columna Total por Distrito y anyo
promedios = (
    df.groupby(["Distrito", "anyo"], as_index=False)
      .agg(promedio_anual=("Total", "mean"))
)

# Quitar decimales de la columna del promedio
promedios["promedio_anual"] = promedios["promedio_anual"].round(0).astype("Int64")

promedios.to_csv("poblacion_promedio_anual.csv", index=False, encoding="utf-8-sig", sep=";")

print("Promedio anual calculado, distritos estandarizados.")
print(promedios.head())


# ## Unificación de datos del dataset poblacion_promedio_anual con el dataset general_11_24_ev_renta_demog_paro_pobl_zv_obras ##

import pandas as pd
import io

# Lector de CSV (detecta separadores como: coma, punto y coma, tab, maneja UTF-8 con BOM
def safe_read(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", skipinitialspace=True)
    if len(df.columns) == 1 and any(ch in df.columns[0] for ch in [",",";","\t","|"]):
        text = df.columns[0] + "\n" + "\n".join(df.iloc[:,0].astype(str).tolist())
        df = pd.read_csv(io.StringIO(text), sep=",", engine="python", encoding="utf-8-sig", skipinitialspace=True)
    return df

prom    = safe_read("poblacion_promedio_anual.csv")
general = safe_read("general_11_24_ev_renta_demog_paro_pobl_zv_obras.csv")

# Quitar comillas y espacios en nombres de columnas y valores
def strip_quotes(s):
    if pd.isna(s): return s
    s = str(s).strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1]
    return s.strip()

prom.columns    = [strip_quotes(c) for c in prom.columns]
general.columns = [strip_quotes(c) for c in general.columns]

for c in ["Distrito", "anyo"]:
    if c in prom.columns:
        prom[c] = prom[c].apply(strip_quotes).astype(str).str.strip().str.upper()
    if c in general.columns:
        general[c] = general[c].apply(strip_quotes).astype(str).str.strip().str.upper()

# Merge con Distrito + anyo
merged = general.merge(prom, on=["Distrito","anyo"], how="left")

output_file = "general_11_24_ev_renta_demog_paro_pobl_zv_obras_unificado.csv"
merged.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Archivo guardado como {output_file}")