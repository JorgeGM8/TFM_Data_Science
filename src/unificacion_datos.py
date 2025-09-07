import pandas as pd
from skrub import TableReport
try:
    from utils.cleaning import convertir_formato_europeo, normaliza_distrito
except ModuleNotFoundError:
    print('--> Ejecuta el archivo con "python -m src.limpieza_venta_alquiler" para importar correctamente las funciones.')
    quit()
import re

#  1. Obtenemos los datos de precio de venta y de alquiler de las viviendas

# 1.1 Obtencion de datos en bruto
path_venta = 'data/raw/venta_vivienda.csv'
path_alquiler = 'data/raw/alquiler_vivienda.csv'

try:
    df_venta = pd.read_csv(path_venta, sep=';', decimal=',', thousands='.')
except FileNotFoundError:
    print(f'No se encontro el archivo: {path_venta}. Por favor comprueba que la ruta sea la correcta.')

try:
    df_alquiler = pd.read_csv(path_alquiler, sep=';', decimal=',', thousands='.')
except FileNotFoundError:
    print(f'No se encontro el archivo: {path_alquiler}. Por favor comprueba que la ruta sea la correcta.')

print('--> Acceso a archivos con exito.')

# 1.2 Transformar valores de cada distrito a numericos y valores faltantes a nulos
columnas_distritos = df_venta.columns.difference(['Año', 'Mes'])
df_venta[columnas_distritos] = df_venta[columnas_distritos].apply(convertir_formato_europeo)
df_alquiler[columnas_distritos] = df_alquiler[columnas_distritos].apply(convertir_formato_europeo)

print('--> Valores transformados a numericos y a nulos.')
# 1.3 Transformar dataframes para mostrar los distritos en una columna y valores en otra
df_venta = df_venta.melt(
    id_vars=['Año', 'Mes'],
    var_name='Distrito',
    value_name='Precio_venta',
    value_vars=columnas_distritos
)

print('--> Transformación de estructura de venta completada.')

df_alquiler = df_alquiler.melt(
    id_vars=['Año', 'Mes'],
    var_name='Distrito',
    value_name='Precio_alquiler',
    value_vars=columnas_distritos
)

print('--> Transformación de estructura de alquiler completada.')

# 1.4 Unión de ambos dataframes
df_merge = pd.merge(df_venta, df_alquiler, on=['Año', 'Mes', 'Distrito'])

print('--> Dataframes de venta y alquiler unidos.')

# 1.5 Modificar meses a números
mapeo_meses = {
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12
}

df_merge['Mes'] = df_merge['Mes'].str.lower().map(mapeo_meses)

print('--> Meses transformados a números.')

# 1.6 Selección de rango sin nulos
cols_valores = ['Precio_venta', 'Precio_alquiler']  # Columnas con valores

nulos_por_mes = ( # Agrupación por meses
    df_merge.groupby(['Año', 'Mes'])[cols_valores]
    .apply(lambda x: x.isnull().sum().sum())  # Suma de todos los nulos de las columnas
    .reset_index(name='nulos_totales')
)

nulos_desc = nulos_por_mes.sort_values(['Año', 'Mes'], ascending=False) # Ordenar descendente por año y mes

primer_mes_info = nulos_desc[nulos_desc['nulos_totales'] > 0].iloc[0] # Filtrar mes y año más nuevo que tenga algún nulo

ano_corte = primer_mes_info['Año']
mes_corte = primer_mes_info['Mes']

df_continuo = df_merge[
    (df_merge['Año'] > ano_corte) |
    ((df_merge['Año'] == ano_corte) & (df_merge['Mes'] > mes_corte)) # Coger todos los meses y años a partir de donde no hay nulos
].sort_values(['Año', 'Mes'], ascending=True).reset_index(drop=True) # Volver al orden original (ascendente)

print(f'--> Seleccionado rango sin nulos ({df_continuo['Mes'][0]}/{df_continuo['Año'][0]} - 12/2024).')

# 1.7 Convertimos los resultados por meses a resultados anuales (media anual)
df_anual = (
    df_continuo.groupby(['Año', 'Distrito'], as_index=False)
    .agg({
        'Precio_venta': 'mean',
        'Precio_alquiler': 'mean'
    })
)

print(f'--> Agregados datos mensuales a anuales.')


# 1.8 Estandarizamos los encabezados de 'venta_alquiler_procesado.csv'
df_venta_alquiler = df_anual.copy()

df_venta_alquiler.columns = [re.sub(r"\s+", " ", c).strip() for c in df_venta_alquiler.columns]

for c in df_venta_alquiler.columns:
    if c.lower().strip() == "distrito" and c != "Distrito":
        df_venta_alquiler = df_venta_alquiler.rename(columns={c: "Distrito"})
        break

if "Distrito" not in df_venta_alquiler.columns:
    raise KeyError("No se encontró una columna llamada 'Distrito' (o equivalente).")

df_venta_alquiler["Distrito"] = df_venta_alquiler["Distrito"].apply(normaliza_distrito)

df_venta_alquiler = df_venta_alquiler.rename(columns={"Año": "Ano", "año": "Ano"})

# 1.9 Guardar csv con datos procesados
df_venta_alquiler.to_csv('data/processed/venta_alquiler_procesado.csv', index=False, encoding="utf-8")

print('--> Datos procesados guardados en "data/processed/venta_alquiler_procesado.csv".')

# -------------------------------------------------------------------------------------------- #

# 2. Añadimos a nuestro CSV la columna de esperanza de vida

df_esperanza_vida = pd.read_csv("data/raw/esperanza_vida.csv", sep=";", encoding="utf-8", decimal=',')

# 2.1 Normalizar columnas de esperanza de vida, renombrando la columna año
df_esperanza_vida = df_esperanza_vida.rename(columns={"Año":"Ano", "año":"Ano", "Esperanza de vida":"Esperanza_vida"})

# 2.2 Limpiar columna Distrito
df_esperanza_vida["Distrito"] = df_esperanza_vida["Distrito"].map(normaliza_distrito)

# 2.5 Unir datasets (Ano + Distrito)
df_completo = df_venta_alquiler.merge(df_esperanza_vida, on=["Ano","Distrito"], how="left")

df_esperanza_vida.to_csv("data/processed/completo_esperanza_vida.csv", index=False, encoding="utf-8")

print('--> Datos procesados guardados en "data/processed/completo_esperanza_vida.csv".')
print('--> Añadido esperanza de vida al dataset.')

# -------------------------------------------------------------------------------------------- #

# 3. Estandarización del dataset de renta_media_persona

# 3.1 Cargar  del archivo con tolerancia de encoding y separador ;
df_renta = pd.read_csv("data/raw/renta_media.csv", sep=";", encoding="utf-8", thousands='.')

# 3.2 Unificar columna año y renta media
df_renta = df_renta.rename(columns={"Año":"Ano", "año":"Ano", "Renta neta media por persona":"Renta_media_persona"})

# 3.3 Unificar columna 'Distrito'
df_renta["Distrito"] = df_renta["Distrito"].map(normaliza_distrito)

# 3.4 Unir datasets (Ano + Distrito) y volcarlo al CSV
df_completo = df_venta_alquiler.merge(df_renta, on=["Ano","Distrito"], how="left")
df_renta.to_csv("data/processed/completo_renta_media.csv", index=False, encoding="utf-8")

print('--> Datos procesados guardados en "data/processed/completo_renta_media.csv".')
print('--> Añadido renta media por persona al dataset.')

# -------------------------------------------------------------------------------------------- #


# Guardar reporte
# with open('reports/reporte_venta_alquiler.html', 'w', encoding='utf-8') as f:
#     f.write(TableReport(df_venta_alquiler).html())

# print('--> Reporte guardado en "reports/reporte_venta_alquiler.html".')

