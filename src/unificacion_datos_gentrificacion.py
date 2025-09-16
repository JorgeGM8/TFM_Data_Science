import pandas as pd
from skrub import TableReport
try:
    from utils.cleaning import convertir_formato_europeo, normaliza_distrito
except ModuleNotFoundError:
    print('--> Ejecuta el archivo con "python -m src.unificacion_datos" para importar correctamente las funciones.')
    quit()
except ImportError as e:
    print(f'--> Error de importación: {e}')
    quit()
except Exception as e:
    print(f'--> Error inesperado: {type(e).__name__}: {e}')
import re

#  1. Obtenemos los datos de precio de venta y de alquiler de las viviendas

# 1.1 Obtencion de datos en bruto
path_venta = 'data/raw/venta_vivienda.csv'
path_alquiler = 'data/raw/alquiler_vivienda.csv'

try:
    df_venta = pd.read_csv(path_venta, sep=';', decimal=',', thousands='.')
except FileNotFoundError:
    print(f'No se encontró el archivo: {path_venta}. Por favor comprueba que la ruta sea la correcta.')

try:
    df_alquiler = pd.read_csv(path_alquiler, sep=';', decimal=',', thousands='.')
except FileNotFoundError:
    print(f'No se encontró el archivo: {path_alquiler}. Por favor comprueba que la ruta sea la correcta.')

# 1.2 Transformar valores de cada distrito a numericos y valores faltantes a nulos
columnas_distritos = df_venta.columns.difference(['Año', 'Mes']).tolist()
df_venta[columnas_distritos] = df_venta[columnas_distritos].apply(convertir_formato_europeo)
df_alquiler[columnas_distritos] = df_alquiler[columnas_distritos].apply(convertir_formato_europeo)

# 1.3 Transformar dataframes para mostrar los distritos en una columna y valores en otra
df_venta = df_venta.melt(
    id_vars=['Año', 'Mes'],
    var_name='Distrito',
    value_name='Precio_venta',
    value_vars=columnas_distritos
)

df_alquiler = df_alquiler.melt(
    id_vars=['Año', 'Mes'],
    var_name='Distrito',
    value_name='Precio_alquiler',
    value_vars=columnas_distritos
)

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

# 1.7 Convertimos los resultados por meses a resultados anuales (media anual)
df_anual = (
    df_continuo.groupby(['Año', 'Distrito'], as_index=False)
    .agg({
        'Precio_venta': 'mean',
        'Precio_alquiler': 'mean'
    })
)

# 1.8 Estandarizamos los encabezados de "venta_alquiler_procesado.csv"
df_venta_alquiler = df_anual.copy()

df_venta_alquiler.columns = [re.sub(r'\s+', ' ', c).strip() for c in df_venta_alquiler.columns]

for c in df_venta_alquiler.columns:
    if c.lower().strip() == 'distrito' and c != 'Distrito':
        df_venta_alquiler = df_venta_alquiler.rename(columns={c: 'Distrito'})
        break

if 'Distrito' not in df_venta_alquiler.columns:
    raise KeyError('No se encontró una columna llamada "Distrito" (o equivalente).')

df_venta_alquiler['Distrito'] = df_venta_alquiler['Distrito'].apply(normaliza_distrito)

df_venta_alquiler = df_venta_alquiler.rename(columns={'Año': 'Ano', 'año': 'Ano'})

# 1.9 Guardar csv con datos procesados
df_venta_alquiler.to_csv('data/processed/venta_alquiler_procesado.csv', index=False, encoding='utf-8')

print('--> Datos procesados guardados en "data/processed/venta_alquiler_procesado.csv".')

# -------------------------------------------------------------------------------------------- #

# 2. Estandarización del dataset de esperanza_vida y añadir al CSV

# 2.1 Cargar el archivo
path_esperanza_vida = 'data/raw/esperanza_vida.csv'

try:
    df_esperanza_vida = pd.read_csv('data/raw/esperanza_vida.csv', sep=';', encoding='utf-8', decimal=',')
except FileNotFoundError:
    print(f'No se encontró el archivo: {path_esperanza_vida}. Por favor comprueba que la ruta sea la correcta.')

# 2.1 Normalizar columnas de esperanza de vida, renombrando la columna año
df_esperanza_vida = df_esperanza_vida.rename(columns={'Año':'Ano', 'año':'Ano', 'Esperanza de vida':'Esperanza_vida'})

# 2.2 Limpiar columna Distrito
df_esperanza_vida['Distrito'] = df_esperanza_vida['Distrito'].map(normaliza_distrito)

# 2.5 Unir datasets (Ano + Distrito)
df_completo = df_venta_alquiler.merge(df_esperanza_vida, on=['Ano','Distrito'], how='left')

df_completo.to_csv('data/processed/completo_esperanza_vida.csv', index=False, encoding='utf-8')

print('--> Datos procesados guardados en "data/processed/completo_esperanza_vida.csv".')
print('--> Añadido esperanza de vida al dataset.')

# -------------------------------------------------------------------------------------------- #

# 3. Estandarización del dataset de renta_media_hogar y añadir al CSV

# 3.1 Cargar el archivo con encoding y separador ;
path_rentas = 'data/raw/renta_media_hogar.csv'

try:
    df_rentas = pd.read_csv(path_rentas, sep=';', encoding='utf-8', thousands='.')
except FileNotFoundError:
    print(f'No se encontró el archivo: {path_rentas}. Por favor comprueba que la ruta sea la correcta.')

# 3.2 Eliminar columnas innecesarias
df_rentas = df_rentas.drop(columns=[
    'Media de la renta neta por unidad de consumo',
    'Mediana de la renta neta por unidad de consumo'
])

# 3.3 Unificar columnas
df_rentas = df_rentas.rename(columns={
    'Año': 'Ano',
    'Renta neta media por persona': 'Renta_neta_persona',
    'Renta neta media por hogar': 'Renta_neta_hogar',
    'Renta bruta media por persona': 'Renta_bruta_persona',
    'Renta media bruta por hogar': 'Renta_bruta_hogar'
})

# 3.4 Unificar columna 'Distrito'
df_rentas['Distrito'] = df_rentas['Distrito'].map(normaliza_distrito)

# 3.4 Unir datasets (Ano + Distrito) y volcarlo al CSV
df_completo = df_completo.merge(df_rentas, on=['Ano','Distrito'], how='left')
df_completo.to_csv('data/processed/completo_renta_media_hogar.csv', index=False, encoding='utf-8')

print('--> Datos procesados guardados en "data/processed/completo_renta_media_hogar.csv".')
print('--> Añadido renta media (neta y bruta) por persona y hogar al dataset.')

# -------------------------------------------------------------------------------------------- #

# 4. Limpiar y añadir datos demográficos al CSV

# 4.1 Cargar el archivo con encoding
path_demograficos = 'data/raw/datos_demograficos_ine.csv'

try:
    df_demograficos = pd.read_csv(path_demograficos, sep=';', encoding='ISO-8859-1', thousands='.', decimal=',')
except FileNotFoundError:
    print(f'No se encontró el archivo: {path_demograficos}. Por favor comprueba que la ruta sea la correcta.')

# 4.2 Eliminación de columnas sin uso o vacías
df_demograficos = df_demograficos.drop(['Municipios', 'Secciones'], axis=1)

# 4.3 Mapeo de distritos con nombres
distritos = {
    '01': '01. Centro',
    '02': '02. Arganzuela',
    '03': '03. Retiro',
    '04': '04. Salamanca',
    '05': '05. Chamartín',
    '06': '06. Tetuán',
    '07': '07. Chamberí',
    '08': '08. Fuencarral-El Pardo',
    '09': '09. Moncloa-Aravaca',
    '10': '10. Latina',
    '11': '11. Carabanchel',
    '12': '12. Usera',
    '13': '13. Puente de Vallecas',
    '14': '14. Moratalaz',
    '15': '15. Ciudad Lineal',
    '16': '16. Hortaleza',
    '17': '17. Villaverde',
    '18': '18. Villa de Vallecas',
    '19': '19. Vicálvaro',
    '20': '20. San Blas-Canillejas',
    '21': '21. Barajas'
}

df_demograficos['Distritos'] = df_demograficos['Distritos'].str[-2:].map(distritos)

# 4.4 Pivotar dataframe para colocar indicadores como columnas
df_demograficos = df_demograficos.pivot(
    index=['Distritos', 'Periodo'],
    columns='Indicadores demográficos',
    values='Total'
).reset_index()

# 4.5 Ajustar nombres de columnas
df_demograficos = df_demograficos.rename(columns={
    'Distritos': 'Distrito',
    'Periodo': 'Ano',
    'Edad media de la población': 'Edad_media',
    'Porcentaje de población menor de 18 años': 'Menores_18anos%',
    'Porcentaje de población de 65 y más años': 'Mayores_65anos%',
    'Tamaño medio del hogar': 'Tamano_vivienda_personas',
    'Población': 'Poblacion'
})

# 4.6 Normalizar nombres de distrito
df_demograficos['Distrito'] = df_demograficos['Distrito'].map(normaliza_distrito)

# 4.7 Transformar porcentajes a valor entre 0 y 1
df_demograficos['Menores_18anos%'] = df_demograficos['Menores_18anos%'] / 100
df_demograficos['Mayores_65anos%'] = df_demograficos['Mayores_65anos%'] / 100

# 4.8 Unir datasets (Ano + Distrito) y volcarlo al CSV
df_completo = df_completo.merge(df_demograficos, on=['Ano','Distrito'], how='left')
df_completo.to_csv('data/processed/completo_demograficos.csv', index=False, encoding='utf-8')

print('--> Datos procesados guardados en "data/processed/completo_demograficos.csv".')
print('--> Añadido edad media, poblacion, % menores 18 años, % mayores 65 años y tamaño vivienda personas al dataset.')

# -------------------------------------------------------------------------------------------- #

# 5. Limpiar y añadir paro registrado al CSV

# 4.1 Cargar el archivo con encoding
path_paro = 'data/raw/paro_registrado_2012_2024.csv'

try:
    df_paro = pd.read_csv(path_paro, sep=';', encoding='utf-8', thousands='.')
except FileNotFoundError:
    print(f'No se encontró el archivo: {path_paro}. Por favor comprueba que la ruta sea la correcta.')

# 4.2 Ajustar nombres de columnas
df_paro = df_paro.rename(columns={'Año': 'Ano', 'Paro registrado': 'Paro_registrado'})

# 4.3 Convertimos los resultados por meses a resultados anuales (media anual)
df_paro = (df_paro.groupby(['Ano', 'Distrito'], as_index=False).agg({'Paro_registrado': 'mean'}))

# 4.4 Normalizar nombres de distritos
df_paro['Distrito'] = df_paro['Distrito'].map(normaliza_distrito)

# 4.5 Unir datasets (Ano + Distrito)
df_completo = df_completo.merge(df_paro, on=['Ano','Distrito'], how='left')

# 4.6 Calcular porcentaje de parados respecto a poblacion y eliminar columna antigua
df_completo['Paro_registrado%'] = df_completo['Paro_registrado'] / df_completo['Poblacion']
df_completo = df_completo.drop(columns=['Paro_registrado'])

# 4.7 Volcado de CSV
df_completo.to_csv('data/processed/completo_paro.csv', index=False, encoding='utf-8')

print('--> Datos procesados guardados en "data/processed/completo_paro.csv".')
print('--> Añadido porcentaje de paro registrado al dataset.')

# -------------------------------------------------------------------------------------------- #

# 6. Limpiar y añadir apartamentos turísticos al CSV

# 6.1 Cargar el archivo con encoding
path_apartamentos = 'data/raw/apartamentos_turisticos_2015_2022.csv'

try:
    df_apartamentos = pd.read_csv(path_apartamentos, sep=';', encoding='utf-8')
except FileNotFoundError:
    print(f'No se encontró el archivo: {path_apartamentos}. Por favor comprueba que la ruta sea la correcta.')

# 6.2 Eliminación de columnas sin uso o vacías
df_apartamentos = df_apartamentos.drop(['Tipo'], axis=1)

# 6.3 Ajustar nombres de columnas
df_apartamentos = df_apartamentos.rename(columns={'Año': 'Ano', 'Total': 'Apartamentos_turisticos'})

# 6.4 Normalizar nombres de distritos
df_apartamentos['Distrito'] = df_apartamentos['Distrito'].map(normaliza_distrito)

# 6.5 Unir datasets (Ano + Distrito) y volcarlo al CSV
df_completo = df_completo.merge(df_apartamentos, on=['Ano','Distrito'], how='left')
df_completo.to_csv('data/processed/completo_apartamentos.csv', index=False, encoding='utf-8')

print('--> Datos procesados guardados en "data/processed/completo_apartamentos.csv".')
print('--> Añadido apartamentos turísticos disponibles al dataset.')

# -------------------------------------------------------------------------------------------- #

# 7. Cálculo de densidad de zonas verdes y de densidad de población

# 7.1 Cargar el archivo con encoding de superficie de distrito
path_superficie = 'data/raw/superficie_distrito.csv'

try:
    df_superficie = pd.read_csv(path_superficie, sep=';', encoding='utf-8', thousands='.', decimal=',')
except FileNotFoundError:
    print(f'No se encontró el archivo: {path_superficie}. Por favor comprueba que la ruta sea la correcta.')

# 7.2 Ajustar nombres de columnas
df_superficie = df_superficie.rename(columns={'Superficie': 'Superficie_distrito_ha'})

# 7.3 Normalizar nombres de distritos
df_superficie['Distrito'] = df_superficie['Distrito'].map(normaliza_distrito)

# 7.4 Unir datasets (Ano + Distrito)
df_completo = df_completo.merge(df_superficie, on=['Distrito'], how='left')

# 7.5 Cargar el archivo con encoding de zonas verdes
path_zonas_verdes = 'data/raw/superficie_zonas_verdes.csv'

try:
    df_zonas_verdes = pd.read_csv(path_zonas_verdes, sep=';', encoding='utf-8', thousands='.', decimal=',')
except FileNotFoundError:
    print(f'No se encontró el archivo: {path_zonas_verdes}. Por favor comprueba que la ruta sea la correcta.')

# 7.6 Ajustar nombres de columnas
df_zonas_verdes = df_zonas_verdes.rename(columns={
    'DISTRITO': 'Distrito',
    'SUPERFICIE DE ZONAS VERDES Y PARQUES DE DISTRITO (ha)': 'Zonas_verdes_ha'
})

# 7.7 Normalizar nombres de distritos
df_zonas_verdes['Distrito'] = df_zonas_verdes['Distrito'].map(normaliza_distrito)

# 7.8 Unir datasets (Ano + Distrito)
df_completo = df_completo.merge(df_zonas_verdes, on=['Distrito'], how='left')

# 7.9 Calcular densidad de población
df_completo['Densidad_poblacion'] = df_completo['Poblacion'] / df_completo['Superficie_distrito_ha']
df_completo = df_completo.drop(columns=['Poblacion'])

# 7.10 Calcular densidad de zonas verdes
df_completo['Zonas_verdes%'] = df_completo['Zonas_verdes_ha'] / df_completo['Superficie_distrito_ha']
df_completo = df_completo.drop(columns=['Zonas_verdes_ha'])

# 7.11 Volcado CSV completo
df_completo.to_csv('data/processed/gentrificacion_madrid.csv', index=False, encoding='utf-8')

print('--> Datos procesados guardados en "data/processed//gentrificacion_madrid.csv".')
print('--> Añadido densidad de poblacion y porcentaje de zonas verdes al dataset.')

# -------------------------------------------------------------------------------------------- #

# 8. Guardar reporte completo
with open('reports/reporte_gentrificacion_madrid.html', 'w', encoding='utf-8') as f:
    f.write(TableReport(df_completo).html())

print('--> Reporte guardado en "reports/reporte_gentrificacion_madrid.html".')