import pandas as pd
from skrub import TableReport
try:
    from utils.cleaning import convertir_formato_europeo
except ModuleNotFoundError:
    print('--> Ejecuta el archivo con "python -m src.limpieza_venta_alquiler" para importar correctamente las funciones.')
    quit()

# Obtención de datos en bruto
path_venta = 'data/raw/venta_vivienda.csv'
path_alquiler = 'data/raw/alquiler_vivienda.csv'

while True:
    try:
        df_venta = pd.read_csv(path_venta, sep=';', decimal=',', thousands='.')
    except FileNotFoundError:
        print(f'No se encontró el archivo: {path_venta}')
        path_venta = input('Introduce ruta correcta: ')
        continue

    try:
        df_alquiler = pd.read_csv(path_alquiler, sep=';', decimal=',', thousands='.')
        break
    except FileNotFoundError:
        print(f'No se encontró el archivo: {path_alquiler}')
        path_alquiler = input('Introduce ruta correcta: ')

print('--> Acceso a archivos con éxito.')

# Transformar valores de cada distrito a numéricos y valores faltantes a nulos
columnas_distritos = df_venta.columns.difference(['Año', 'Mes'])

df_venta[columnas_distritos] = df_venta[columnas_distritos].apply(convertir_formato_europeo)
df_alquiler[columnas_distritos] = df_alquiler[columnas_distritos].apply(convertir_formato_europeo)

print('--> Valores transformados a numéricos y a nulos.')

# Transformar dataframes para mostrar los distritos en una columna y valores en otra
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

# Unión de ambos dataframes
df_merge = pd.merge(
    df_venta,
    df_alquiler,
    on=['Año', 'Mes', 'Distrito']
)

print('--> Dataframes de venta y alquiler unidos.')

# Modificar meses a números
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

# Selección de rango sin nulos
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

# Guardar reporte
with open('reports/reporte_venta_alquiler.html', 'w') as f:
    f.write(TableReport(df_continuo).html())

print('--> Reporte guardado en "reports/reporte_venta_alquiler.html".')

# Guardar csv con datos procesados
df_continuo.to_csv('data/processed/venta_alquiler_procesado.csv')

print('--> Datos procesados guardados en "data/processed/venta_alquiler_procesado.csv".')