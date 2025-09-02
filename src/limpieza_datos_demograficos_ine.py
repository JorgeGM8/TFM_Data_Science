import pandas as pd
from skrub import TableReport

# Obtención de datos en bruto
path = 'data/raw/datos_demograficos_ine.csv'

while True:
    try:
        df = pd.read_csv(path, sep=';', encoding='ISO-8859-1',
                 thousands='.', decimal=',')
        break
    except FileNotFoundError:
        print(f'No se encontró el archivo: {path}')
        path = input('Introduce ruta correcta: ')

print('--> Acceso a archivos con éxito.')

# Eliminación de columnas sin uso o vacías
df = df.drop(['Municipios', 'Secciones'], axis=1)

print('--> Eliminadas columnas sin uso o vacías.')

# Mapeo de distritos con nombres
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

df['Distritos'] = df['Distritos'].str[-2:].map(distritos)

print('--> Realizado mapeo de distritos con nombres estandarizados.')

# Pivotar dataframe para colocar indicadores como columnas
df_pivot = df.pivot(
    index=['Distritos', 'Periodo'],
    columns='Indicadores demográficos',
    values='Total'
).reset_index()

print('--> Columnas pivotadas.')

# Estandarizar nombres de columnas
df_pivot = df_pivot.rename(columns={'Distritos': 'Distrito', 'Periodo': 'Año'})

# Guardar reporte
with open('reports/reporte_datos_demograficos_ine.html', 'w') as f:
    f.write(TableReport(df_pivot).html())

print('--> Reporte guardado en "reports/datos_demograficos_ine.html".')

# Guardar csv con datos procesados
df_pivot.to_csv('data/processed/datos_demograficos_ine_procesado.csv')

print('--> Datos procesados guardados en "data/processed/datos_demograficos_ine_procesado.csv".')