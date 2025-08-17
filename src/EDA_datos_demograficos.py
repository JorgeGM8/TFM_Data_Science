import pandas as pd
from skrub import TableReport

df = pd.read_csv('data/datos_demograficos_ine.csv', sep=';', encoding='ISO-8859-1')

# Eliminación de columnas sin uso o vacías
df = df.drop(['Municipios', 'Secciones'], axis=1)

# Mapeo de distritos con nombres
distritos = {
    '01': 'Centro',
    '02': 'Arganzuela',
    '03': 'Retiro',
    '04': 'Salamanca',
    '05': 'Chamartín',
    '06': 'Tetuán',
    '07': 'Chamberí',
    '08': 'Fuencarral',
    '09': 'Moncloa',
    '10': 'Latina',
    '11': 'Carabanchel',
    '12': 'Usera',
    '13': 'Puente de Vallecas',
    '14': 'Moratalaz',
    '15': 'Ciudad Lineal',
    '16': 'Hortaleza',
    '17': 'Villaverde',
    '18': 'Villa de Vallecas',
    '19': 'Vicálvaro',
    '20': 'San Blas',
    '21': 'Barajas'
}

df['Distritos'] = df['Distritos'].str[-2:].map(distritos)

# Pivotar dataframe para colocar indicadores como columnas
df_pivot = df.pivot(
    index=['Distritos', 'Periodo'],
    columns='Indicadores demográficos',
    values='Total'
).reset_index()

report = TableReport(df_pivot)

# Guardar reporte HTML
with open('reports/report_demograficos.html', 'w') as f:
    f.write(report.html())
print('\n--> Reporte guardado en reports/report_demograficos.html\n')

# Guardar datos procesados
df_pivot.to_csv('data/processed/datos_demograficos_ine_procesado.csv')

print('\n--> Datos procesados guardados en data/processed/datos_demograficos_ine_procesado.csv\n')