import pandas as pd
from skrub import TableReport

# Obtención de datos en bruto
path = 'data/raw/renta_media.csv'

while True:
    try:
        df = pd.read_csv(path, sep=';', thousands='.')
        break
    except FileNotFoundError:
        print(f'No se encontró el archivo: {path}')
        path = input('Introduce ruta correcta: ')

print('--> Acceso a archivos con éxito.')

# Guardar reporte
with open('reports/reporte_renta_media.html', 'w') as f:
    f.write(TableReport(df).html())

print('--> Reporte guardado en "reports/reporte_renta_media.html".')

# Guardar csv con datos procesados
df.to_csv('data/processed/renta_media_procesado.csv')

print('--> Datos procesados guardados en "data/processed/renta_media_procesado.csv".')