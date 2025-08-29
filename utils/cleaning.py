import pandas as pd

# Función para eliminar comillas y convertir formato numérico europeo a americano
def convertir_formato_europeo(col):
    col = col.astype(str).str.replace(r'[\"\']', '', regex=True).str.strip()  # Quita comillas y espacios
    col = col.str.replace('.', '', regex=False)  # Quita puntos de miles
    col = col.str.replace(',', '.', regex=False)  # Coma decimal a punto
    return pd.to_numeric(col, errors='coerce')