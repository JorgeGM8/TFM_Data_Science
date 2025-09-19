import pandas as pd
import numpy as np

def calcular_alquiler_venta(df: pd.DataFrame,
                            tipo: str,
                            precio: str = 'Precio',
                            tamano: str = 'Tamano',
                            operacion: str = 'Operacion',
                            distrito: str = 'Distrito',
                            nueva_col: str = 'Precio_predicho') -> pd.DataFrame:
    """
    Función que calcula la predicción del precio que tendría una vivienda cada año, basándose en el precio de 2025, su tamaño y su distrito.

    La regla de 3 se basa en: si en 2025 esta vivienda vale X y la media de precio es Y, ¿cuánto valdría la vivienda en cada año si la media era Z?

    Parameters
    ----------
    df : Dataframe de Pandas
        Dataframe donde se quiere aplicar el cálculo.
    tipo : str
        "venta" o "alquiler". Elegir para aplicar sobre unos valores u otros.
    tamano : str
        Columna del dataframe donde se indica el tamaño.
        Por defecto: "Tamano"
    operacion : str
        Columna del dataframe donde se indica la operación.
        Por defecto: "Operacion"
    distrito : str
        Columna del dataframe donde se indica el distrito.
        Por defecto: "Distrito"
    nueva_col : str
        Columna del dataframe que se creará para apuntar las predicciones al pasado.
        Por defecto: "Precio_predicho"
    
    Returns
    ----------
    df : Dataframe de Pandas
        Dataframe actualizado con los nuevos datos (sustituir por el original o por una copia).
    """
    
    # Media de precios de alquiler o venta por distrito en 2025
    medias_2025 = df.loc[df[operacion] == tipo].groupby(distrito)[precio].mean()
    
    # Inicializar columna de precio predicho (si no existe)
    if nueva_col not in df.columns:
        df[nueva_col] = pd.NA
    
    # Máscara para filtrar por alquiler o venta
    mask = df[operacion] == tipo
    p_medio_antiguo = f'Precio_{tipo}'
    
    # Cálculo fila a fila para sacar la predicción usando la regla de 3
    df.loc[mask, nueva_col] = (
        df.loc[mask, precio] *
        (df.loc[mask, p_medio_antiguo] * df.loc[mask, tamano]) /
        df.loc[mask, distrito].map(medias_2025)
    )
    
    return df


def ajustar_predicciones(df):
    ajustados = []

    # Guardar orden original
    orden_original = df.index

    # Agrupar sin ordenar alfabéticamente
    for (distrito, operacion, ano), grupo in df.groupby(["Distrito", "Operacion", "Ano"], sort=False):
        
        # valor real medio según operación
        if operacion == "venta":
            real_medio = grupo["Precio_venta"].mean()
        else:
            real_medio = grupo["Precio_alquiler"].mean()
        
        pred_medio = grupo["Precio_predicho"].mean()
        sesgo = real_medio - pred_medio

        # desviación típica de errores
        errores = (real_medio - grupo["Precio_predicho"])
        sigma = errores.std()

        # aplicar corrección + ruido
        grupo = grupo.copy()
        grupo["Precio_predicho_ajustado"] = (
            grupo["Precio_predicho"] + sesgo + np.random.normal(0, sigma, size=len(grupo))
        )
        
        ajustados.append(grupo)

    # Concatenar y reordenar igual que el df original
    df_ajustado = pd.concat(ajustados)
    df_ajustado = df_ajustado.loc[orden_original]

    return df_ajustado