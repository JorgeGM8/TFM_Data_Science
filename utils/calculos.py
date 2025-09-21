import pandas as pd
import numpy as np
import warnings

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


def ajustar_predicciones_con_limites(df:pd.DataFrame, 
                                   max_desv_venta:float=0.15, 
                                   max_desv_alquiler:float=0.05,
                                   aplicar_ruido:bool=True,
                                   factor_ruido:float=1.0,
                                   seed:int=None):
    """
    Ajusta predicciones por grupos (Distrito, Operacion, Ano) con límites de desviación.
    
    Parameters
    -----------
    df : DataFrame de Pandas
        DataFrame con las predicciones originales.
    max_desv_venta : float
        Máxima desviación permitida para ventas (ej: 0.15 = 15%).
    max_desv_alquiler : float  
        Máxima desviación permitida para alquileres (ej: 0.05 = 5%).
    aplicar_ruido : bool
        Si aplicar ruido aleatorio realista.
    factor_ruido : float
        Factor para escalar el ruido (1.0 = ruido completo).
    seed : int
        Semilla para reproducibilidad del ruido aleatorio.
        
    Returns
    --------
    DataFrame con columna "Precio_ajustado".
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Verificar columnas necesarias
    required_cols = ['Distrito', 'Operacion', 'Ano', 'Precio_venta', 'Precio_alquiler', 'Precio_predicho']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan las columnas: {missing_cols}")
    
    ajustados = []
    orden_original = df.index
    
    # Agrupar datos por distrito, año y tipo de operación
    for (distrito, operacion, ano), grupo in df.groupby(["Distrito", "Operacion", "Ano"], sort=False):
        
        # Valor real medio según operación
        if operacion == "venta":
            real_medio = grupo["Precio_venta"].mean()
            max_desviacion = max_desv_venta
        else:
            real_medio = grupo["Precio_alquiler"].mean()
            max_desviacion = max_desv_alquiler
        
        pred_medio = grupo["Precio_predicho"].mean()
        sesgo = real_medio - pred_medio
        
        # Calcular límites de cambio antes del ajuste
        limite_inf = grupo["Precio_predicho"] * (1 - max_desviacion)
        limite_sup = grupo["Precio_predicho"] * (1 + max_desviacion)
        
        # Aplicar corrección de sesgo
        grupo = grupo.copy()
        predicciones_corregidas = grupo["Precio_predicho"] + sesgo
        
        # Aplicar ruido si se pide
        if aplicar_ruido:
            # Desviación típica de errores
            errores = real_medio - grupo["Precio_predicho"]
            sigma = errores.std()
            
            # Evitar sigma = 0 o NaN (da lugar a errores)
            if pd.isna(sigma) or sigma == 0:
                sigma = abs(real_medio) * 0.02  # 2% como mínimo
            
            ruido = np.random.normal(0, sigma * factor_ruido, size=len(grupo))
            predicciones_con_ruido = predicciones_corregidas + ruido
        else:
            predicciones_con_ruido = predicciones_corregidas
        
        # Aplicar límites para no desviarse demasiado del precio predicho
        grupo["Precio_ajustado"] = np.clip(
            predicciones_con_ruido, 
            limite_inf, 
            limite_sup
        )
        
        ajustados.append(grupo)
    
    # Concatenar y reordenar igual que el df original
    df_ajustado = pd.concat(ajustados)
    df_ajustado = df_ajustado.loc[orden_original]
    
    return df_ajustado


def ajustar_predicciones_hibrido(df:pd.DataFrame, 
                                alpha:float=0.5,
                                max_desv_venta:float=0.15, 
                                max_desv_alquiler:float=0.05,
                                aplicar_ruido:bool=True,
                                seed:int=None):
    """
    Enfoque híbrido: Primero ajusta predicciones por grupos (Distrito, Operacion, Ano) con límites de desviación.
    Luego, ajusta individualmente.
    
    Parameters
    -----------
    df : DataFrame de Pandas
        DataFrame con las predicciones originales.
    max_desv_venta : float
        Máxima desviación permitida para ventas (ej: 0.15 = 15%).
    max_desv_alquiler : float  
        Máxima desviación permitida para alquileres (ej: 0.05 = 5%).
    aplicar_ruido : bool
        Si aplicar ruido aleatorio realista.
    factor_ruido : float
        Factor para escalar el ruido (1.0 = ruido completo).
    seed : int
        Semilla para reproducibilidad del ruido aleatorio.
        
    Returns
    --------
    DataFrame con columna "Precio_ajustado".
    """
    # Silenciar avisos ya revisados
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Paso 1: Ajuste por grupos
    df_grupos = ajustar_predicciones_con_limites(
        df, max_desv_venta, max_desv_alquiler, 
        aplicar_ruido=aplicar_ruido, seed=seed
    )
    
    # Paso 2: Ajuste individual suave sobre el resultado
    df_final = df_grupos.copy()
    
    # Calcular precio real individual
    df_final['Precio_real'] = np.where(
        df_final['Operacion'] == 'venta',
        df_final['Precio_venta'],
        df_final['Precio_alquiler']
    )
    
    # Suavización individual sobre el precio ya ajustado por grupos
    desv_rel = (df_final['Precio_ajustado'] - df_final['Precio_real']) / df_final['Precio_real']
    desv_rel = desv_rel.fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)
    
    factor = np.exp(-alpha * np.abs(desv_rel))
    ajuste_individual = df_final['Precio_real'] + factor * (df_final['Precio_ajustado'] - df_final['Precio_real'])
    
    # Aplicar límites finales
    limite_inf = np.where(
        df_final['Operacion'] == 'venta',
        df_final['Precio_predicho'] * (1 - max_desv_venta),
        df_final['Precio_predicho'] * (1 - max_desv_alquiler)
    )
    limite_sup = np.where(
        df_final['Operacion'] == 'venta',
        df_final['Precio_predicho'] * (1 + max_desv_venta),
        df_final['Precio_predicho'] * (1 + max_desv_alquiler)
    )
    
    df_final['Precio_ajustado'] = np.clip(ajuste_individual, limite_inf, limite_sup)
    
    return df_final


def ajustar_precios_simple(df:pd.DataFrame, alpha=0.5, max_desv_venta=0.15, max_desv_alquiler=0.05):
    """
    Ajusta los precios predichos acercándolos a la media real,
    pero limitando el ajuste a ±max_desv del valor predicho.
    No aplica ruido ni sesgos.
    
    Parameters
    ----------
    df : Dataframe de Pandas
        Columnas necesarias:
        ['Operacion','Distrito','Ano','Precio_venta','Precio_alquiler',
         'Precio_predicho','Tamano']
    alpha : float
        Fuerza del ajuste (0 = sin ajuste, 1 = fuerte).
    max_desv_venta : float
        Máxima desviación permitida respecto al predicho para ventas (ej: 0.1 = 10%).
    max_desv_alquiler : float
        Máxima desviación permitida respecto al predicho para alquileres (ej: 0.05 = 5%).
    
    Returns
    --------
    df : DataFrame de Pandas
        Dataframe actualizado con columna nueva "Precio_ajustado".
    """

    df = df.copy()
    
    # Verificar que las columnas necesarias existen
    required_cols = ['Operacion', 'Precio_venta', 'Precio_alquiler', 'Tamano', 'Precio_predicho']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan las columnas: {missing_cols}")
    
    # Convertir a numérico las columnas necesarias
    numeric_cols = ['Precio_venta', 'Precio_alquiler', 'Tamano', 'Precio_predicho']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calcular precio real
    df['Precio_real'] = np.where(
        df['Operacion'] == 'venta',
        df['Precio_venta'] * df['Tamano'],
        df['Precio_alquiler'] * df['Tamano']
    )
    
    # Verificar que no hay valores nulos o ceros en Precio_real
    if df['Precio_real'].isna().any():
        print("Advertencia: Se encontraron valores NaN en Precio_real")
        df['Precio_real'] = df['Precio_real'].fillna(df['Precio_predicho'])
    
    if (df['Precio_real'] == 0).any():
        print("Advertencia: Se encontraron valores cero en Precio_real")
        df['Precio_real'] = df['Precio_real'].replace(0, df['Precio_predicho'])
    
    # Calcular desviación relativa con manejo de divisiones por cero
    desv_rel = (df['Precio_predicho'] - df['Precio_real']) / df['Precio_real']
    
    # Manejar valores infinitos o NaN
    desv_rel = desv_rel.replace([np.inf, -np.inf], 0)
    desv_rel = desv_rel.fillna(0)
    
    # Calcular factor de ajuste
    try:
        factor = np.exp(-alpha * np.abs(desv_rel))
    except Exception as e:
        print(f"Error al calcular el factor: {e}")
        print(f"Tipo de desv_rel: {type(desv_rel)}")
        print(f"Valores únicos de desv_rel: {desv_rel.unique()[:10]}")  # Solo primeros 10
        raise
    
    # Calcular ajuste
    ajuste = df['Precio_real'] + factor * (df['Precio_predicho'] - df['Precio_real'])
    
    # Definir límites
    limite_inf = np.where(
        df['Operacion'] == 'venta',
        df['Precio_predicho'] * (1 - max_desv_venta),
        df['Precio_predicho'] * (1 - max_desv_alquiler)
    )
    limite_sup = np.where(
        df['Operacion'] == 'venta',
        df['Precio_predicho'] * (1 + max_desv_venta),
        df['Precio_predicho'] * (1 + max_desv_alquiler)
    )
    
    # Aplicar límites
    df['Precio_ajustado'] = np.clip(ajuste, limite_inf, limite_sup)
    
    return df