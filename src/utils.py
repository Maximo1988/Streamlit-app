import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def procesar_datos_apple(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    return df

def filtrar_por_fecha(df, fecha_inicio, fecha_fin):
    return df[df["Date"].between(fecha_inicio, fecha_fin)]

def calcular_retornos(df):
    """Calcula retornos diarios y acumulados"""
    df_copy = df.copy()
    df_copy['retorno_diario'] = df_copy['Close'].pct_change() * 100
    df_copy['retorno_acumulado'] = (1 + df_copy['Close'].pct_change()).cumprod() - 1
    df_copy['retorno_acumulado'] = df_copy['retorno_acumulado'] * 100
    return df_copy

def analizar_convergencia_precios(df):
    """Analiza tendencias de convergencia de precios"""
    df_copy = df.copy()
    # Media móvil de 30 días
    df_copy['MA_30'] = df_copy['Close'].rolling(window=30).mean()
    # Media móvil de 90 días
    df_copy['MA_90'] = df_copy['Close'].rolling(window=90).mean()
    # Desviación estándar
    df_copy['std_30'] = df_copy['Close'].rolling(window=30).std()
    return df_copy

def kmeans_clustering(df, n_clusters=3):
    """Agrupa períodos similares usando K-means"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import numpy as np
    
    df_copy = df.copy()
    
    # Seleccionar features para clustering
    features = ['Close', 'Volume', 'retorno_diario']
    
    # Calcular retorno diario si no existe
    if 'retorno_diario' not in df_copy.columns:
        df_copy['retorno_diario'] = df_copy['Close'].pct_change() * 100
    
    # Preparar datos (eliminar NaN)
    df_features = df_copy[features].dropna().copy()
    
    # Normalizar datos
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(df_features)
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_normalized)
    
    # Agregar clusters al dataframe
    df_copy = df_copy.iloc[len(df_copy) - len(clusters):].copy()
    df_copy['cluster'] = clusters
    
    return df_copy, kmeans

def random_forest_prediccion(df, test_size=0.2):
    """Predice si el precio subirá o bajará el siguiente dia"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    df_copy = df.copy()
    
    # Crear target: 1 si el precio sube, 0 si baja
    df_copy['target'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int)
    
    # Crear features técnicas
    df_copy['MA_5'] = df_copy['Close'].rolling(window=5).mean()
    df_copy['MA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['retorno_diario'] = df_copy['Close'].pct_change() * 100
    df_copy['rsi'] = calcular_rsi(df_copy['Close'])
    df_copy['volatilidad'] = df_copy['Close'].rolling(window=20).std()
    df_copy['volume_ratio'] = df_copy['Volume'] / df_copy['Volume'].rolling(window=20).mean()
    
    # Seleccionar features y eliminar NaN
    feature_cols = ['MA_5', 'MA_20', 'retorno_diario', 'rsi', 'volatilidad', 'volume_ratio']
    df_model = df_copy[feature_cols + ['target']].dropna()
    
    if len(df_model) < 10:
        return None, None, None, None, None, None
    
    X = df_model[feature_cols]
    y = df_model['target']
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Predicción
    y_pred = rf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Predicción para el último día
    ultima_fila = X_scaled[-1].reshape(1, -1)
    prediccion_proxima = rf.predict(ultima_fila)[0]
    probabilidad = rf.predict_proba(ultima_fila)[0]
    
    # Crear histórico de predicciones vs realidad
    y_pred_all = rf.predict(X_scaled)
    historico = df_copy[feature_cols + ['target', 'Date']].dropna().copy()
    historico['prediccion'] = y_pred_all
    historico = historico.tail(60)  # Últimos 60 días
    
    return rf, accuracy, feature_importance, prediccion_proxima, probabilidad, historico

def calcular_rsi(prices, period=14):
    """Calcula el Relative Strength Index"""
    import numpy as np
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
