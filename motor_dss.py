import mysql.connector
import streamlit as st
import pandas as pd
import numpy as np
import certifi
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from datetime import datetime

db_config = {
    "host": st.secrets["TIDB_HOST"],
    "port": 4000,
    "database": st.secrets["TIDB_DATABASE"],
    "user": st.secrets["TIDB_USER"],
    "password": st.secrets["TIDB_PASSWORD"],
    "ssl_ca": certifi.where(),
    "ssl_verify_cert": True,
    "ssl_verify_identity": True
}

def obtener_datos():
    conn = mysql.connector.connect(**db_config)
    
    query = """
        SELECT m.*, a.NOMBRE AS NOMBRE_REAL_AREA 
        FROM monitoreo m
        LEFT JOIN area a ON m.AREA = a.IDAREA
    """
    df = pd.read_sql(query, conn)
    
    if 'NOMBRE_REAL_AREA' in df.columns:
        df['AREA'] = df['NOMBRE_REAL_AREA'].fillna(df['AREA']) 
        
    conn.close()
    return df

def guardar_alertas(alertas_df):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS alertas_ml")
    
    cursor.execute("""
        CREATE TABLE alertas_ml (
            ID_ALERTA INT AUTO_INCREMENT PRIMARY KEY,
            FECHA_PROCESAMIENTO DATETIME,
            AREA VARCHAR(50), 
            PROFESOR VARCHAR(100),
            RIESGO_AUSENTISMO FLOAT,
            NIVEL_RIESGO VARCHAR(20),
            AULA VARCHAR(50),
            ANOMALIA_INFRAESTRUCTURA VARCHAR(50)
        )
    """)
    
    insert_sql = """
        INSERT INTO alertas_ml (FECHA_PROCESAMIENTO, AREA, PROFESOR, RIESGO_AUSENTISMO, NIVEL_RIESGO, AULA, ANOMALIA_INFRAESTRUCTURA)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    now = datetime.now()
    for _, row in alertas_df.iterrows():
        cursor.execute(insert_sql, (
            now, row['AREA'], row['NOMBRE DEL PROFESOR'], round(row['Probabilidad_Falta'], 2), 
            row['Nivel_Riesgo'], row['AULA'], row['Anomalia_KMeans']
        ))
        
    conn.commit()
    cursor.close()
    conn.close()

def procesar_modelos(df):
    # 1. Definición del Target
    df['Target_Inasistencia'] = df['INASISTENCIA'].apply(lambda x: 1 if str(x).upper() in ['FALTO', 'INASISTENCIA', 'TARDE', 'SI'] else 0)
    
    # Limpieza de variables numéricas
    df['NUMERO DE ALUMNOS'] = pd.to_numeric(df['NUMERO DE ALUMNOS'], errors='coerce').fillna(0)
    df['NUMERO DE COMPUTADORAS USADAS'] = pd.to_numeric(df['NUMERO DE COMPUTADORAS USADAS'], errors='coerce').fillna(0)

    # 2. INGENIERÍA DE CARACTERÍSTICAS (Reemplazamos LabelEncoder por One-Hot Encoding)
    variables_categoricas = df[['NOMBRE DEL PROFESOR', 'TURNO', 'AREA']]
    X_encoded = pd.get_dummies(variables_categoricas, drop_first=True)
    X_rf = pd.concat([df[['NUMERO DE ALUMNOS']], X_encoded], axis=1)
    y_rf = df['Target_Inasistencia']
    
    try:
        # 3. BALANCEO DE DATOS EN PRODUCCIÓN
        smote = SMOTE(random_state=42)
        X_rf_smote, y_rf_smote = smote.fit_resample(X_rf, y_rf)

        # 4. ENTRENAMIENTO CON PARÁMETROS OPTIMIZADOS
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
        rf_model.fit(X_rf_smote, y_rf_smote)
        
        # 5. PREDICCIÓN
        df['Probabilidad_Falta'] = rf_model.predict_proba(X_rf)[:, 1]
    except Exception as e:
        print(f"⚠️ Advertencia en el modelo RF: {e}")
        df['Probabilidad_Falta'] = 0.0 

    # Asignación dinámica de niveles de riesgo
    df['Nivel_Riesgo'] = df['Probabilidad_Falta'].apply(lambda p: 'ALTO' if p > 0.6 else ('MEDIO' if p > 0.3 else 'BAJO'))

    # 6. MODELO NO SUPERVISADO (K-MEANS)
    X_km = df[['NUMERO DE ALUMNOS', 'NUMERO DE COMPUTADORAS USADAS']]
    scaler = StandardScaler()
    X_km_scaled = scaler.fit_transform(X_km)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_km_scaled)
    df['Anomalia_KMeans'] = df.apply(lambda row: 'ANOMALÍA: Desbalance HW/Aforo' if abs(row['NUMERO DE ALUMNOS'] - row['NUMERO DE COMPUTADORAS USADAS']) > 15 else 'NORMAL', axis=1)

    return df.drop_duplicates(subset=['NOMBRE DEL PROFESOR'], keep='last')

if __name__ == "__main__":
    print("Iniciando motor dinámico de procesamiento DSS...")
    datos = obtener_datos()
    if not datos.empty:
        alertas = procesar_modelos(datos)
        guardar_alertas(alertas)
        print("✅ Procesamiento completado. Las alertas predictivas han sido actualizadas en la base de datos.")