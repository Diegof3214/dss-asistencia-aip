import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import motor_dss

def calcular_metricas_optimizadas():
    print("⏳ Obteniendo datos históricos de TiDB...")
    df = motor_dss.obtener_datos()
    
    if df.empty:
        print("❌ No hay datos en la base de datos.")
        return

    # 1. Definir Target (1 para inasistencias/tardanzas, 0 para asistencias)
    df['Target_Inasistencia'] = df['INASISTENCIA'].apply(lambda x: 1 if str(x).upper() in ['FALTO', 'INASISTENCIA', 'TARDE', 'SI'] else 0)
    
    # 2. Ingeniería de Características
    df['NUMERO DE ALUMNOS'] = pd.to_numeric(df['NUMERO DE ALUMNOS'], errors='coerce').fillna(0)
    variables_categoricas = df[['NOMBRE DEL PROFESOR', 'TURNO', 'AREA']]
    
    # One-Hot Encoding para evitar jerarquías falsas en los textos
    X_encoded = pd.get_dummies(variables_categoricas, drop_first=True)
    X_num = df[['NUMERO DE ALUMNOS']]
    X = pd.concat([X_num, X_encoded], axis=1)
    y = df['Target_Inasistencia']
    
    # 3. División de datos (70% entrenamiento, 30% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. Aplicación de SMOTE (Balanceo sintético exclusivo para entrenamiento)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # 5. Modelo Ganador (Configuración manual que evita el sobreajuste)
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,           # Profundidad limitada para generalizar mejor
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train_smote, y_train_smote)
    
    # 6. Predicciones
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # 7. Cálculo de Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc_roc = "No calculable"
        
    cm = confusion_matrix(y_test, y_pred)

    print("\n==================================================")
    print(" 🚀 MÉTRICAS DE TU MODELO GANADOR (RANDOM FOREST)")
    print("==================================================")
    print(f" Accuracy:   {accuracy*100:.2f}%")
    print(f" Precision:  {precision*100:.2f}%")
    print(f" Recall:     {recall*100:.2f}%")
    print(f" F1-Score:   {f1:.4f}")
    if isinstance(auc_roc, float):
        print(f" AUC-ROC:    {auc_roc:.4f}")
    else:
        print(f" AUC-ROC:    {auc_roc}")
        
    print("\n 📌 MATRIZ DE CONFUSIÓN")
    if len(cm) == 2:
        print(f" Verdaderos Negativos (VN):  {cm[0][0]}")
        print(f" Falsos Positivos (FP):      {cm[0][1]}")
        print(f" Falsos Negativos (FN):      {cm[1][0]}")
        print(f" Verdaderos Positivos (VP):  {cm[1][1]}")
    else:
        print(cm)

if __name__ == "__main__":
    calcular_metricas_optimizadas()