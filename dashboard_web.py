import streamlit as st
import pandas as pd
import certifi
import mysql.connector
import motor_dss

st.set_page_config(page_title="DSS Predictivo", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #F8FAFC; border-radius: 10px; padding: 20px; text-align: center; border-top: 4px solid #3B82F6; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Panel de Alertas Predictivas en Tiempo Real")
st.markdown("Sistema de Apoyo a la Decisión mediante Machine Learning")

db_config = motor_dss.db_config

@st.cache_data(ttl=60)
def cargar_datos():
    try:
        conn = mysql.connector.connect(**db_config)
        df = pd.read_sql("SELECT * FROM alertas_ml ORDER BY RIESGO_AUSENTISMO DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def cargar_historial():
    try:
        conn = mysql.connector.connect(**db_config)
        query = """
            SELECT m.FECHA, m.NOMBRE_DEL_PROFESOR AS PROFESOR, m.TURNO, 
                   a.NOMBRE AS AREA, m.NUMERO_DE_ALUMNOS AS ALUMNOS, m.INASISTENCIA
            FROM monitoreo m
            LEFT JOIN area a ON m.AREA = a.IDAREA
            ORDER BY m.FECHA DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception:
        try:
            conn = mysql.connector.connect(**db_config)
            df = pd.read_sql("SELECT * FROM monitoreo", conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()

def pintar_riesgo(val):
    if val == 'ALTO': return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
    elif val == 'MEDIO': return 'background-color: #FEF3C7; color: #92400E; font-weight: 500;'
    elif val == 'BAJO': return 'background-color: #DCFCE7; color: #166534;'
    return ''

def pintar_anomalia(val):
    if isinstance(val, str) and 'ANOMALÍA' in val: return 'background-color: #FFEDD5; color: #C2410C; font-weight: bold;'
    return ''

df_alertas = cargar_datos()
df_historico = cargar_historial()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942789.png", width=80)
    st.header("Controles")
    if st.button("🚀 Re-entrenar Modelos", use_container_width=True, type="primary"):
        with st.spinner('Procesando algoritmos...'):
            try:
                datos_crudos = motor_dss.obtener_datos()
                if not datos_crudos.empty:
                    alertas_nuevas = motor_dss.procesar_modelos(datos_crudos)
                    motor_dss.guardar_alertas(alertas_nuevas)
                    st.success("¡Base actualizada!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("Faltan datos históricos.")
            except Exception as e:
                st.error(f"Error: {e}")
                
    st.markdown("---")
    st.header("🔍 Filtros de Búsqueda")
    
    filtro_area = []
    filtro_docente = []
    
    if not df_alertas.empty:
        if 'AREA' in df_alertas.columns:
            areas_unicas = df_alertas['AREA'].dropna().unique().tolist()
            filtro_area = st.multiselect("Filtrar por Área:", options=areas_unicas, default=[])
            
        df_para_docentes = df_alertas.copy()
        if filtro_area:
            df_para_docentes = df_para_docentes[df_para_docentes['AREA'].isin(filtro_area)]
            
        docentes_unicos = df_para_docentes['PROFESOR'].dropna().unique().tolist()
        filtro_docente = st.multiselect("Filtrar por Docente:", options=docentes_unicos, default=[])

tab_predictivo, tab_auditoria = st.tabs(["🔮 Inteligencia Artificial (Alertas)", "📜 Auditoría de Asistencia (Historial)"])

with tab_predictivo:
    if not df_alertas.empty:
        df_filtrado = df_alertas.copy()
        
        if 'AREA' in df_alertas.columns and filtro_area:
            df_filtrado = df_filtrado[df_filtrado['AREA'].isin(filtro_area)]
            
        if filtro_docente:
            df_filtrado = df_filtrado[df_filtrado['PROFESOR'].isin(filtro_docente)]

        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f'<div class="metric-card"><h3>Docentes Evaluados</h3><h2>{len(df_filtrado)}</h2></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card" style="border-top-color: #EF4444;"><h3>Riesgo de Ausentismo</h3><h2 style="color: #EF4444;">{len(df_filtrado[df_filtrado["NIVEL_RIESGO"] == "ALTO"])}</h2></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card" style="border-top-color: #F59E0B;"><h3>Anomalías en Infraestructura</h3><h2 style="color: #F59E0B;">{len(df_filtrado[df_filtrado["ANOMALIA_INFRAESTRUCTURA"].str.contains("ANOMALÍA", na=False)])}</h2></div>', unsafe_allow_html=True)
        
        df_mostrar = df_filtrado.drop(columns=['ID_ALERTA'], errors='ignore')
        if 'RIESGO_AUSENTISMO' in df_mostrar.columns:
            df_mostrar['RIESGO_AUSENTISMO'] = df_mostrar['RIESGO_AUSENTISMO'].apply(lambda x: f"{x*100:.0f}%")
            
        columnas_renombradas = {
            'FECHA_PROCESAMIENTO': 'Actualizado', 
            'AREA': 'Área Académica',
            'PROFESOR': 'Docente', 
            'RIESGO_AUSENTISMO': 'Prob. Falta (RF)', 
            'NIVEL_RIESGO': 'Alerta', 
            'ANOMALIA_INFRAESTRUCTURA': 'Infraestructura'
        }
        df_mostrar.rename(columns=columnas_renombradas, inplace=True)
        
        tabla_estilizada = df_mostrar.style.map(pintar_riesgo, subset=['Alerta']).map(pintar_anomalia, subset=['Infraestructura'])
        st.dataframe(tabla_estilizada, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("Presiona 'Re-entrenar Modelos' en la barra lateral para generar las primeras alertas.")

with tab_auditoria:
    if not df_historico.empty:
        st.write("Verifica el historial de los docentes para confirmar el contexto de las alertas generadas por el modelo predictivo.")
        
        busqueda = st.text_input("🔍 Buscar profesor específico en el historial:")
        
        df_hist_mostrar = df_historico.copy()
        if busqueda:
            df_hist_mostrar = df_hist_mostrar[df_hist_mostrar.astype(str).apply(lambda x: x.str.contains(busqueda, case=False)).any(axis=1)]
            
        def resaltar_faltas(val):
            if str(val).upper() in ['FALTO', 'INASISTENCIA', 'TARDE', 'SI']:
                return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
            return ''
            
        if 'INASISTENCIA' in df_hist_mostrar.columns:
            st.dataframe(df_hist_mostrar.style.map(resaltar_faltas, subset=['INASISTENCIA']), use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_hist_mostrar, use_container_width=True, hide_index=True)
    else:
        st.warning("No se encontraron registros en la tabla histórica (monitoreo).")