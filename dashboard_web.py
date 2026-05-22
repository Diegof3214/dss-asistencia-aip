import streamlit as st
import pandas as pd
import certifi
import mysql.connector
import motor_dss

st.set_page_config(page_title="DSS Predictivo", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .metric-card { 
        background-color: var(--secondary-background-color); 
        color: var(--text-color);
        border-radius: 10px; 
        padding: 20px; 
        text-align: center; 
        border-top: 4px solid #3B82F6; 
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Panel de Alertas Predictivas en Tiempo Real")
st.markdown("Sistema de Apoyo a la Decisión mediante Machine Learning")

db_config = motor_dss.db_config

@st.cache_data(ttl=60)
def cargar_datos():
    try:
        conn = mysql.connector.connect(**db_config)

        query = """
            SELECT am.*, a.NOMBRE AS NOMBRE_REAL_AREA 
            FROM alertas_ml am
            LEFT JOIN area a ON am.AREA = a.IDAREA
            WHERE am.AULA NOT IN ('Lab Computo 1', 'Lab Computo 2') 
              AND am.AULA NOT LIKE '%GRADO - SECCION%'
              AND YEAR(am.FECHA_PROCESAMIENTO) = 2026
            ORDER BY am.RIESGO_AUSENTISMO DESC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if 'NOMBRE_REAL_AREA' in df.columns:
            df['AREA'] = df['NOMBRE_REAL_AREA'].fillna(df['AREA'])
            df.drop(columns=['NOMBRE_REAL_AREA'], inplace=True)

        if 'IDMONITOREO' in df.columns:
            df.drop(columns=['IDMONITOREO'], inplace=True)

        return df

    except Exception as e:
        st.error(f"⚠️ Error al cargar Alertas Predictivas: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def cargar_historial():
    try:
        conn = mysql.connector.connect(**db_config)

        query = """
            SELECT m.`FECHA`, 
                   m.`NOMBRE DEL PROFESOR` AS PROFESOR, 
                   m.`TURNO`, 
                   a.`NOMBRE` AS AREA_NOMBRE, 
                   m.`AREA` AS AREA_NUM, 
                   m.`NUMERO DE ALUMNOS` AS ALUMNOS, 
                   m.`NUMERO DE COMPUTADORAS USADAS` AS EQUIPOS, 
                   m.`AULA`, 
                   m.`INASISTENCIA`
            FROM monitoreo m
            LEFT JOIN area a ON m.`AREA` = a.`IDAREA`
            WHERE m.`AULA` NOT IN ('Lab Computo 1', 'Lab Computo 2')
              AND m.`AULA` NOT LIKE '%GRADO - SECCION%'
              AND m.`FECHA` LIKE '%2026'
            ORDER BY m.`FECHA` DESC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if 'AREA_NOMBRE' in df.columns:
            df['AREA'] = df['AREA_NOMBRE'].fillna(df['AREA_NUM'])
            df.drop(columns=['AREA_NOMBRE', 'AREA_NUM'], inplace=True)

        return df

    except Exception as e:
        st.error(f"⚠️ Nota en Historial (Consulta Principal): {e}")

        try:
            conn = mysql.connector.connect(**db_config)

            query_fallback = """
                SELECT *, 
                       `NOMBRE DEL PROFESOR` AS PROFESOR, 
                       `NUMERO DE ALUMNOS` AS ALUMNOS, 
                       `NUMERO DE COMPUTADORAS USADAS` AS EQUIPOS 
                FROM monitoreo 
                WHERE `AULA` NOT IN ('Lab Computo 1', 'Lab Computo 2') 
                  AND `AULA` NOT LIKE '%GRADO - SECCION%'
                  AND `FECHA` LIKE '%2026'
            """

            df = pd.read_sql(query_fallback, conn)
            conn.close()

            if 'IDMONITOREO' in df.columns:
                df.drop(columns=['IDMONITOREO'], inplace=True)

            return df

        except Exception as e_fallback:
            st.error(f"❌ Error Crítico en Consulta de Respaldo: {e_fallback}")
            return pd.DataFrame()


def pintar_riesgo(val):
    if val == 'ALTO':
        return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'

    elif val == 'MEDIO':
        return 'background-color: #FEF3C7; color: #92400E; font-weight: 500;'

    elif val == 'BAJO':
        return 'background-color: #DCFCE7; color: #166534;'

    return ''


def pintar_anomalia(val):
    if isinstance(val, str) and 'ANOMALÍA' in val:
        return 'background-color: #FFEDD5; color: #C2410C; font-weight: bold;'

    return ''


df_alertas = cargar_datos()
df_historico = cargar_historial()

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2942/2942789.png",
        width=80
    )

    st.header("Controles")

    if st.button(
        "🚀 Re-entrenar Modelos",
        use_container_width=True,
        type="primary"
    ):

        with st.spinner('Procesando algoritmos...'):

            try:
                datos_crudos = motor_dss.obtener_datos()

                if not datos_crudos.empty:

                    alertas_nuevas = motor_dss.procesar_modelos(
                        datos_crudos
                    )

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

    if not df_alertas.empty or not df_historico.empty:

        areas_alertas = (
            df_alertas['AREA'].dropna().unique().tolist()
            if 'AREA' in df_alertas.columns else []
        )

        areas_historial = (
            df_historico['AREA'].dropna().unique().tolist()
            if 'AREA' in df_historico.columns else []
        )

        areas_unicas = sorted(
            list(set(areas_alertas + areas_historial))
        )

        filtro_area = st.multiselect(
            "Filtrar por Área Académica:",
            options=areas_unicas,
            default=[]
        )

        df_doc_alertas = df_alertas.copy()
        df_doc_historial = df_historico.copy()

        if filtro_area:

            if 'AREA' in df_doc_alertas.columns:
                df_doc_alertas = df_doc_alertas[
                    df_doc_alertas['AREA'].isin(filtro_area)
                ]

            if 'AREA' in df_doc_historial.columns:
                df_doc_historial = df_doc_historial[
                    df_doc_historial['AREA'].isin(filtro_area)
                ]

        docentes_alertas = (
            df_doc_alertas['PROFESOR'].dropna().unique().tolist()
            if 'PROFESOR' in df_doc_alertas.columns else []
        )

        docentes_historial = (
            df_doc_historial['PROFESOR'].dropna().unique().tolist()
            if 'PROFESOR' in df_doc_historial.columns else []
        )

        docentes_unicos = sorted(
            list(set(docentes_alertas + docentes_historial))
        )

        filtro_docente = st.multiselect(
            "Filtrar por Docente:",
            options=docentes_unicos,
            default=[]
        )

# =========================
# TABS
# =========================

tab_predictivo, tab_auditoria = st.tabs([
    "🔮 Inteligencia Artificial (Alertas)",
    "📜 Auditoría de Asistencia (Historial)"
])
# =========================
# TAB 1 - ALERTAS
# =========================

with tab_predictivo:

    if not df_alertas.empty:

        df_filtrado = df_alertas.copy()

        if 'AREA' in df_filtrado.columns and filtro_area:
            df_filtrado = df_filtrado[
                df_filtrado['AREA'].isin(filtro_area)
            ]

        if 'PROFESOR' in df_filtrado.columns and filtro_docente:
            df_filtrado = df_filtrado[
                df_filtrado['PROFESOR'].isin(filtro_docente)
            ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f'''
                <div class="metric-card">
                    <h3>Docentes Evaluados</h3>
                    <h2>{len(df_filtrado)}</h2>
                </div>
                ''',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f'''
                <div class="metric-card" style="border-top-color: #EF4444;">
                    <h3>Riesgo de Ausentismo</h3>
                    <h2 style="color: #EF4444;">
                        {
                            len(
                                df_filtrado[
                                    df_filtrado["NIVEL_RIESGO"] == "ALTO"
                                ]
                            )
                        }
                    </h2>
                </div>
                ''',
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f'''
                <div class="metric-card" style="border-top-color: #F59E0B;">
                    <h3>Anomalías en Infraestructura</h3>
                    <h2 style="color: #F59E0B;">
                        {
                            len(
                                df_filtrado[
                                    df_filtrado[
                                        "ANOMALIA_INFRAESTRUCTURA"
                                    ].str.contains("ANOMALÍA", na=False)
                                ]
                            )
                        }
                    </h2>
                </div>
                ''',
                unsafe_allow_html=True
            )

        df_mostrar = df_filtrado.drop(
            columns=['ID_ALERTA', 'IDMONITOREO'],
            errors='ignore'
        )


        if 'FECHA_PROCESAMIENTO' in df_mostrar.columns:

            df_mostrar['FECHA_PROCESAMIENTO'] = pd.to_datetime(
                df_mostrar['FECHA_PROCESAMIENTO']
            ).dt.strftime('%d/%m/%Y')

        if 'RIESGO_AUSENTISMO' in df_mostrar.columns:

            df_mostrar['RIESGO_AUSENTISMO'] = (
                df_mostrar['RIESGO_AUSENTISMO']
                .apply(lambda x: f"{x*100:.0f}%")
            )

        columnas_renombradas = {
            'FECHA_PROCESAMIENTO': 'Actualizado',
            'AREA': 'Área Académica',
            'PROFESOR': 'Docente',
            'RIESGO_AUSENTISMO': 'Prob. Falta (RF)',
            'NIVEL_RIESGO': 'Alerta',
            'AULA': 'Aula Asignada',
            'ANOMALIA_INFRAESTRUCTURA': 'Infraestructura'
        }

        df_mostrar.rename(
            columns=columnas_renombradas,
            inplace=True
        )

        tabla_estilizada = (
            df_mostrar.style
            .map(pintar_riesgo, subset=['Alerta'])
            .map(pintar_anomalia, subset=['Infraestructura'])
        )

        st.dataframe(
            tabla_estilizada,
            use_container_width=True,
            hide_index=True,
            height=400
        )

    else:
        st.info(
            "Presiona 'Re-entrenar Modelos' "
            "en la barra lateral para generar las primeras alertas."
        )

# =========================
# TAB 2 - HISTORIAL
# =========================

with tab_auditoria:

    if not df_historico.empty:

        st.write(
            "Verifica el historial detallado del año 2026 "
            "para contrastar y confirmar el contexto "
            "de las alertas predictivas."
        )

        df_hist_filtrado = df_historico.copy()

        if 'AREA' in df_hist_filtrado.columns and filtro_area:

            df_hist_filtrado = df_hist_filtrado[
                df_hist_filtrado['AREA'].isin(filtro_area)
            ]

        if 'PROFESOR' in df_hist_filtrado.columns and filtro_docente:

            df_hist_filtrado = df_hist_filtrado[
                df_hist_filtrado['PROFESOR'].isin(filtro_docente)
            ]

        busqueda_adicional = st.text_input(
            "🔍 Búsqueda rápida adicional "
            "en el historial (Ej. un aula o turno específico):"
        )

        if busqueda_adicional:

            df_hist_filtrado = df_hist_filtrado[
                df_hist_filtrado.astype(str).apply(
                    lambda x: x.str.contains(
                        busqueda_adicional,
                        case=False
                    )
                ).any(axis=1)
            ]

        if 'FECHA' in df_hist_filtrado.columns:

            df_hist_filtrado['FECHA'] = pd.to_datetime(
                df_hist_filtrado['FECHA']
            ).dt.strftime('%d/%m/%Y')

        columnas_renombradas_hist = {
            'FECHA': 'Fecha',
            'PROFESOR': 'Docente',
            'TURNO': 'Turno',
            'AREA': 'Área Académica',
            'ALUMNOS': 'Aforo Alumnos',
            'EQUIPOS': 'Equipos Usados',
            'AULA': 'Aula',
            'INASISTENCIA': 'Inasistencia'
        }

        columnas_existentes = {
            k: v
            for k, v in columnas_renombradas_hist.items()
            if k in df_hist_filtrado.columns
        }

        df_hist_filtrado.rename(
            columns=columnas_existentes,
            inplace=True
        )

        def resaltar_faltas(val):

            if str(val).upper() in [
                'FALTO',
                'INASISTENCIA',
                'TARDE',
                'SI'
            ]:

                return (
                    'background-color: #FEE2E2; '
                    'color: #991B1B; '
                    'font-weight: bold;'
                )

            return ''

        target_col = (
            'Inasistencia'
            if 'Inasistencia' in df_hist_filtrado.columns
            else (
                'INASISTENCIA'
                if 'INASISTENCIA' in df_hist_filtrado.columns
                else None
            )
        )

        if target_col:

            st.dataframe(
                df_hist_filtrado.style.map(
                    resaltar_faltas,
                    subset=[target_col]
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.dataframe(
                df_hist_filtrado,
                use_container_width=True,
                hide_index=True
            )

    else:
        st.warning(
            "No se encontraron registros "
            "en la tabla histórica (monitoreo)."
        )

        st.warning("No se encontraron registros en la tabla histórica (monitoreo).")