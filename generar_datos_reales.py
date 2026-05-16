import mysql.connector
import random
from datetime import datetime, timedelta
import certifi

db_config = {
    "host": "gateway01.us-east-1.prod.aws.tidbcloud.com",
    "port": 4000,
    "database": "asistencia_db",
    "user": "AvxXHDnHQFeZrfM.root",
    "password": "ytcjqQGK0qWnwuIa",
    "ssl_ca": certifi.where(), 
    "ssl_verify_cert": True,
    "ssl_verify_identity": True
}

def obtener_catalogos(conn):
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT AREA, NOMBRE FROM area")
    areas = {row['NOMBRE']: row['AREA'] for row in cursor.fetchall()}
    
    cursor.execute("SELECT Docente, AREA FROM docente")
    docentes = cursor.fetchall()
    
    cursor.execute("SELECT Turno FROM turno")
    turnos = [row['Turno'] for row in cursor.fetchall()]
    
    cursor.execute("SELECT SOFTWARE FROM software")
    softwares = [row['SOFTWARE'].strip() for row in cursor.fetchall()]
    
    cursor.close()
    return areas, docentes, turnos, softwares

try:
    print("Conectando a la base de datos...")
    conn = mysql.connector.connect(**db_config)
    areas, docentes, turnos, softwares = obtener_catalogos(conn)
    
    cursor = conn.cursor()
    insert_sql = """
        INSERT INTO `monitoreo` 
        (`AREA`, `NOMBRE DEL PROFESOR`, `INASISTENCIA`, `AULA`, `FECHA`, 
        `HORA DE ENTRADA`, `HORA DE SALIDA`, `TURNO`, `NUMERO DE ALUMNOS`, 
        `NUMERO DE COMPUTADORAS USADAS`, `TEMA`, `SOFTWARE`) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    fecha_base = datetime.now() - timedelta(days=90) 
    aulas_posibles = ["1 - A", "2 - B", "3 - C", "4 - A", "5 - D", "Lab Computo 1", "Lab Computo 2"]
    
    print("Generando 200 registros de historial operativo...")
    
    for i in range(200):
        docente_seleccionado = random.choice(docentes)
        profe_nombre = docente_seleccionado['Docente']
        area_id = str(docente_seleccionado['AREA']) 
        
        turno = random.choice(turnos)
        aula = random.choice(aulas_posibles)
        software = random.choice(softwares)
        tema = f"Unidad {random.randint(1, 5)} - Sesión {random.randint(1, 10)}"
        
        fecha_registro = fecha_base + timedelta(days=(i // 2))
        fecha_str = fecha_registro.strftime("%m/%d/%Y")
        
        asis = "Si" if random.random() > 0.90 else "No"
        
        es_anomalia = random.random() > 0.85
        if es_anomalia:
            alumnos = random.randint(10, 15)
            computadoras = random.randint(35, 40)
        else:
            alumnos = random.randint(25, 35)
            computadoras = alumnos + random.randint(-2, 2) 
            
        valores = (
            area_id, profe_nombre, asis, aula, fecha_str, 
            "08:00 a.m.", "10:00 a.m.", turno, alumnos, computadoras, tema, software
        )
        
        cursor.execute(insert_sql, valores)

    conn.commit()
    cursor.close()
    conn.close()
    print("¡Base de datos alimentada con éxito! Ahora puedes ejecutar el Dashboard Web.")

except Exception as e:
    print("Error en la base de datos:", e)