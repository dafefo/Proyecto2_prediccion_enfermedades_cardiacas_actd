import psycopg2
engine = psycopg2.connect(
    dbname="danielchrister",
    user="danielchrister",
    password="danielactd",
    host="danielchrister.cpucxidqdwyr.us-east-1.rds.amazonaws.com",
    port='5432'
)
engine.autocommit = True
cursor = engine.cursor()

sql_script = """
CREATE TABLE datacardiaca(
    age_group char(20)
    sex INT
    cp INT
    trestbps_group char(50)
    chol_group char(30)
    fbs INT
    restecg INT
    exang INT
    slope INT
    ca INT
    thal INT
    cardiac boolean
);
"""

sql_script += 'INSERT INTO datacardiaca (age_group, sex, cp, trestbps_group, chol_group, fbs, restecg, exang, slope, ca, thal, cardiac) VALUES\n'

values = []
for i, row in df.interrows():
    value = f"({row['age_group']},{int(row['sex'])},{row['cp']},{row['trestbps_group']},{row['chol_group']},{int(row['fbs'])},{int(row['restecg'])},{int(row['exang'])},{int(row['slope'])},{int(row['ca'])},{int(row['thal'])},{row['cardiac']})"
    values.append(value)

#Guardar el script
with open("../../Decimo Semestre/Analitica computacional/Proyecto 2 Enfermedades cardiacas/datacardica.sql", "w")as sql_file:
    sql_file.write(sql_script)