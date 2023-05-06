import pandas as pd
import numpy as np
import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc


#Apertura de los datos Daniel
#data_cardiaca = pd.read_csv("Analitica computacional/Proyecto1 Enfermedades cardiacas/cleveland_data.csv")

#Apertura datos Christer
#data_cardiaca = pd.read_csv("cleveland_data.csv")
"""data_cardiaca = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None)
data_cardiaca.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol','fbs', 'restecg','thalac', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'num'
]
#En las variables ca y thal hay datos faltantes que se ubican con el simbolo ?
print(data_cardiaca.loc[data_cardiaca["ca"]=="?"])
#en la variable ca estan en las posiciones 166,192,287,302
print(data_cardiaca.loc[data_cardiaca["thal"]=="?"])
#en la variable Thal estan en 87 y 266
#Es decir en total se deben eliminar 6 datos de la base de 303

#Remover los datos faltantes
faltantes = np.array([87,166,192,266,287,302])
for i in faltantes:
    data_cardiaca = data_cardiaca.drop(i)

#Pasar las columans de tipo object a numeros
data_cardiaca["ca"]=pd.to_numeric(data_cardiaca["ca"])
data_cardiaca["thal"]=pd.to_numeric(data_cardiaca["thal"])

estadisticas = data_cardiaca.describe()
#Discretizar las variables
#se crea una varaible categorica de 1 si esta diagnosticado con enfermedad cardiaca y 0 no
data_cardiaca["cardiac"]=np.where(data_cardiaca["num"]>0,True,False)

#Cambiar los nombres de sexo
data_cardiaca["sex"]=np.where(data_cardiaca["sex"]>0,"Hombre","Mujer")

#Cambiar los nombres de fbs
data_cardiaca["fbs"]=np.where(data_cardiaca["fbs"]>0,">120","<120")

#Cambiar los nombres de exang
data_cardiaca["exang"]=np.where(data_cardiaca["exang"]>0,"Si","No")

#Cambiar los nombres de slope
data_cardiaca["slope"]= pd.cut(data_cardiaca["slope"], bins=[0,1,2,3],labels=["positiva","plana","negativa"])

#Cambiar los nombres de thal
data_cardiaca["thal"]= pd.cut(data_cardiaca["thal"], bins=[0,3,6,7],labels=["Normal","Fijo","Reversible"])

#Cambiar los nombres de cp
data_cardiaca["cp"]= pd.cut(data_cardiaca["cp"], bins=[0,1,2,3,4],labels=["Angina normal","Angina atipica","No angina","Asintomatico"])

#Cambiar los nombres de restecg
data_cardiaca["restecg"]= pd.cut(data_cardiaca["restecg"], bins=[-1,0,1,2],labels=["Normal","ST anormal","Hipertrofia ventricular"])

#Se crea una avraible categorica de la edad. 
edad_discrt = pd.cut(data_cardiaca["age"],bins = [0,50,100], labels = ["Joven","Mayor"])
data_cardiaca.insert(1,"age_group", edad_discrt)

#Se crea una variable categorica para el colesterol
chol_discrt = pd.cut(data_cardiaca["chol"],bins=[0,200,240,600], labels = ["normal","alto","muy alto"])
data_cardiaca.insert(6,"chol_group",chol_discrt)

#Se crea una varibale categorica para la presion sanguinea en reposo
trestbps_discrt = pd.cut(data_cardiaca["trestbps"],bins=[0,119,129,139,179,210], labels=["normal","elevada","presion arterial nivel 1","presion arterial nivel 2","crisis"])
data_cardiaca.insert(5,"trestbps_group", trestbps_discrt)
print(data_cardiaca)"""

#Abrir los datos desde RDS
import psycopg2
engine = psycopg2.connect(
    dbname="datacardiaca",
    user="danielchrister",
    password="danielactd",
    host="danielchrister.cpucxidqdwyr.us-east-1.rds.amazonaws.com",
    port='5432'
)
cursor = engine.cursor()
query = """
SELECT *
FROM datacardiaca
;"""
cursor.execute(query)
result = cursor.fetchall()
result
import pandas as pd
df = pd.read_sql(query, engine)
df.drop(df.tail(594).index,inplace=True)
df["cp"]= pd.cut(df["cp"], bins=[0,1,2,3,4],labels=["Angina normal","Angina atipica","No angina","Asintomatico"])

#Cp
CP = px.histogram(df, x = "cp", color = "cardiac", text_auto='.1f', barnorm= "percent",
                   category_orders={"cp":["Angina normal","Angina atipica","No angina","Asintomatico"],
                                    "cardiac":[True,False]},
                    color_discrete_map={True:"#E99220", False:"#BAB0AC"},
                    title="Relación dolor de pecho y enfermedad cardiaca")
CP.update_layout(legend_title = "Sufre de enfermedad cardiaca",
                  xaxis_title = "Dolor de Pecho",
                  yaxis_title = "Conteo(Porcentaje)",
                  title_x = 0.5,
                  font = dict(size = 20))
CP.update_layout({'plot_bgcolor': 'white'})
#Presion arterial
PA = px.histogram(df, x = "trestbps_group", color = "cardiac", text_auto='.1f', barnorm= "percent",
                   category_orders={"trestbps_group":["normal","elevada","presionarterialnivel1","presionarterialnivel2","crisis"],
                                    "cardiac":[True,False]},
                                    color_discrete_map={True:"#E99220", False:"#BAB0AC"},
                                    title="Relación presión arterial en reposo y enfermedad cardiaca")
PA.update_layout(legend_title = "Sufre de enfermedad cardiaca",
                  xaxis_title = "Presión Arterial",
                  yaxis_title = "Conteo(Porcentaje)",
                  title_x = 0.5,
                  font = dict(size = 14))
PA.update_layout({'plot_bgcolor': 'white'})
#Colesterol
Cole = px.histogram(df, x = "chol_group", color = "cardiac", text_auto='.1f', barnorm= "percent",
                   category_orders={"chol_group":["normal","alto","muyalto"],
                                    "cardiac":[True,False]},
                    color_discrete_map={True:"#E99220", False:"#BAB0AC"},
                    title="Relación nivel de colesterol y enfermedad cardiaca")
Cole.update_layout(legend_title = "Sufre de enfermedad cardiaca",
                  xaxis_title = "Nivel de Colesterol",
                  yaxis_title = "Conteo(Porcentaje)",
                  title_x = 0.5,
                  font = dict(size = 14))
Cole.update_layout({'plot_bgcolor': 'white'})

dash.register_page(__name__, path='/', name='Análisis Descriptivo') # '/' is home page

# page 1 data

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(options=[{"label":"Dolor de pecho", "value":"chest"},
                                              {"label":"Presion arterial", "value":"trest"},
                                              {"label":"Nivel de colesterol", "value":"chol"}],
                                     id='fig-choice')
                    ], xs=10, sm=10, md=8, lg=4, xl=4, xxl=4
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id='line-fig',
                                  figure=CP)
                    ], width=12
                )
            ]
        )
    ]
)

@callback(
    Output('line-fig', 'figure'),
    Input('fig-choice', 'value')
)
def update_graph(value):
    if value == None:
        fig = CP
    elif value == "chest":
        fig = CP
    elif value == "trest":
        fig = PA
    elif value == "chol":
        fig = Cole
    return fig
"""
layout = html.Div(
    html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Arteriopatía Coronaria. Una enfermedad cardiaca",style={'textAlign':'center'}
                ),
                html.P(
                    children= "Por medio de los siguientes graficos se puede intuir el efecto que tienen algunas variables sobre padecer o no este tipo de enfermedad cardiaca",
                    style={'textAlign':'center'}
                ),
            ],style={'backgroundColor' : "black",'color':'white'}),
        html.Div(
            html.Div([
                dcc.Graph(id="CP", figure=CP),
                dcc.Graph(id="Trest", figure=PA),
                dcc.Graph(id="Coles", figure=Cole)
            ]))]))

if __name__ == "__main__":
    app.run_server(debug=True)
"""