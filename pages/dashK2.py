# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
import plotly.express as px
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader
from pgmpy.readwrite import XMLBIFReader
##################CARGA Y MANIPULACION DE DATOS#####################################################
#Apertura de los datos Daniel
#data_cardiaca = pd.read_csv("Analitica computacional/Proyecto1 Enfermedades cardiacas/cleveland_data.csv")

# NO PUDE LEERLO USANDO BIF, PROBABLEMENTE PORQUE LOS NOMBRES DE LAS VARIABLES NO SON COMPATIBLES 
#reader = BIFReader("C:/Users/baka/Desktop/analitica proyecto 2 local/modeloBIC.bif")
#modelo = reader.get_model()

reader = XMLBIFReader("modeloK2.xml")
model = reader.get_model()

#Apertura datos Christer
#data_cardiaca = pd.read_csv("C:/Users/baka/Desktop/analitica/Proyectos/Proyecto_prediccion_enfermedades_cardiacas__actd/cleveland_data.csv")
data_cardiaca = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None)
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

#Se crea una avraible categorica de la edad. 
edad_discrt = pd.cut(data_cardiaca["age"],bins = [0,50,100], labels = ["Joven","Mayor"])
data_cardiaca.insert(1,"age_group", edad_discrt)

#Se crea una variable categorica para el colesterol
chol_discrt = pd.cut(data_cardiaca["chol"],bins=[0,200,240,600], labels = ["normal","alto","muy alto"])
data_cardiaca.insert(6,"chol_group",chol_discrt)

#Se crea una varibale categorica para la presion sanguinea en reposo
trestbps_discrt = pd.cut(data_cardiaca["trestbps"],bins=[0,119,129,139,179,210], labels=["normal","elevada","presion arterial nivel 1","presion arterial nivel 2","crisis"])
data_cardiaca.insert(5,"trestbps_group", trestbps_discrt)

#Se quitan algunas de las variables que ya no vamos a usar para que no generen nodos adicionales en el modelo
data_cardiaca = data_cardiaca.drop('age', axis=1)
data_cardiaca = data_cardiaca.drop('trestbps', axis=1)
data_cardiaca = data_cardiaca.drop('chol', axis=1)
data_cardiaca = data_cardiaca.drop('num', axis=1)
data_cardiaca = data_cardiaca.drop('thalac', axis=1)
data_cardiaca = data_cardiaca.drop('oldpeak', axis=1)

print(data_cardiaca.head())
print(data_cardiaca.columns)

#########RED BAYESIANA################################################
#YA NO NECESITAMOS ENTRENAR EL MODELO NI ES ESTE EL MODELO
#Se crea el modelo Bayesiano
#model = BayesianNetwork([("sex", "cardiac"), ("fbs", "cardiac"), ("age_group","cardiac"), ("chol_group","cardiac"), ("cardiac", "exang"),("cardiac","slope"),("cardiac", "thal"), ("cardiac","cp"), ("cardiac","ca"), ("cardiac","trestbps_group"), ("cardiac","restecg")])
#emv = MaximumLikelihoodEstimator(model=model, data=data_cardiaca)

#model.fit(data=data_cardiaca, estimator = MaximumLikelihoodEstimator) 
#for i in model.nodes():
 #   print(model.get_cpds(i))
    
model.check_model()

infer = VariableElimination(model)
posterior_p = infer.query(["cardiac"], evidence={"sex": "0_0"})

valid_values_sex = model.get_cpds("sex").state_names["sex"]
valid_values_cardiac = model.get_cpds("cardiac").state_names["cardiac"]
valid_values_age = model.get_cpds("age_group").state_names["age_group"]
valid_values_cp = model.get_cpds("cp").state_names["cp"]
valid_values_ca = model.get_cpds("ca").state_names["ca"]
valid_values_chol = model.get_cpds("chol_group").state_names["chol_group"]
valid_values_exang = model.get_cpds("exang").state_names["exang"]
valid_values_fbs = model.get_cpds("fbs").state_names["fbs"]
valid_values_restecg = model.get_cpds("restecg").state_names["restecg"]
valid_values_slope = model.get_cpds("slope").state_names["slope"]
valid_values_thal = model.get_cpds("thal").state_names["thal"]
valid_values_trestbps = model.get_cpds("trestbps_group").state_names["trestbps_group"]

print(posterior_p)
print(model.nodes)
######################Función###########################################


#def calcularProbabilidad(psexo, pgrupoEdad, pgrupoColesterol, pfbs, pexang, pcp, ptrestbpsgroup, prestcg, pslope,pca, pthal):
    
 #   probabilidadEstimada=infer.query(["cardiac"], evidence={"sex": psexo, "age_group":pgrupoEdad, "chol_group": pgrupoColesterol, "fbs": pfbs, "exang": pexang , "cp":pcp , "trestbps_group":ptrestbpsgroup , "restecg":prestcg , "slope":pslope , "ca":pca , "thal":pthal})
  #  return probabilidadEstimada
    

#print(calcularProbabilidad(1,"Mayor","normal",1,1,4,"elevada",2,3,2,6))
###############################
def calcularProbabilidad(selected_values_list):
    probabilidadEstimada=infer.query(["cardiac"], evidence={"sex": selected_values_list[1], "age_group":selected_values_list[0], "chol_group": selected_values_list[4], "fbs": selected_values_list[5], "exang": selected_values_list[7], "cp":selected_values_list[2] , "trestbps_group":selected_values_list[3] , "restecg":selected_values_list[6], "slope":selected_values_list[8] , "ca":selected_values_list[9] , "thal":selected_values_list[10]})
    return probabilidadEstimada

import numpy as np 
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import dash_daq as daq
from dash import dcc, html, callback, Output, Input, State


# Load the Cleveland Heart Disease dataset
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None)
df= data_cardiaca
print(df.columns)
# Rename the columns with descriptive names
df.columns = [
     'edad', 'sexo', 'cp', 'trestbps', 'colesterol', 'fbs', 'restecg', 
    'exang', 'pendiente', 'ca', 'thal', 'cardiac'
]

# Define the variables to include in the dropdown menus
dropdown_vars = [col for col in df.columns if col not in ['index','age','rtrestbps','rchol','thalach', 'oldpeak','num','cardiac', 'target']]



# Define the app layout
dash.register_page(__name__, name='Predicción')
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
layout = html.Div([
    html.Div([
        html.Label(f'Seleccione un valor para {var}'),
        dcc.Dropdown(
            id=f'{var}-dropdown',
            options=[
                {'label': 'Mayor de 50', 'value': 'Mayor'},
                {'label': 'Menor de 50', 'value': 'Joven'}
            ] if var == 'edad' else (
                [
                    {'label': 'normal', 'value': 'normal'},
                    {'label': 'elevada', 'value': 'elevada'},
                    {'label': 'presion arterial nivel 1', 'value': 'presion_arterial_nivel_1'},
                    {'label': 'presion arterial nivel 2', 'value': 'presion_arterial_nivel_2'},
                    {'label': 'crisis', 'value': 'crisis'}
                ] if var == 'trestbps' else (
                    [
                        {'label': 'normal', 'value': 'normal'},
                        {'label': 'alto', 'value': 'alto'},
                        {'label': 'muy alto', 'value': 'muy_alto'}
                    ] if var == 'colesterol' else
                        [
                            {'label': 'Ningún vaso coloreado', 'value': '0_0'},
                            {'label': '3 coloreados', 'value': '3_0'},
                            {'label': '2 coloreados', 'value': '2_0'},
                            {'label': '1 coloreado', 'value': '1_0'}
                        ] if var == 'ca' else
                            [
                                {'label': 'Error fijo', 'value': '6_0'},
                                {'label': 'Normal', 'value': '3_0'},
                                {'label': 'Error reversible', 'value': '7_0'}
                            ] if var == 'thal' else
                          
                                [
                                    {'label': 'Mujer', 'value': '0_0'},
                                    {'label': 'Hombre', 'value': '1_0'},
                                    
                                ] if var == 'sexo' else
                                    [
                                        {'label': 'Angina típica', 'value': '1_0'},
                                        {'label': 'Angina atípica', 'value': '2_0'},
                                        {'label': 'Dolor no angina', 'value': '3_0'},
                                        {'label': 'Asintomático', 'value': '4_0'}
                                    ] if var == 'cp' else
                                        [
                                            {'label': 'Ausencia angina ejercicio', 'value': '0_0'},
                                            {'label': 'Presencia angina ejercicio', 'value': '1_0'},
                                            
                                        ] if var == 'exang' else
                                            [
                                                {'label': 'Menor o igual a 120', 'value': '0_0'},
                                                {'label': 'Mayor a 120', 'value': '1_0'},
                                                
                                            ] if var == 'fbs' else
                                                [
                                                    {'label': 'Normal', 'value': '0_0'},
                                                    {'label': 'Anormalidad en la curva ST', 'value': '1_0'},
                                                    {'label': 'Hipertrofia ventricular probable o definitiva', 'value': '2_0'}
                                                    
                                                ] if var == 'restecg' else
                                                    [
                                                        {'label': 'Ascendente', 'value': '1_0'},
                                                        {'label': 'Plana', 'value': '2_0'},
                                                        {'label': 'Descendente', 'value': '3_0'}
                                                        
                                                    ] if var == 'pendiente' else
                            [{'label': val, 'value': val} for val in df[var].unique()]
                            
                )
            ),
            value=df[var].unique()[0],  # Default value
            
        )
    ],style={"color":"black","background-color":"white",'width': '43%', "margin-left":0}) for var in dropdown_vars
    
] + [
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div([daq.Gauge(id="my_gauge", value = 0, max = 1, min = 0,
                        color = {"gradient":True,"ranges":{"green":[0,0.3], "yellow":[0.3,0.6],"red":[0.6,1]}},size = 400)],style={'width': '50%', "float":"right"})
    
])

# Define the app callback
@callback(
    Output('my_gauge', 'value'),
    [Input('submit-button', 'n_clicks')],
    [State(f'{var}-dropdown', 'value') for var in dropdown_vars]
)
def update_output(n_clicks, *selected_values):
    #lista= ["Mayor",1,1,"normal","normal",1,2,0,3,0,6]
    if n_clicks > 0:
        selected_values_list = [val if val != "?" else None for val in selected_values]
        print(selected_values_list)
        print(calcularProbabilidad(selected_values_list))
        probs=calcularProbabilidad(selected_values_list)
        value = probs.values[1]
        return value
