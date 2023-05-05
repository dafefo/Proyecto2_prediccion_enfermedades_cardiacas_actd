# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:06:22 2023

@author: baka
"""

import numpy as np 
import pandas as pd
import plotly.express as px
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
##################CARGA Y MANIPULACION DE DATOS#####################################################
#Apertura de los datos Daniel
#data_cardiaca = pd.read_csv("Analitica computacional/Proyecto1 Enfermedades cardiacas/cleveland_data.csv")

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
edad_discrt = pd.cut(data_cardiaca["age"],bins = [0,50,100], labels = [0,1])
data_cardiaca.insert(1,"age_group", edad_discrt)

#Se crea una variable categorica para el colesterol
chol_discrt = pd.cut(data_cardiaca["chol"],bins=[0,200,240,600], labels = [1,2,3])
data_cardiaca.insert(6,"chol_group",chol_discrt)

#Se crea una varibale categorica para la presion sanguinea en reposo
trestbps_discrt = pd.cut(data_cardiaca["trestbps"],bins=[0,119,129,139,179,210], labels=[1,2,3,4,5])
data_cardiaca.insert(5,"trestbps_group", trestbps_discrt)

data_cardiaca = data_cardiaca.drop('age', axis=1)
data_cardiaca = data_cardiaca.drop('trestbps', axis=1)
data_cardiaca = data_cardiaca.drop('chol', axis=1)
data_cardiaca = data_cardiaca.drop('num', axis=1)
data_cardiaca = data_cardiaca.drop('thalac', axis=1)
data_cardiaca = data_cardiaca.drop('oldpeak', axis=1)

#data_cardiaca.to_csv("datoscat.csv", index = False)

print(data_cardiaca.head())
print(data_cardiaca.columns)



data_cardiaca_prueba= data_cardiaca.tail(30)
data_cardiaca.drop(data_cardiaca.tail(30).index,inplace=True)


#########RED BAYESIANA################################################
#Se crea el modelo Bayesiano
model = BayesianNetwork([("sex", "cardiac"), ("fbs", "cardiac"), ("age_group","cardiac"), ("chol_group","cardiac"), ("cardiac", "exang"),("cardiac","slope"),("cardiac", "thal"), ("cardiac","cp"), ("cardiac","ca"), ("cardiac","trestbps_group"), ("cardiac","restecg")])
emv = MaximumLikelihoodEstimator(model=model, data=data_cardiaca)

model.fit(data=data_cardiaca, estimator = MaximumLikelihoodEstimator) 
for i in model.nodes():
    print(model.get_cpds(i))
    
model.check_model()

infer = VariableElimination(model)
posterior_p = infer.query(["cardiac"], evidence={"sex": 1, "age_group":"Mayor", "chol_group": "normal", "fbs": 1, "exang": 1})
print(posterior_p)
#prob de verdadero
print(posterior_p.values[1])


######################Función###########################################


def calcularProbabilidad(psexo, pgrupoEdad, pgrupoColesterol, pfbs, pexang, pcp, ptrestbpsgroup, prestcg, pslope,pca, pthal):
    
    probabilidadEstimada=infer.query(["cardiac"], evidence={"sex": psexo, "age_group":pgrupoEdad, "chol_group": pgrupoColesterol, "fbs": pfbs, "exang": pexang , "cp":pcp , "trestbps_group":ptrestbpsgroup , "restecg":prestcg , "slope":pslope , "ca":pca , "thal":pthal})
    return probabilidadEstimada
    

print(calcularProbabilidad(1,"Mayor","normal",1,1,4,"elevada",2,3,2,6).values)

lista= ["Mayor",1,1,"normal","normal",1,2,0,3,0,6]

###############################
###############################
def calcularProbabilidadl(selected_values_list):
    
    probabilidadEstimada=infer.query(["cardiac"], evidence={"sex": selected_values_list[1], "age_group":selected_values_list[0], "chol_group": selected_values_list[4], "fbs": selected_values_list[5], "exang": selected_values_list[7], "cp":selected_values_list[2] , "trestbps_group":selected_values_list[3] , "restecg":selected_values_list[6], "slope":selected_values_list[8] , "ca":selected_values_list[9] , "thal":selected_values_list[10]})
    return probabilidadEstimada
    


print(calcularProbabilidadl(lista))
####################################
###################################
####NUEVO MODELO###################

bl=[]
causas =['age_group', 'chol_group', 'sex', 'fbs']
examenes =['restecg', 'thalac', 'exang', 'oldpeak', 'slope', 'thal', 'ca', 'cp','trestbps_group']
for i in causas:
    for j in data_cardiaca.columns:
        bl.append((j,i))
        
for i in examenes:
    bl.append((i,'cardiac'))
    
        
bl.remove(('age_group', 'chol_group'))
bl.remove(('age_group', 'fbs'))
bl.remove(('sex', 'chol_group'))
bl.remove(('sex', 'fbs'))

#####K2##########################

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score

scoring_method = K2Score(data=data_cardiaca)
esth = HillClimbSearch(data=data_cardiaca)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=20, max_iter=int(20000), black_list=(bl)
)
print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())

print(scoring_method.score(estimated_modelh))
print(scoring_method.score(model))

#####BIC##########################

from pgmpy.estimators import BicScore
scoring_method = BicScore(data=data_cardiaca)
esth = HillClimbSearch(data=data_cardiaca)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=20, max_iter=int(20000), black_list=(bl)
)
print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())

print(scoring_method.score(estimated_modelh))
print(scoring_method.score(model))




print(estimated_modelh.edges)




########################################
estimated_modelh=BayesianNetwork(estimated_modelh)

estimated_modelh.fit(data=data_cardiaca, estimator = MaximumLikelihoodEstimator) 
for i in estimated_modelh.nodes():
    print(estimated_modelh.get_cpds(i))
    
estimated_modelh.check_model()

##############Serializacion############
from pgmpy.readwrite import BIFWriter
writer = BIFWriter(estimated_modelh)
writer.write_bif(filename='modeloK2.bif')

from pgmpy.readwrite import XMLBIFWriter
# write model to an XML BIF file 
writer = XMLBIFWriter(estimated_modelh)
writer.write_xmlbif('modeloK2.xml')

def process_data(datosPrueba, pModelo):
    inferencia = VariableElimination(pModelo)
    VP=0
    FP=0
    FN=0
    VN=0
    
    for index, row in datosPrueba.iterrows():
        cardiac=row['cardiac']
        sex = row['sex']
        age = row['age_group']
        chol = row['chol_group']
        fbs = row['fbs']
        exang = row['exang']
        cp = row['cp']
        tbs = row["trestbps_group"]
        restecg = row["restecg"]
        pslope = row["slope"]
        thal = row["thal"]
        ca = row["ca"]
        
        pp = inferencia.query(["cardiac"], evidence={"sex": sex, "age_group":age, "chol_group": chol, "fbs": fbs, "exang": exang, "cp": cp, "trestbps_group": tbs, "restecg": restecg, "slope": pslope, "thal": thal, "ca": ca})
        pverdadero= pp.values[1]
       
        
        if (cardiac== True and pverdadero > 0.5):
            VP= VP+1
           
        if (cardiac== False and pverdadero > 0.5):
            FP= FP+1
         
        if (cardiac== True and pverdadero <= 0.5):
            FN= FN+1
            
        if (cardiac== False and pverdadero <= 0.5):
            VN= VN+1
    return VP,FP,FN,VN
            
ResutadosModeloInicial= process_data(data_cardiaca_prueba, model)
ResutadosModeloK2= process_data(data_cardiaca_prueba, estimated_modelh)
ResutadosModeloBIC= process_data(data_cardiaca_prueba, estimated_modelh)


#Probar con modelo de un companero
dataCOM = pd.read_csv("Analitica computacional/Proyecto 2 Enfermedades cardiacas/DatosSantiago.csv")
dataCOM = dataCOM.tail(30)

from pgmpy.readwrite import BIFReader
reader = BIFReader("Analitica computacional/Proyecto 2 Enfermedades cardiacas/OriginalSantiago.bif")
modeloCOM = reader.get_model()

validacion_nodos = modeloCOM.get_cpds("cp").state_names["cp"]
print(validacion_nodos)
print(dataCOM.dtypes)



def process_dataCOM(datosPrueba, pModelo):
    inferencia = VariableElimination(pModelo)
    VP=0
    FP=0
    FN=0
    VN=0
    
    for index, row in datosPrueba.iterrows():
        cardiac=row['heartdis']
        sex = row['sex']
        age = row['age']
        exang = row['exang']
        cp = row['cp']
        tbs = row["trestbps"]
        pslope = row["slope"]
        thal = row["thal"]
        ca = row["ca"]
        thalach = row["thalach"]
        oldpeak = row["oldpeak"]
        
        pp = inferencia.query(["heartdis"], evidence={"sex": sex, "age":age, "exang": exang, "cp": cp, "trestbps": tbs, "slope": pslope, "thal": thal, "ca": ca, "thalach":thalach,"oldpeak":oldpeak})
        pverdadero= pp.values[1]
       
        
        if (cardiac== True and pverdadero > 0.5):
            VP= VP+1
           
        if (cardiac== False and pverdadero > 0.5):
            FP= FP+1
         
        if (cardiac== True and pverdadero <= 0.5):
            FN= FN+1
            
        if (cardiac== False and pverdadero <= 0.5):
            VN= VN+1
    return VP,FP,FN,VN
ResutadosModeloCOM= process_dataCOM(dataCOM, modeloCOM)
print(ResutadosModeloCOM)


