# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:48:32 2020

@author: Joaquin
"""

#importaciones de las libreiras usadas
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import statistics
import matplotlib.cm as cm
from collections import Counter

#variables con texto o variables globales que se usaran

menu="""hola porfavor ingresa el numero de lo que deseas hacer
0- ver conjunto
1- ralizar PCA
2- entrenar y probar Clasificador con PCA
3- Entrenar Ensamble y probar ensamble
4 o cualquier otro numero- salir
"""

llave=True
global y
global x
global finalPCA
global Ncompo
global clases
global Clasi1
global Clasi2
global Clasi3
global prom1
global prom2
global prom3
global InfoF2

#inicio del programa 
# carga del archivo txt
#porfavor cambiar la ruta del archivo txt a usar
"""
si tu txt se encuentra fuera de la carpeta de este archivo ejecutable
por favor coloque la ruta completa del txt
"""


Conjunto=pd.read_csv('cre300.txt', sep=",", header=None,skiprows=3)
y=Conjunto.iloc[:,-1]
x=Conjunto.iloc[:,0:-1]
clases=y.groupby(y).mean()

#fin del la carga del conjunto de datos

#funciones usadas

def bayesPCA(Nflod,tam):
    xBayes=finalPCA.iloc[:,0:-1]
    yBayes=finalPCA.iloc[:,-1] 
    X_train, X_test, y_train, y_test = train_test_split(xBayes, yBayes, test_size=tam)
    model =GaussianNB()
    model.fit(X_train,y_train)
    scores = cross_val_score(model, xBayes, yBayes, cv=Nflod)
    print('precision con validacion cruzada (parcial)',scores)
    print('precision con validacion cruzada (promedio)',np.average(scores))
    y_pred=model.predict(X_test)
    print('matriz de confusion del clasificador con el set de prueba')
    print(confusion_matrix(y_test,y_pred))
    print(pd.crosstab(y_test,y_pred,rownames=['Clase real'],colnames=['Clase predecida']
                 ,margins=True))
    print("precision con el set de prueba")
    print(model.score(X_test,y_test))



def graficarPCA():
    global Ncompo
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Componenete Principal A', fontsize = 15)
    ax.set_ylabel('Componente Principal B', fontsize = 15)
    ax.set_title('Grafica PCA', fontsize = 20)
    targets = clases
    colors = ['r', 'g', 'b','yellow','black','magenta','cyan']
    for target, color in zip(targets,colors):
        indicesToKeep = finalPCA.iloc[:,-1] == target
        print(indicesToKeep)
        ax.scatter(finalPCA.loc[indicesToKeep,0]
                   , finalPCA.loc[indicesToKeep,Ncompo-1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    graficaFoto=(input("ingresa nombre de tu archivo con extencion .jpg: "))
    fig.savefig(graficaFoto, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    print("tu grafica se ha generado")
    return


def pcauser(Nconjuntos):
    pca = PCA(n_components=Nconjuntos)
    xpca = StandardScaler().fit_transform(x)
    principalComponents = pca.fit_transform(xpca)
    principalDf = pd.DataFrame(data = principalComponents)
    global finalPCA
    finalPCA = pd.concat([principalDf,y], axis = 1)
    print(finalPCA)

def bayesOrig(Nflod,tam):
    print("Clasificador 1 ----------------------------")
    xBayes=x
    yBayes=y 
    X_train, X_test, y_train, y_test = train_test_split(xBayes, yBayes, test_size=tam)
    model =GaussianNB()
    model.fit(X_train,y_train)
    scores = cross_val_score(model, xBayes, yBayes, cv=Nflod)
    print('precision con validacion cruzada (parcial)',scores)
    global prom1
    prom1=np.average(scores)
    print('precision con validacion cruzada (promedio)',np.average(scores))
    y_pred=model.predict(X_test)
    global Clasi1
    Clasi1=y_pred
    print('matriz de confusion del clasificador con el set de prueba')
    print(confusion_matrix(y_test,y_pred))
    print("precision con el set de prueba")
    print(model.score(X_test,y_test))

def knnOrig(Nnb,Nflod,tam):
    print("Clasificador 2 ----------------------------")
    Xknn=x
    yknn=y
    X_train, X_test, y_train, y_test = train_test_split(Xknn, yknn,test_size=tam, random_state=0)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    n_neighbors = 7
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train,y_train)
    scores = cross_val_score(knn, Xknn, yknn, cv=Nflod)
    print('precision con validacion cruzada (parcial)',scores)
    global prom2
    prom2=np.average(scores)
    print('precision con validacion cruzada (promedio)',np.average(scores))
    pred = knn.predict(X_test)
    global Clasi2
    Clasi2=pred
    print(confusion_matrix(y_test, pred))
    print('precision del set de prueba {:.2f}'.format(knn.score(X_test, y_test)))
    #reporteKNN=classification_report(y_test, pred)
    #print(classification_report(y_test, pred))

    
    
def treeOrig(Nflod,tam):    
    print("Clasificador 3 ----------------------------")
    xt=x
    yt=y
    X_train,X_test, y_train,y_test= train_test_split(xt,yt,test_size=tam,random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    scores = cross_val_score(clf, xt, yt, cv=Nflod)
    print('precision con validacion cruzada (parcial)',scores)
    global prom3
    prom3=np.average(scores)
    print('precision con validacion cruzada (promedio)',np.average(scores))
    y_pred = clf.predict(X_test)
    global Clasi3
    Clasi3= y_pred
    matrixConfusCompleta=confusion_matrix(y_test,y_pred)
    print(matrixConfusCompleta)
    print(pd.crosstab(y_test,y_pred,rownames=['Clase real'],colnames=['Clase predecida']
                 ,margins=True))
    print('precision del set de prueba {:.2f}'.format(clf.score(X_test, y_test)))

def switch(eleccion):   
    if eleccion == 0:
        print(Conjunto)
    
    elif eleccion == 1:
        global Ncompo
        Ncompo=int (input("Elige N componentes: "))
        print("Realizar PCA con Componentes= ",Ncompo)
        pcauser(Ncompo)
        grafica=int (input("Deseas graficar si-1, no-0: "))
        if grafica == 1:
            graficarPCA()
            
    elif eleccion == 2:
        print("CLasificador con PCA Bayes con el conjunto PCA que se creo")
        flods=int (input("Ingresa el numero de K-flods para el uso de la validacion cruzada: "))
        tamPrueba=float (input("Ingresa el tamaño del set de prueba rango(0.9-0.1): "))
        bayesPCA(flods,tamPrueba)
    
    elif eleccion == 3:
        global Clasi1
        global Clasi2
        global Clasi3 
        print("entrenamiento y prueba del ensamble")
        kflods=int(input("ingresa el numero de K-flod: "))
        vecinos=int(input("ingresa el numero de vecinos: "))
        tamPrueba=float (input("Ingresa el tamaño del set de prueba rango(0.9-0.1): "))
        print("RESULTADOS")
        bayesOrig(kflods,tamPrueba)
        knnOrig(vecinos,kflods,tamPrueba)
        treeOrig(kflods,tamPrueba)
        print("Rsultados Ensamble ------------------------")
        xt=x
        yt=y
        X_train,X_test, y_train,y_test= train_test_split(xt,yt,test_size=0.2,random_state=0)
        DatosOri=y_test
        DatosOriO=pd.DataFrame(columns=('Clase','dato'))
        for cla in y_test:
            DatosOriO.loc[len(DatosOriO)]=[cla,cla]
        print("precision del ensamble: ",np.average([prom1,prom2,prom3]))
        InfoF=pd.DataFrame(columns=('Clasificador 1', 'Clasificador 2', 'Clasificador 3', 'Decision Ensamble'))
        global InfoF2
        InfoF2=pd.DataFrame(columns=('Clasificador 1', 'Clasificador 2', 'Clasificador 3'))                
        veces=len(Clasi1)  
        for cla in range(veces):
            InfoF2.loc[len(InfoF2)]=[Clasi1[cla], Clasi2[cla],Clasi3[cla]]
            #print(InfoF2.loc[cla])
        DatosMatriz1=np.array([])
        fir=max([prom1,prom2,prom3])
        for cla in range(veces):
            if InfoF2.iloc[cla]['Clasificador 1']!=InfoF2.iloc[cla]['Clasificador 2']!=InfoF2.iloc[cla]['Clasificador 3']:
                if prom1==fir:
                    InfoF.loc[len(InfoF)]=[Clasi1[cla], Clasi2[cla],Clasi3[cla],InfoF2.iloc[cla]['Clasificador 1']]
                    DatosMatriz1=np.append(DatosMatriz1,InfoF2.iloc[cla]['Clasificador 1']    )
                elif prom2==fir:
                    InfoF.loc[len(InfoF)]=[Clasi1[cla], Clasi2[cla],Clasi3[cla],InfoF2.iloc[cla]['Clasificador 2']]
                    DatosMatriz1=np.append(DatosMatriz1,InfoF2.iloc[cla]['Clasificador 2']    )
                elif prom3==fir:
                    InfoF.loc[len(InfoF)]=[Clasi1[cla], Clasi2[cla],Clasi3[cla],InfoF2.iloc[cla]['Clasificador 3']]
                    DatosMatriz1=np.append(DatosMatriz1,InfoF2.iloc[cla]['Clasificador 3']    )
            else:
                InfoF.loc[len(InfoF)]=[Clasi1[cla], Clasi2[cla],Clasi3[cla],statistics.mode(InfoF2.iloc[cla]) ]
                DatosMatriz1=np.append(DatosMatriz1,statistics.mode(InfoF2.iloc[cla])    )
        MatrizFinalEnsamble=confusion_matrix(y_test,DatosMatriz1)   
        print(MatrizFinalEnsamble) 
        print(pd.crosstab(y_test,DatosMatriz1,rownames=['Clase real'],colnames=['Clase predecida']
                 ,margins=True))
        fig=plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Clase Real',fontsize= 15)
        ax.set_ylabel('Clase Predecida',fontsize= 15)
        ax.set_title('Grafica de Resultados del Ensamble',fontsize= 20)
        #crear tabla que contenga los targets ya que si no solo funcionara para los casos que indiques aqui
        targets = clases
        colors = ['b', 'g', 'r','c','m','y','k','w']
        for target,c in zip(targets,colors):
            indicesToKeep = DatosOriO['Clase'] == target
            ax.scatter(DatosOriO.loc[indicesToKeep, 'dato'] 
                       ,InfoF.loc[indicesToKeep, 'Decision Ensamble'],
                       color=c
                       , s = 50)
        counterF=Counter(InfoF['Decision Ensamble'])
        counterI=Counter(DatosOriO['dato'])
        print("original",counterI)
        print("ensamble",counterF)
        for i, txt in enumerate(clases):
            for j,jo in enumerate (clases):
                if MatrizFinalEnsamble[i][j]!=0:
                    ax.annotate("  casos:  "+str(MatrizFinalEnsamble[i][j]), (clases[i], clases[j]))
        
        ax.legend(targets)
        ax.grid()
        graficaFoto=(input("ingresa nombre de tu archivo con extencion .jpg: "))
        fig.savefig(graficaFoto, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        print("tu grafica se ha generado")
           # print(statistics.mode(InfoF2.loc[cla]))
            #InfoF.loc[len(InfoF)]=[Clasi1[plo], Clasi2[plo],Clasi3[plo],statistics.mode(InfoF2.loc[plo])]
 
        

    else:
        print("Hasta luego")
        global llave
        llave=not llave
        
    


#parte final del programa en donde se produce el desarrolo total

while llave:
    print(menu);
    eleccion=int (input("Elige: "))
    switch(eleccion)