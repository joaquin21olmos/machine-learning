# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:05:22 2020

@author: joaquin antonio ibanez de olmos
"""

import pandas as pd
mydataset=pd.read_csv('datosEjemplo.csv')
"""x=mydataset.iloc[:,:-1].values
"""
df = pd.read_csv('datosEjemplo.csv', names=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','clase'])
from sklearn.preprocessing import StandardScaler
features = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19']# Separating out the features
x = df.loc[:, features].values
xt=x
# Separating out the target
y = df.loc[:,['clase']].values
yt=y
# Standardizing the features
x = StandardScaler().fit_transform(x)

#dos componentes
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2'])
finalDf2PC = pd.concat([principalDf, df[['clase']]], axis = 1)

#tres componentes
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2','pc3'])
finalDf3PC = pd.concat([principalDf, df[['clase']]], axis = 1)


#4 componentes
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2','pc3','pc4'])
finalDf4PC = pd.concat([principalDf, df[['clase']]], axis = 1)




import matplotlib.pyplot as plt

#grafica con 2 componentes 1 y 2


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1, 2,3,4,5,6]
colors = ['r', 'g', 'b','yellow','black','magenta','cyan']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf2PC['clase'] == target
    ax.scatter(finalDf2PC.loc[indicesToKeep, 'pc1']
               , finalDf2PC.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
"""
fig.savefig('2PCA.jpg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
"""
#grafica con 3 componentes 1 y 3

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)

targets = [0, 1, 2,3,4,5,6]
colors = ['r', 'g', 'b','yellow','black','magenta','cyan']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf3PC['clase'] == target
    ax.scatter(finalDf3PC.loc[indicesToKeep, 'pc1']
               , finalDf3PC.loc[indicesToKeep, 'pc3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
"""
fig.savefig('3PCA.jpg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
"""

#grafica con 4  componentes 1 y 4

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 4', fontsize = 15)
ax.set_title('4 component PCA', fontsize = 20)

targets = [0, 1, 2,3,4,5,6]
colors = ['r', 'g', 'b','yellow','black','magenta','cyan']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf4PC['clase'] == target
    ax.scatter(finalDf4PC.loc[indicesToKeep, 'pc1']
               , finalDf4PC.loc[indicesToKeep, 'pc4']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
"""
fig.savefig('4PCA.jpg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
"""

from sklearn.model_selection import train_test_split

from sklearn import tree
#resultado clasificador de 2 pc

X=finalDf2PC[['pc1','pc2']]
y=finalDf2PC['clase']
f_train,f_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(f_train, y_train)
prediccionV=clf.predict(f_test)
print("Valor de acertacion 2PCA ",clf.score(f_test,y_test))

from sklearn.metrics import confusion_matrix

y_pred = clf.predict(f_test)

species = y_test
predictions = y_pred
matrixConfus2PCA=confusion_matrix(species, predictions)
print(matrixConfus2PCA)


#resultado clasificador de 3 pc

X=finalDf3PC[['pc1','pc2']]
y=finalDf3PC['clase']
f_train,f_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(f_train, y_train)
prediccionV=clf.predict(f_test)
print("Valor de acertacion 3PCA ",clf.score(f_test,y_test))

y_pred = clf.predict(f_test)

species = y_test
predictions = y_pred
matrixConfus3PCA=confusion_matrix(species, predictions)
print(matrixConfus3PCA)
#resultado clasificador 4 pc

X=finalDf4PC[['pc1','pc2']]
y=finalDf4PC['clase']
f_train,f_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(f_train, y_train)
prediccionV=clf.predict(f_test)
print("Valor de acertacion 4PCA8 ",clf.score(f_test,y_test))

y_pred = clf.predict(f_test)

species = y_test
predictions = y_pred
matrixConfus4PCA=confusion_matrix(species, predictions)
print(matrixConfus4PCA)


#uso del clasificador tree con dataset original


X=xt
y=yt
f_train,f_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(f_train, y_train)
prediccionV=clf.predict(f_test)
print("Valor de acertacion test ",clf.score(f_test,y_test))

y_pred = clf.predict(f_test)
Clasif1= y_pred
species = y_test
predictions = y_pred
matrixConfusCompleta=confusion_matrix(species, predictions)
print(matrixConfusCompleta)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#knn con 2 PCA
	
Xknn=finalDf2PC[['pc1','pc2']]
yknn=finalDf2PC['clase']
 
X_train, X_test, y_train, y_test = train_test_split(Xknn, yknn,test_size=0.2, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_neighbors = 7

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('precision del set de entranamiento: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

#knn con datos originales
	
Xknn=xt
yknn=yt
 
X_train, X_test, y_train, y_test = train_test_split(Xknn, yknn,test_size=0.2, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_neighbors = 7


knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train,y_train)

print('precision del set de entranamiento: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('precision del set de prueba {:.2f}'
     .format(knn.score(X_test, y_test)))


pred = knn.predict(X_test)
Clasif2=pred
matrizConfusKNN=confusion_matrix(y_test, pred)
print(confusion_matrix(y_test, pred))
reporteKNN=classification_report(y_test, pred)
print(classification_report(y_test, pred))


#bayes con datos origninales
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


#bayes con 2 PCA
Xknn=finalDf2PC[['pc1','pc2']]
yknn=finalDf2PC['clase']
 
X_train, X_test, y_train, y_test = train_test_split(Xknn, yknn,test_size=0.2, random_state=0)
model =GaussianNB()
model.fit(X_train,y_train)

print("bayes Resultados con 2 PCA: ");
#print(accuracy)

print('precision del set de entranamiento: {:.2f}'
     .format(model.score(X_train, y_train)))
print('precision del set de prueba {:.2f}'
     .format(model.score(X_test, y_test)))



y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
cmb2=confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



#bayes con 4 PCA
X=finalDf4PC[['pc1','pc2','pc3','pc4']]
y=finalDf4PC['clase']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)
model =GaussianNB()
model.fit(X_train,y_train)

print("bayes Resultados con 4 PCA: ");
#print(accuracy)

print('precision del set de entranamiento: {:.2f}'
     .format(model.score(X_train, y_train)))
print('precision del set de prueba {:.2f}'
     .format(model.score(X_test, y_test)))



y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
cmb4=confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


#con datos originales 
Xknn=xt
yknn=yt
 
X_train, X_test, y_train, y_test = train_test_split(Xknn, yknn,test_size=0.2, random_state=0)



model =GaussianNB()
model.fit(X_train,y_train)

print("bayes Resultados: ");
#print(accuracy)

print('precision del set de entranamiento: {:.2f}'
     .format(model.score(X_train, y_train)))
print('precision del set de prueba {:.2f}'
     .format(model.score(X_test, y_test)))



RellenoOrio=X_test
Relleno=pd.DataFrame(columns=('a1','a2'))
for cla in X_test[:,[0,1]]:
    Relleno.loc[len(Relleno)]=[cla[0],cla[1]]

DatosOri=y_test
DatosOriO=pd.DataFrame(columns=('Clase','dato'))
for cla in y_test:
    DatosOriO.loc[len(DatosOriO)]=[cla,cla]
    
    
y_pred=model.predict(X_test)
Clasif3=y_pred
accuracy=accuracy_score(y_test,y_pred)
cmb=confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


#parte de la seleccion del ensamble

import statistics
InfoF=pd.DataFrame(columns=('Clasificador 1', 'Clasificador 2', 'Clasificador 3', 'Decision Ensamble'))

InfoF2=pd.DataFrame(columns=('Clasificador 1', 'Clasificador 2', 'Clasificador 3'))


veces=len(Clasif1)
    

for cla in range(veces):
           InfoF2.loc[len(InfoF2)]=[Clasif1[cla], Clasif2[cla],Clasif3[cla]]


print(InfoF2.loc[0])
#--imprime todo el objeto  print("moda ", InfoF2.loc[0].mode() )
#--solo imprime el valor print("Super moda",statistics.mode(InfoF2.loc[0]0))


for cla in range(veces):
          InfoF.loc[len(InfoF)]=[Clasif1[cla], Clasif2[cla],Clasif3[cla],statistics.mode(InfoF2.loc[cla])]
    

import numpy as np
DatosMatriz1=np.array([])

for cla in range(veces):
    DatosMatriz1=np.append(DatosMatriz1,statistics.mode(InfoF2.loc[cla])    )
    
print(InfoF.loc[:,'Decision Ensamble'])
#=InfoF.loc[:,'Decision Ensamble']
MatrizFinalEnsamble=confusion_matrix(y_test,DatosMatriz1)   
print(classification_report(y_test,DatosMatriz1))
print(MatrizFinalEnsamble) 


#grafica de la matriz de confusion

fig=plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('clase real',fontsize= 15)
ax.set_ylabel('clase predecida',fontsize= 15)
ax.set_title('Grafica de Resultados',fontsize= 20)
#crear tabla que contenga los targets ya que si no solo funcionara para los casos que indiques aqui
targets = [0,1,2,3,4,5,6]
colors = ['r', 'g', 'b','yellow','black','magenta','cyan']
colors = ['r', 'g', 'b','yellow','black','magenta','cyan']
for target, color in zip(targets,colors):
    indicesToKeep = DatosOriO['Clase'] == target
    ax.scatter(DatosOriO.loc[indicesToKeep, 'dato']
               , InfoF.loc[indicesToKeep, 'Decision Ensamble']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#grafica de las dos clases

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('a1', fontsize = 15)
ax.set_ylabel('a2', fontsize = 15)
ax.set_title('Clase Original', fontsize = 20)

targets = [0, 1, 2,3,4,5,6]
colors = ['r', 'g', 'b','yellow','black','magenta','cyan']
for target, color in zip(targets,colors):
    indicesToKeep = DatosOriO['Clase'] == target
    ax.scatter(Relleno.loc[indicesToKeep, 'a1']
               , Relleno.loc[indicesToKeep, 'a2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
fig.savefig('CalseORi.jpg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


#despues de la prediccion


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('a1', fontsize = 15)
ax.set_ylabel('a2', fontsize = 15)
ax.set_title('Clase Predecida', fontsize = 20)

targets = [0, 1, 2,3,4,5,6]
colors = ['r', 'g', 'b','yellow','black','magenta','cyan']
for target, color in zip(targets,colors):
    indicesToKeep = InfoF['Decision Ensamble'] == target
    ax.scatter(Relleno.loc[indicesToKeep, 'a1']
               , Relleno.loc[indicesToKeep, 'a2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

fig.savefig('ClasePRedic.jpg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)