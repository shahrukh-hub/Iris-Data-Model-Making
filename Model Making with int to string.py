# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 00:05:21 2021

@author: Shahrukh Khan
"""
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
ar1=pd.read_csv('Iris_1.csv')
le=preprocessing.LabelEncoder()
ar1.Species=le.fit_transform(ar1["Species"])
print(ar1)

x=ar1.iloc[0:,1:5]
y=ar1.iloc[:,5]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


#Logistic
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
modelLR=lr.fit(x_train,y_train)
#
prediction1 = modelLR.predict(x_test)
print("====================Prediction Of model=================")
print(prediction1)
print("====================ACtual Answers=================")
print(y_test)
from sklearn.metrics import accuracy_score
# =====================ACCUARACY===========================
print("=====================Training Accuarcy=============")
trac=lr.score(x_train,y_train)
trainingAccLR=trac*100
print(trainingAccLR)
print("====================Testing Accuracy============")
teacLr=accuracy_score(y_test,prediction1)
testingAccLR=teacLr*100
print(testingAccLR)

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
prediction2 = knn.predict(x_test)
print("====================Prediction Of model=================")
print(prediction2)
print("====================ACtual Answers=================")
print(y_test)
# =====================ACCUARACY===========================
print("=====================Training Accuarcy=============")
trac=knn.score(x_train,y_train)
trainingAcc=trac*100
print(trainingAcc)
print("====================Testing Accuracy============")
teac=accuracy_score(y_test,prediction2)
testingAcc=teac*100
print(testingAcc)

#SVM
from sklearn import svm
lr=svm.SVC()
modelLR=lr.fit(x_train,y_train)
#
prediction3 = modelLR.predict(x_test)
print("====================Prediction Of model=================")
print(prediction3)
print("====================ACtual Answers=================")
print(y_test)
from sklearn.metrics import accuracy_score
# =====================ACCUARACY===========================
print("=====================SVM-Training Accuarcy=============")
trac=lr.score(x_train,y_train)
trainingAccLR=trac*100
print(trainingAccLR)
print("====================SVM-Testing Accuracy============")
teacLr=accuracy_score(y_test,prediction3)
testingAccLR=teacLr*100
print(testingAccLR)


#Navbies
from sklearn.naive_bayes import GaussianNB
lr=GaussianNB()
modelLR=lr.fit(x_train,y_train)
#
prediction4 = modelLR.predict(x_test)
print("====================Prediction Of model=================")
print(prediction4)
print("====================ACtual Answers=================")
print(y_test)
from sklearn.metrics import accuracy_score
# =====================ACCUARACY===========================
print("=====================Nav-Training Accuarcy=============")
trac=lr.score(x_train,y_train)
trainingAccLR=trac*100
print(trainingAccLR)
print("====================Nav-Testing Accuracy============")
teacLr=accuracy_score(y_test,prediction4)
testingAccLR=teacLr*100
print(testingAccLR)

#Random-Forest
from sklearn.ensemble import RandomForestClassifier
lr=RandomForestClassifier()
modelLR=lr.fit(x_train,y_train)
#
prediction5 = modelLR.predict(x_test)
print("====================Prediction Of model=================")
print(prediction5)
print("====================ACtual Answers=================")
print(y_test)
from sklearn.metrics import accuracy_score
# =====================ACCUARACY===========================
print("=====================R-D-Training Accuarcy=============")
trac=lr.score(x_train,y_train)
trainingAccLR=trac*100
print(trainingAccLR)
print("====================R-D-Testing Accuracy============")
teacLr=accuracy_score(y_test,prediction5)
testingAccLR=teacLr*100
print(testingAccLR)

#Decision Tree
from sklearn import tree
lr=tree.DecisionTreeClassifier()
modelLR=lr.fit(x_train,y_train)
#
prediction6 = modelLR.predict(x_test)
print("====================Prediction Of model=================")
print(prediction6)
print("====================ACtual Answers=================")
print(y_test)
from sklearn.metrics import accuracy_score
# =====================ACCUARACY===========================
print("=====================D-T-Training Accuarcy=============")
trac=lr.score(x_train,y_train)
trainingAccLR=trac*100
print(trainingAccLR)
print("====================D-T-Testing Accuracy============")
teacLr=accuracy_score(y_test,prediction6)
testingAccLR=teacLr*100
print(testingAccLR)



val_logstic=0
val_KNN=0
val_SVM=0
val_Navbies=0
val_R_F=0
val_D_T=0
x=0
for check_val in prediction1:
  if check_val != y_test.iloc[x,]:
    val_logstic+=1
  x+=1

x=0
for check_val in prediction1:
  if check_val != y_test.iloc[x,]:
    val_KNN+=1
  x+=1

x=0
for check_val in prediction1:
  if check_val != y_test.iloc[x,]:
    val_SVM+=1
  x+=1

x=0
for check_val in prediction1:
  if check_val != y_test.iloc[x,]:
    val_Navbies+=1
  x+=1

x=0
for check_val in prediction1:
  if check_val != y_test.iloc[x,]:
    val_R_F+=1
  x+=1

x=0
for check_val in prediction1:
  if check_val != y_test.iloc[x,]:
    val_D_T+=1
  x+=1

Prediction={"Prediction-Logistic":prediction1,
            "Prediction-KNN":prediction2,
            "Prediction-SVM":prediction3,
            "Prediction-Navbies":prediction4,
            "Prediction-R-F":prediction5,
            "Prediction-D-T":prediction6}
df=pd.DataFrame(Prediction)
print(df)

df.to_csv(r"D\Iris-DAta.csv")

print("======================Inverse====================")
Invers_Log=le.inverse_transform(prediction1)
Invers_Knn=le.inverse_transform(prediction2)
Invers_Svm=le.inverse_transform(prediction3)
Invers_NB=le.inverse_transform(prediction4)
Invers_RF=le.inverse_transform(prediction5)
Invers_DT=le.inverse_transform(prediction6)


Inverse_Predic={"Logistic":Invers_Log,
                "KNN":Invers_Knn,
                "SVM":Invers_Svm,
                "NavBies":Invers_NB,
                "Random-Forest":Invers_RF,
                "Discussion Tree":Invers_DT}
IP=pd.DataFrame(Inverse_Predic)
print(IP)


print("Wrong-Logistic:",val_logstic,
      "Wrong-KNN:",val_KNN,
      "Wrong-SVM:",val_SVM,
      "Wrong- R-F:",val_R_F,
      "Wrong-D-T:",val_D_T)



