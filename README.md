# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values. 

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:

```PYTHON
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SRI SRINIVASAN K
RegisterNumber:  212224220104
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

TOP 5 ELEMENTS

<img width="1177" height="217" alt="image" src="https://github.com/user-attachments/assets/20e271d9-4ded-496f-a6b4-fa2d8077b944" />
<img width="1062" height="205" alt="image" src="https://github.com/user-attachments/assets/4ca90102-946d-4e77-a678-cd1c237ddb56" />

PRINT DATA

<img width="1005" height="448" alt="image" src="https://github.com/user-attachments/assets/c514ca2b-0dcf-4f37-b090-a0121f7852b3" />

CONFUSION ARRAY

<img width="715" height="70" alt="image" src="https://github.com/user-attachments/assets/37d339fd-51bc-4f95-9d28-c91994d764bc" />


ACCURACY VALUE

<img width="291" height="50" alt="image" src="https://github.com/user-attachments/assets/e9498a29-9d9d-48c0-9c9b-85c74dfc9df3" />


CLASSFICATION REPORT

<img width="666" height="210" alt="image" src="https://github.com/user-attachments/assets/5d7fd603-d4aa-4ed4-9e64-15da3be2d354" />


PREDICTION

<img width="161" height="34" alt="image" src="https://github.com/user-attachments/assets/47bc7d93-9bf0-42fc-96dc-1d48eb942253" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
