from operator import mod
from statistics import mode
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data=pd.read_csv('C:\py\ml heartD\hrtdata.csv')
#heart_data.shape
'''
print(heart_data.isnull().sum())
print(heart_data.tail())
print(heart_data.head())
print(heart_data.describe())

print(heart_data['target'].value_counts()) #1 oe 0, how many

'''


x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']


#split 
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,stratify=y, random_state=2)
#print(x.shape,x_train.shape , x_test.shape)

#traning

#LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)


#acc
x_train_prd=model.predict(x_train)
traning_data_acc=accuracy_score(x_train_prd,y_train)
#print('acc',traning_data_acc)

x_test_prd=model.predict(x_test)
test_data_acc=accuracy_score(x_test_prd,y_test)
#print('acc',test_data_acc)

input_data= (56,1,1,120,236,0,1,178,0,0.8,2,0,2)

input_data_numarry = np.asarray(input_data)
input_data_reshaped = input_data_numarry.reshape(1,-1)#?????
prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print('there is no heart disease')
else:
    print('person has heart disease')
   