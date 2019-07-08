#importing the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

#importing the dataset
data=pd.read_csv('digit_recognizer.csv')
data.head()
a=data.iloc[0,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

#Splitting the dataset into the Training set and Test set
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)
x_train.head()
y_train.head()

#Fitting Random Forest Classifier to the dataset
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train,y_train)

#Predicting the result
pred=rf.predict(x_test)

# variable s contain actual values
s=y_test.values

#counting the correct number of predictions
count=0 
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1
count
len(pred)
print(count/len(pred))
        



