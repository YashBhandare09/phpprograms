import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df=pd.read_csv('suv_data.csv')
print(df.head())
df.isnull().sum()

X=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
LogisticRegression(random_state=0)


y_pred=classifier.predict(X_test)
print(y_pred)
print(y_test)

#print(accuracy_score(y_test,y_pred))
#cm=confusion_matrix(y_test,y_pred)
#print(cm)


confusion_matrix=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'])
sn.heatmap(confusion_matrix,annot=True)

print('Accuracy:',metrics.accuracy_score(y_test,y_pred))
plt.show()





