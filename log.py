import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearnmetrics
from sklearn.matrics import accuracy_score
from sklearn.metrics import confusion_matrix
  df = pd.read_csv("")
 print(df.head())
print(df.isnull().sum())
 x = df.iloc[:,[2,3]].values
y = df.i;oc[:,4].values
x_ttarin,x_test,y_tarin,y_test = train_test_split(x,y,test_size = o.25)
sc = StandardScaler()
x_tarin = sc.fit_transform(x_train)
x_test =  sc.fit_transform(x_test)
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)
LogisticRegression(random_state = 0)

y_pred = classifier.predict(x_test)
print(y_pred)
print(y_test)

confusion_matrix = pd.crosstab(y_test,y_pred,rownames = ['actual'],colnames = ['predicted']
sns.heatmap(confusion_matrix,annot = True)

print("Acc : ",metrics.accuracy_score(y_test,y_pred))
plt.show()

