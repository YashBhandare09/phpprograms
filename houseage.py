import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv('realestate.csv')
#print(df.head(10))
print(df.describe())
#print(df.shape)

new_df=df[['houseage','unit_area']]

print(new_df)
new_df['houseage'].dropna()
new_df['unit_area'].dropna()
new_df.dropna(inplace=True)

print("________________")

x=np.array(new_df[['houseage']])
y=np.array(new_df[['unit_area']])

print(x.shape)
print(y.shape)

plt.scatter(x,y,color='Blue')
plt.title('houseage vs unit_area')
plt.xlabel('houseage')
plt.ylabel('unit_area')
plt.show()

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=15)
regressor=LinearRegression()

print("_________________")
regressor.fit(x_train, y_train)
print("_______________")

plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="red",linewidth=3)
plt.title('Regression(test set)')
plt.xlabel('houseage')
plt.ylabel('unit_area')
plt.show()

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="red",linewidth=3)
plt.title('Regression(training set)')
plt.xlabel('houseage')
plt.ylabel('unit_area')
plt.show()

 
