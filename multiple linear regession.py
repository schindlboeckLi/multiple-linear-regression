# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:16:45 2019

@author: lisa
"""

# import relevant packages
# pandas package is used to import data
# statsmodels is used to inoke the functions that help in lm
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import dataset
data = pd.read_csv(r'C:\Users\lisa\Documents\regAgeWeight1.txt', delimiter=',')
X = data.iloc[:,1]
Y = data.iloc[:, 2]
X = [[3.54, 40],[4.29, 50],[4.59, 51],[4.79, 56],[5.24, 57],[6,62],[6.19, 63],
    [7.04, 68],[7.19, 69],[7.5, 74],[8.59, 80]]#data.iloc[:,1]
Y = [0,1,2,3,4,5,5,7,8,9,10]#data.iloc[:, 2]
print(data)

#ata = data[['Age', 'Weight', 'New']].dropna()
# run regression


#X = sm.add_constant(X)
results = smf.ols((X,Y),data=data).fit()
#est2=est.fit()
#print(results.summary())
array=results.params
#print(results.params)
print(array)


df2=pd.DataFrame(X,columns=['Weight','New'])
df2['Age']=pd.Series(Y)
print(df2)

model = smf.ols(formula='Age ~ Weight + New', data=df2)
results_formula = model.fit()
print(results_formula.params)

x_surf, y_surf = np.meshgrid(np.linspace(df2.Weight.min(), df2.Weight.max(), 110),np.linspace(df2.New.min(), df2.New.max(), 110))
onlyX = pd.DataFrame({'Weight': x_surf.ravel(), 'New': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)
#rint(fittedY)
z_surf=fittedY.values.reshape(x_surf.shape)
fig= plt.figure(figsize=(18, 16))
#plt.figsize(6,15)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['Weight'],df2['New'],df2['Age'],c='black', marker='o', alpha=1)
ax.plot_surface(x_surf,y_surf,z_surf, color='None', alpha=0.25)
ax.set_xlabel('Weight')
ax.set_ylabel('New')
ax.set_zlabel('Age')
#ax.view_init(50, -65)
plt.show()