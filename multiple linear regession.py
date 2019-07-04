# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:09:43 2019

@author: lisa
"""

# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
#plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = np.loadtxt('AgeWeightHeight.txt', delimiter=',')
#print(data)
X_1 = data[:, 0]
X_2 = data[:, 1]
X = data[:, 0:2]
print(X)
X = sm.add_constant(X)

Y = data[:, 2]
m = np.size(Y, 0)
#plt.scatter(X[:,0],y[:,0])
print(X, 'x')
#plt.show()
#print(X, y)
Q = X.T.dot(X)
Xy = X.T.dot(Y)
Qinv = np.linalg.inv(Q)
beta = Qinv.dot(Xy) 
r_squared = 1 - (Y - X.dot(beta)).T.dot(Y - X.dot(beta))/ ((Y-np.mean(Y)).T.dot(Y-np.mean(Y)))
print(beta, 'beta')
#print()
#print(Qinv)
#print()
#print(Xy)
#print()
#print(Q)
   
print(r_squared, 'R2')
fig= plt.figure(figsize=(20, 18))
#plt.figsize(6,15)
ax = fig.add_subplot(111, projection='3d')
a = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
predictedY = np.dot(X, a)
print(predictedY)

x_surf, y_surf = np.meshgrid(np.linspace(X_1.min(), X_1.max(), 110),np.linspace(0, X_2.max(), 110))
Y_pred = beta[0] + beta[1]*x_surf + beta[2]*y_surf

ax.scatter(X_1,X_2,Y,c='black', marker='o', alpha=1)
ax.scatter(X_1,X_2, predictedY, color='g',alpha=0.0)
ax.plot_surface(x_surf,y_surf,Y_pred,color='None',  alpha=0.1)
ax.set_xlabel('Weight', fontsize= 25)
ax.set_ylabel('Height', fontsize= 25)
ax.set_zlabel('Age', fontsize= 25)
ax.grid(linewidth=.1)
#ax.view_init(45,3)
i=0
print('pred=', beta[0] + beta[1]*12 + beta[2]*85, )
for xx,yy,zz in zip(X_1,X_2,predictedY):
    res = Y[i]-(zz)
    line=art3d.Line3D(*zip((xx, yy, zz), (xx, yy, Y[i])))
    ax.add_line(line)
    i+=1

#plt.show()
plt.tight_layout()

plt.savefig(r'C:\urLocation',bbox_inches='tight',pad_inches=0)
