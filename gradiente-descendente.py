#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: samuelperezh
Samuel Pérez Hurtado - ID 000459067
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PUNTO 1

df=pd.read_csv("data/Dataset.csv")

# PUNTO 2

plt.plot(df["x"],df["y"],"+")
plt.xlabel("x")
plt.ylabel("y")

# PUNTO 3

x = df["x"].values
y = df["y"].values

def costo(theta_0, theta_1, x, y):
    m = len(y)
    error = (theta_0 + theta_1 * x) - y
    J = (1/(2*m)) * np.sum(np.square(error))
    return J

theta_0_vals = np.arange(-10, 10, 0.1)
theta_1_vals = np.arange(-10, 10, 0.1)
theta_0, theta_1 = np.meshgrid(theta_0_vals, theta_1_vals)

J_vals = np.zeros_like(theta_0)
for i in range(len(theta_0_vals)):
    for j in range(len(theta_1_vals)):
        J_vals[i,j] = costo(theta_0_vals[i], theta_1_vals[j], x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta_0, theta_1, J_vals)
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Costo')
plt.show()

# PUNTO 4

m = len(x)
th_v = np.array([0, -10])
alfa = 0.0001

for i in range(5000):
    th_v = th_v-alfa*np.array([1/m*np.sum(((th_v[0]+th_v[1]*df["x"])-df["y"])), 1/(m)*np.sum(((th_v[0]+th_v[1]*df["x"])-df["y"])*df["x"])])

print(f"El valor de theta_0 es de: {th_v[0]} y el valor de theta_1 es de: {th_v[1]}")

# PUNTO 5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta_0, theta_1, J_vals)
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Costo')
plt.show()

th_v = np.array([0,-10])
alfa = 0.0001

for i in range(50):
    th_v = th_v-alfa*np.array([1/m*np.sum(((th_v[0]+th_v[1]*df["x"])-df["y"])), 1/(m)*np.sum(((th_v[0]+th_v[1]*df["x"])-df["y"])*df["x"])])
    plt.plot(th_v[0],th_v[1],".", color='red')
    plt.pause(0.001)
    
# PUNTO 6

fig = plt.figure()
plt.plot(df["x"],df["y"],"+")
plt.xlabel("x")
plt.ylabel("y")

th_v = np.array([0,-10])
alfa=0.0001

for i in range(500):
    th_v = th_v-alfa*np.array([1/m*np.sum(((th_v[0]+th_v[1]*df["x"])-df["y"])), 1/(m)*np.sum(((th_v[0]+th_v[1]*df["x"])-df["y"])*df["x"])])
    x = np.arange(0,100,1)
    h = th_v[0]+th_v[1]*x
    plt.plot(x,h,color='red')
    plt.pause(0.2)