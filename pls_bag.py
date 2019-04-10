# pls_bag.py

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
import os
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from sympy import Symbol, diff, exp, pprint


# function to model and create data
def func(X, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    t = X[0]
    rh = X[1]
    f = 100 * a2 * (a8 + 1 - a3 * np.exp(- a2 * (a0 + a1 * rh + a5 * rh ** 2) * t)) ** 4 - t ** a7 * a6 * rh - a4 / np.exp(a9) + a8 * rh
    return f


# function parameters
def printParameters(popt):
    print('Parameters Values:', '\n', popt, '\n')


# curve_fit
def optimalPrameters(func, X, drainage):
    # popt returns the best fit values for parameters of the given model (func)
    t, rh = X
    p0 = 1., 1.19209e-07, 1.19209e-07, 1.19209e-07, 1.19209e-07, 1.19209e-07, 1.19209e-07, 1.19209e-07, 1.19209e-07, 1.19209e-07
    popt, _ = curve_fit(func, (X[0], X[1]), drainage, p0, method='trf')
    return popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9]


# using root mean squared error as non Linear regression metric
def rmse(drainage, fit):
    rmse = sqrt(mean_squared_error(drainage, fit))
    print('Root Mean Squared Error:', rmse, '\n')


# converts seconds to days
def secondsToDays(t):
    t = t / (3600.0 * 24.0)
    return t


# plot 3D data
def plot3D(t, rh, drainage, fit, label, name):
    plt.close()
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('time [days]', fontsize=14)
    ax.set_ylabel('relative humidity', fontsize=14)
    ax.set_zlabel('drainage [%]', fontsize=14)
    ax.set_zlim((0, 100))
    ax.plot(t, rh, drainage, 'ro', linewidth=1, label=label, c='k')
    ax.plot(t, rh, fit, 'rx', linewidth=1, label='Best fit', c='b')
    ax.legend(loc=2, shadow=False, fontsize=10)
    plt.savefig('figures/PLS bag/PLS bag plot')
    plt.clf()


# plot gradient
def plotGradient(t, y, t35, t60, t90, y35, y60, y90):
    plt.close()
    plt.figure(figsize=plt.figaspect(0.5))
    xmax = max(t)
    ymin = -10
    ymax = 100
    plt.axis([0, xmax * 1.01, ymin, ymax * 1.01])
    plt.xticks(np.arange(0, xmax * 1.001, 1))
    plt.yticks(np.arange(0, ymax * 1.001, 10))
    plt.xlabel('time [days]', fontsize=14)
    plt.ylabel('drainage rate', fontsize=14)
    plt.plot(t35, y35, '-', linewidth=1, label='35%', c='b')
    plt.plot(t60, y60, '-', linewidth=1, label='60%', c='r')
    plt.plot(t90, y90, '-', linewidth=1, label='90%', c='g')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tick_params(top='off', right='off')
    plt.tick_params(direction='out')
    plt.legend(loc=1, shadow=False, fontsize=10)
    plt.savefig('figures/PLS bag/PLS bag gradient')
    plt.clf()


# counts and prints negative values
def negativeValues(y):
    print('Negative Gradient Values:', np.count_nonzero(y < 0), '\n')
    print(y[y < 0], '\n')
    print('Min:', min(y), '\n')


# read XLSX
file_location = './data/PLS bag.xlsx'
dataset = pd.read_excel(file_location, sheet_name='drainage', header=0)
workbook = xlrd.open_workbook(file_location)
worksheet = workbook.sheet_by_name('drainage')
name = worksheet.cell(0, 1).value

# data preprocessing
t = secondsToDays(dataset.iloc[:, 0])
t[t <= 0] = 1e-8
drainage = dataset.iloc[:, 1]
rh = dataset.iloc[:, 2]
label = worksheet.cell(0, 1).value

# ignore NaNs
mask = ~np.isnan(t) & ~np.isnan(drainage) & ~np.isnan(rh)
t = t[mask]
drainage = drainage[mask]
rh = rh[mask]

print(label, '\n')
optimalPrameters(func, (t, rh), drainage)
printParameters(optimalPrameters(func, (t, rh), drainage))
fit = func((t, rh), *optimalPrameters(func, (t, rh), drainage))
rmse(drainage, fit)
fig_directory = './figures/PLS bag'
if not os.path.exists(fig_directory):
    os.makedirs(fig_directory)
t35 = secondsToDays(dataset.iloc[:189, 0])
t60 = secondsToDays(dataset.iloc[189:378, 0])
t90 = secondsToDays(dataset.iloc[378:, 0])
t35[t35 <= 0] = 1e-4
t60[t60 <= 0] = 1e-4
t90[t90 <= 0] = 1e-4
rh35 = dataset.iloc[:189, 2]
rh60 = dataset.iloc[189:378, 2]
rh90 = dataset.iloc[378:, 2]
drainage35 = dataset.iloc[:189, 1]
drainage60 = dataset.iloc[189:378, 1]
drainage90 = dataset.iloc[378:, 1]
plot3D(t, rh, drainage, fit, label, name)

a0 = optimalPrameters(func, (t, rh), drainage)[0]
a1 = optimalPrameters(func, (t, rh), drainage)[1]
a2 = optimalPrameters(func, (t, rh), drainage)[2]
a3 = optimalPrameters(func, (t, rh), drainage)[3]
a4 = optimalPrameters(func, (t, rh), drainage)[4]
a5 = optimalPrameters(func, (t, rh), drainage)[5]
a6 = optimalPrameters(func, (t, rh), drainage)[6]
a7 = optimalPrameters(func, (t, rh), drainage)[7]
a8 = optimalPrameters(func, (t, rh), drainage)[8]
a9 = optimalPrameters(func, (t, rh), drainage)[9]

f = 100 * a2 * (a8 + 1 - a3 * np.exp(- a2 * (a0 + a1 * rh + a5 * rh ** 2) * t)) ** 4 - t ** a7 * a6 * rh - a4 / np.exp(a9) + a8 * rh
f35 = 100 * a2 * (a8 + 1 - a3 * np.exp(- a2 * (a0 + a1 * rh35 + a5 * rh35 ** 2) * t35)) ** 4 - t35 ** a7 * a6 * rh35 - a4 / np.exp(a9) + a8 * rh35
f60 = 100 * a2 * (a8 + 1 - a3 * np.exp(- a2 * (a0 + a1 * rh60 + a5 * rh60 ** 2) * t60)) ** 4 - t60 ** a7 * a6 * rh60 - a4 / np.exp(a9) + a8 * rh60
f90 = 100 * a2 * (a8 + 1 - a3 * np.exp(- a2 * (a0 + a1 * rh90 + a5 * rh90 ** 2) * t90)) ** 4 - t90 ** a7 * a6 * rh90 - a4 / np.exp(a9) + a8 * rh90

y = np.gradient(f, t)
y35 = np.gradient(f35, t35)
y60 = np.gradient(f60, t60)
y90 = np.gradient(f90, t90)
i = 0
plotGradient(t, y, t35, t60, t90, y35, y60, y90)
negativeValues(y)

# pretty printing
t = Symbol('t')
rh = Symbol('rh')
a0 = Symbol('a0')
a1 = Symbol('a1')
a2 = Symbol('a2')
a3 = Symbol('a3')
a4 = Symbol('a4')
a5 = Symbol('a5')
a6 = Symbol('a6')
a7 = Symbol('a7')
a8 = Symbol('a8')
a9 = Symbol('a9')
F = Symbol('DR(t,rh) =')
dF = Symbol('dDR(t,rh)')
dt = Symbol('dt')
Eq = Symbol('=')
f = F + 100 * a2 * (a8 + 1 - a3 * exp(- a2 * (a0 + a1 * rh + a5 * rh ** 2) * t)) ** 4 - t ** a7 * a6 * rh - a4 / exp(a9) + a8 * rh
f1 = dF/dt

print('Function', '\n')
pprint(f)
print(3 * '\n', 'Gradient', '\n')
pprint(f1)
pprint(Eq + f.diff(t))
print(10 * '\n')
