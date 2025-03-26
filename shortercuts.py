# -*- coding: utf-8 -*-
"""
Created on Sat May 20 08:48:19 2017

@author: aes05kgb
The following are just shortened and renamed versions of python functons
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np


from numpy import concatenate,shape,kron,exp,array,pi,round,floor,ceil,square,sqrt,power,copy,empty,zeros_like,squeeze,matrix,arange,linspace,isnan
from numpy import log as ln
from numpy import log10 as log
import pandas as pd
from pandas import DataFrame as frame


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'axes.titlesize':'x-large',
          'legend.fontsize':'x-large'}
pylab.rcParams.update(params)

def pltsize(x1=10,x2=16):
    plt.rcParams['figure.figsize'] = [x2, x1]
    return


#____________________________________________________________________________________________
def twodm(x): 
    s=shape(x)
    if len(s)>2:
        x=np.squeeze(x)
        s=shape(x)    
    if len(s)==1:
        x=x.reshape(s[0],1)
    if len(s)==0:
        x=x.reshape(1,1)
    return x

#This will give 2.d shape to the array, preserving 2.d. shape if it exists, otherwise a column vector
def twodma(x):
    if type(x) !=np.ndarray:
        x=np.array(x)
    x=twodm(x)
    return x
   
#This will give 2.d. to the the array, but enforcing a column array, or row array if one of the dimensions is 1
def twodmca(x,column=True):
    x=squeeze(twodma(x))
    s=shape(x)
    if len(s)==0:
        x=x.reshape(1,1)
    if len(s)==1:
        if column:
            x=x.reshape(s[0],1)
        else:
            x=x.reshape(1,s[0])
    return x    
    
def checkob(ob):
    if type(ob)==list:
        if len(ob)==1:
           ob=[ob[0],1] 
    if type(ob)==int:
        ob=[ob,1]
    return ob
    
def ones(ob):
    ob=checkob(ob)
    return twodm(np.ones(ob))

def zeros(ob):
    ob=checkob(ob)
    return twodm(np.zeros(ob))
    
def cov(x):
    return twodm(np.cov(x.T))    

def rows(x):
    return shape(twodma(x))[0]

def cols(x):
    return shape(twodma(x))[1]
    
def sumc(x):
    return twodm(np.sum(x,axis=0))

def cumsumc(x):
    return twodm(np.cumsum(x,axis=0))

def sumc(x):
    x=twodma(x)
    return twodm(np.sum(x,axis=0))

def meanc(x):
    x=twodma(x)
    return twodm(np.mean(x,axis=0))

def medianc(x):
    x=twodma(x)
    return twodm(np.median(x,axis=0))
    
def stdc(x):
    x=twodma(x)
    return twodm(np.std(x,axis=0))
    
def minc(x):
    x=twodma(x)
    return twodm(np.min(x,axis=0))    

def maxc(x):
    x=twodma(x)
    return twodm(np.max(x,axis=0))    

def cc(ob):
    if type(ob[0])==pd.core.frame.DataFrame or type(ob[0])==pd.core.series.Series:
        return pd.concat(ob,axis=1)
    else:
        return concatenate(ob,axis=1)

def rc(ob):
    if type(ob[0])==pd.core.frame.DataFrame or type(ob[0])==pd.core.series.Series:
        return pd.concat(ob,axis=0)
    else:
        return concatenate(ob,axis=0)
    
def reshape(x,rows,cols):
    return np.reshape(x,[cols,rows]).T

def findex(x,start=1):
    return from1(frame(x),start)

