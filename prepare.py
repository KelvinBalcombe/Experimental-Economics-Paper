# -*- coding: utf-8 -*-
"""
Created on Sat May 20 08:48:19 2017

@author: aes05kgb
"""
import os

#For setting working directories that are different from the base
def cwd(level=0):
    v0=os.getcwd()
    v1=os.path.split(v0)[0]
    v2=os.path.split(v1)[0]
    v3=os.path.split(v2)[0]
    v4=os.path.split(v3)[0]
    v5=os.path.split(v4)[0]
    if level==0:
        return v0
    elif level==1 or level==-1:
        return v1
    elif level==2 or level==-2:
        return v2
    elif level==3 or level==-3:
        return v3
    elif level==4 or level==-4:
        return v4
    elif level==5 or level==-5:
        return v4
    
print('current path',cwd())    
#sys.path.append(cwd(-1))



#This is a file with some shortcuts that I like to use when coding
from shortercuts import frame,shape,cols,rows,exp,meanc,sumc,maxc,minc,ln,twodma,twodmca,ones,cc,rc,squeeze,array,findex,zeros,sqrt,stdc,pd,np,pltsize,plt
import pickle
import pystan
from pystan.misc import *

#These functions simplifying saving code, output and reading it back in
def report(ob,name):
    a=ob.summary(pars=name)
    df=frame(a['summary'],columns=a['summary_colnames'])
    return df

#Extract and print the MCMC output from a model run by STAN
def pull(ob):
    z=ob.extract()
    k=z.keys()
    for i in k:
        print(i,shape(z[i]))
    return z

#Save a model or mcmc output that has been compiled by STAN, or run using STAN
def savemodel(sm,name):
    name=name+'.pkl'
    with open(name, 'wb') as f:
        pickle.dump(sm, f)
    return

#Load a model previously compiled by STAN, or run using STAN
def loadmodel(name):
    name=name+'.pkl'
    sm = pickle.load(open(name, 'rb'))
    return sm


#This is the WAIC information criteria for models and for comparison of models
def WAICf(f): #f here is (MCMC trials T, by Number of Data Points N)
    #f[:,n] is a T dimensional array
    #f[t,:] is a N dimensional array
    N=shape(f)[1] #The number of data points
    T=shape(f)[0] #The number of MCMC trials
    lppd=0
    mi=[]
    vlppd=[]
    maxloglik=maxc(sumc(f.T))     #Maximum of the log-likelihood encountered in the sample

    for i in range(N):
        wi=twodma(f[:,i])         #vector of logged densities for a data point
        lpdi=lmean(wi)            #the logged-Mean (across the mcmc draws) of the densities
        vlppd=vlppd+[lpdi]
        mi=mi+[meanc(wi)]         #list of Mean-logged densities for each data point by increment
    vlppd=twodma(squeeze(array(vlppd)))
    mi=twodma(squeeze(array(mi))) #list of Mean-logged densities  as a vector
    si=[]
    for i in range(N):
        wi=twodma(f[:,i])                 #vector of logged densities for a data point 
        S=rows(wi)                        #The number of MCMC samples
        si=si+[sumc((wi-mi[i])**2)/(S-1)] #list of sum of squared deviations of wi from its mean for each data point
    si=twodma(squeeze(array(si)))         #si as an array
    elppdi=vlppd-si
    p_waic2=sumc(si)                 #This is the estimated variance of logged density for a point, pwaic type 2 (eq 13)
    p_waic1=2*sumc(vlppd)-2*sumc(mi) #This is the pwaic type 1
    elppd=sumc(vlppd)-p_waic2        #This needs to be maximised, it is called the elpd in the paper  
    waic=-2*elppd                    #This needs to be minimised   
    sewaic=2*(sqrt(N)*stdc(elppdi))
    out1=frame([float(waic),float(elppd),float(p_waic1),float(p_waic2),float(maxloglik),float(sewaic)])
    out1=out1.T
    out1.columns=['waic','elpd','p_waic1','p_waic2','maxloglik','se_waic']
    return out1,elppdi

def comparewaic(f1,f2):
    out1a,out2a=WAICf(f1)
    out1b,out2b=WAICf(f2)
    #This is the negativised eldp difference, a negative score indicates that the first model is preferred
    out=frame([float(-sumc(out2a-out2b)),float(sqrt(rows(out2a))*stdc(out2a-out2b))])
    out=out.T
    out.columns=['-elpd','se']
    print('A negative value favours the first model')
    return out


def lmean(z):
    T=rows(z)
    m=maxc(z)
    d=exp(z-m)
    return -ln(T)+m+ln(sumc(d))



def sortc(x,n=0,ascending=True):
    x=pd.DataFrame(x)
    z=twodma(x.sort_values(by=n,ascending=ascending))
    return(z)  

#__________________________________________________________________
#equalize the dimensions of a set (by adding 0s with 0 probabilities)
def equalize(x,p):
    x=np.array(x)
    p=np.array(p)
    if rows(x) != rows(p):
       print("Payoff and probabilty matrices are not the same length")
    n=1
    try:
        for i in range(len(x)):
            if len(x[i]) >n:
                n=len(x[i])
        for i in range(len(x)):
            while len(x[i])<n:
                x[i]=np.append(x[i],0)
                p[i]=np.append(p[i],0) 
        x=np.array(list(x))
        p=np.array(list(p))
    except:
        pass
    return x,p
#__________________________________________________________________ 
#Removes repeated values of payoffs and merges them (for one lottery)
def compressone(x,p):
    x=twodmca(x,False)
    p=twodmca(p,False)
    #print(p)
    for i in range(cols(x)):
        for j in range(i+1,cols(x)):
            if x[:,i]==x[:,j]:
                x[:,j]=0
                p[:,i]=p[:,i]+p[:,j]
                p[:,j]=0
    return x,p
#__________________________________________________________________ 
#Compresses and equalises a set of 
def equalize_and_compress(x,p,compress=True) :
    x,p=equalize(x,p)
    if compress:
        
        x=twodmca(x,False)
        p=twodmca(p,False)
        
        for i in range(rows(x)):
            x[i,:],p[i,:]=compressone(x[i,:],p[i,:])
    return(x,p)
#__________________________________________________________________ 
#This function takes a set of loteries, orders them and compresses them in the sense that common payoffs are merged    
def equalize_and_compress_and_sort(x,p,compress=True):
    #Sorts the lotteries in ascending order with  to payoffs
    x,p=equalize_and_compress(x,p,compress)# lotteries in rows
    #print(p)
    p=twodmca(p,False).T
    z=((np.sum(p,axis=0))-1.0)
    #print(z)
    test=twodmca([np.abs(z) <10**-10])
    #print(test)
    ntest=nobool(test)
    
    if np.all(test)==False:
       print("warning, one of the lotteries has probabilities that do not sum to one, the number of the lottery is below")
       for i in range(len(ntest)):
           if ntest[i]==0:
              print(i+1)
       #sys.exit()
       
    x=twodmca(x,False).T
    
    k=cols(p)
    listp_=[]
    listx_=[]
    s=np.shape(x)
    if s[0]==1:
       p=p.T
       x=x.T
       k=cols(p)
    
    for i in range(k):
        z=cc([p[:,[int(i)]],x[:,[int(i)]]])
        z=sortc(z,1)
        p_=z[:,[0]]
        x_=z[:,[1]]
        listp_.append(p_.T)
        listx_.append(x_.T)
            
    listp_=twodmca(np.squeeze(listp_),False)      
    listx_=twodmca(np.squeeze(listx_),False)
    #The following just gets the maximums and minimums of x and associated robs
    p=listp_
    x=listx_
    
    listxmin=[]
    listpmin=[]
    for i in range(rows(p)):
        j=0
        while p[i,j]==0:
            j=j+1
        listxmin.append(x[i,j]) 
        listpmin.append(p[i,j])
    
    listxmax=[] 
    listpmax=[]     
    for i in range(rows(p)):
        j=cols(p)-1
        try:
            while p[i,j]==0:
                j=j-1
        except:
            pass
        listxmax.append(x[i,j]) 
        listpmax.append(p[i,j])
        
    minx=np.array(listxmin)
    maxx=np.array(listxmax)    
    minp=np.array(listpmin)
    maxp=np.array(listpmax)   
    
    return(listx_,listp_)

    
#__________________________________________________________________     
#The cumulative distribution    
def cumdist(p):
    return np.cumsum(p,axis=1)      
#__________________________________________________________________    
#The inverse of the cumlative distribution   
def invcumdist(p):
    z=p.copy()
    x=np.diff(p,n=1,axis=1)
    z[:,1:]=x
    return(z)    

#__________________________________________________________________ 
#The decumulative distribution    
def dcumdist(p):
    pd=np.ones([rows(p),cols(p)])
    b=(1-cumdist(p))
    pd[:,1:]=b[:,0:-1]
    return(pd)
#__________________________________________________________________     
#The inverse of the decumulative distribution    
def invdcumdist(p):
    x=np.ones([rows(p),cols(p)])
    z=np.abs(np.diff(p,n=1,axis=1))
    x[:,0:-1]=z
    h=1-((np.sum(z,axis=1)))
    x[:,cols(x)-1]=h
    return(x)

#This function is used above
def nobool(x):
    z=twodmca(np.squeeze(twodmca(x).astype(int)))
    return z    