# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:43:24 2016

@author: aes05kgb
"""

from shortercuts import *

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
    
    return(listx_,listp_,minx,maxx,minp,maxp)

    
#__________________________________________________________________ 
def domain(x):
    x=np.array(x)
    s=[x>=0]
    t=[x<0]
    pos=np.array(s).astype(int)
    neg=np.array(t).astype(int)
    return pos,neg
#__________________________________________________________________     
#Construct a utility Function (u) and its certainty equivalent (iu)
def u_function(alpha=1.0,beta=1.0,lamda=1.0,uform='power',usym=False,expscale=1):
    np.seterr(all='ignore')    
    if usym:
        alpha=beta 
    if uform=='power':
        def u(x):
            x=np.array(x)
            s,t=domain(x)
            u=(x*s)**alpha[0] -lamda*((np.abs(x)*t)**beta[0])
            return np.squeeze(u)
        def iu(u):
            u=np.array(u)
            s,t=domain(u)
            x=(np.abs(u)*s)**(1/alpha[0]) - ((np.abs(u)*t/lamda)**(1/beta[0]))
            return np.squeeze(x)    
        
    if uform=='exp': 
        talpha=alpha/expscale
        tbeta=beta/expscale
        u_=np.isclose(talpha[0],0)
        _u=np.isclose(tbeta[0],0)
        def u(x):
            x=np.array(x)
            s,t=domain(x)
            if u_:
                upper=x
            else:
                upper=(1-np.exp(-talpha[0]*x))/talpha[0]
                upper=np.nan_to_num(upper)
            if _u:
                lower=x
            else:
                lower=(1-np.exp(-tbeta[0]*x))/tbeta[0]                    
                lower=np.nan_to_num(lower)            
            u=s*upper+ lamda*t*lower
            
            return np.squeeze(u)
            
        def iu(u):
            #print(alpha[0])
            u=np.array(u)
            s,t=domain(u)
            if u_:
                upper=u
            else:
                upper=(-np.log(1-talpha[0]*u)/talpha[0])
                upper=np.nan_to_num(upper)
            if _u:
                lower=u
            else: 
                lower=(-np.log(1-tbeta[0]*u/lamda)/tbeta[0])
                lower=np.nan_to_num(lower)
            x=s*upper+t*lower
            
            return np.squeeze(x)  
        
        
    
    if uform=='Exp':
        talpha=alpha/expscale
        tbeta=beta/expscale
        u_=np.isclose(talpha[0],0)
        _u=np.isclose(tbeta[0],0)
        def u(x):
            x=np.array(x)
            s,t=domain(x)
            if u_:
                upper=x
            else:
                upper=(1-np.exp(-talpha[0]*np.abs(x)))/talpha[0]
                upper=np.nan_to_num(upper)
            if _u:
                lower=x
            else:
                lower=-(1-np.exp(-tbeta[0]*np.abs(x)))/tbeta[0]
                lower=np.nan_to_num(lower)                      
            u=s*upper+ lamda*t*lower
            return np.squeeze(u)
            
        def iu(u):
            #print(alpha[0])
            u=np.array(u)
            s,t=domain(u)
            if u_:
                upper=u
            else:
                upper=(-np.log(1-talpha[0]*np.abs(u))/talpha[0])
                upper=np.nan_to_num(upper)
            if _u:
                lower=u
            else: 
                lower=(np.log(1+tbeta[0]*u/lamda)/tbeta[0])
                lower=np.nan_to_num(lower)
            x=s*upper+t*lower
            return np.squeeze(x)  

    return u,iu
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
#__________________________________________________________________ 
def w_function(gama,delta,wform='k&t'):
    np.seterr(all='ignore')
    if wform=='k&t':
        def weight_pos(p):
            f1=p**gama[0]
            f2=(p**gama[0] + (1-p)**gama[0])**(1/gama[0]);
            fp=f1/f2
            return fp
        def weight_neg(p):
            f1=p**delta[0]
            f2=(p**delta[0] + (1-p)**delta[0])**(1/delta[0]);
            fn=f1/f2
            return fn
    elif wform=='power':
        def weight_pos(p):
            fp=p**gama[0]
            return fp
        def weight_neg(p):
            fn=p**delta[0]
            return fn
    elif wform=='prelecI':            
        def weight_pos(p):
            f1=(np.abs(np.log(p)))**gama[0]
            fp=np.exp(-f1)
            return fp
        def weight_neg(p):
            f1=(np.abs(np.log(p)))**delta[0]
            fn=np.exp(-f1)
            return fn
    elif wform=='prelecII':
        def weight_pos(p):
            f1=(gama[1]*np.abs(np.log(p))**gama[0])
            fp=np.exp(-f1)
            return fp
        def weight_neg(p):
            f1=(delta[1]*np.abs(np.log(p))**delta[0])
            fn=np.exp(-f1)
            return fn
    elif wform=='g&h':
        def weight_pos(p):
            f1=gama[1]*(p**gama[0])
            f2=(gama[1]*(p**gama[0]) + (1-p)**gama[0])
            fp=f1/f2
            return fp
        def weight_neg(p):
            f1=delta[1]*(p**delta[0])
            f2=delta[1]*(p**delta[0]) + (1-p)**delta[0]
            fn=f1/f2
            return fn
    elif wform=='beta':
        '''
        def weight_pos(p):
            r=np.exp(gama[0])
            alpha=r*np.exp(gama[1])
            beta=r*np.exp(-gama[1])
            fp=sc.beta.cdf(p,alpha,beta)
            return fp
        def weight_neg(p):
            r=np.exp(delta[0])
            alpha=r*np.exp(delta[1])
            beta=r*np.exp(-delta[1])
            fn=sc.beta.cdf(p,alpha,beta)
            return fn
        '''
        def weight_pos(p):
        
            alpha=gama[0]
            beta=gama[1]
            fp=sps.beta.cdf(p,alpha,beta)
            return fp
        def weight_neg(p):
            alpha=delta[0]
            beta= delta[1]
            fn=sps.beta.cdf(p,alpha,beta)
            return fn
        
    return weight_pos,weight_neg

#__________________________________________________________________ 
def pcalc(x,p,alpha=1.0,beta=1.0,lamda=1.0,gama=1.0,delta=1.0,uform='power',wform='k&t',usym=False,expscale=1,compress=True,processed=False,fast=False,more=False):
        
    if fast==False:
        processed=False
    if processed==False:
       x,p,minx,maxx,minp,maxp=equalize_and_compress_and_sort(x,p,compress)
    alpha=twodmca(alpha)
    beta=twodmca(beta)
    lamda=twodmca(lamda)
    gama=twodmca(gama)
    delta=twodmca(delta) 
    uf,iuf=u_function(alpha,beta,lamda,uform,usym,expscale)
    weight_pos,weight_neg=w_function(gama,delta,wform)
    
    cp=dcumdist(p)
    dp=cumdist(p)
    
    w1=invdcumdist(weight_pos(cp))
    w2=invcumdist(weight_neg(dp))
    #print(x)
    #print(p.round(6))
    #print(cp.round(6))
    #print(dp.round(16))
    #print(w1.round(6))
    #print(w2.round(6))
    #stop()
    s,t=domain(x)                           #Domain assignment
    w=twodmca(np.squeeze((w1*s) + (w2*t)),False) #Matrix of probabiity weights
    u=twodmca(np.squeeze(uf(x)),False)           #The matrix of utilities
    if processed==False:
        maxu=twodmca(uf(maxx))
        minu=twodmca(uf(minx))    
    eu=np.sum(w*u,axis=1)                   #Expected uitlities  under transformed probabilities  
    eu_=np.sum(p*u,axis=1) #Expected utility under orginal probabilities
    ce=iuf(eu)             #Certainty equivalents under transformed probabilities
    if fast==False:
        ce_=iuf(eu_)           #Certainty equivalents under original probabilties  
        ev,sd=mav(x,p)
        perceived_ev,perceived_sd=mav(x,w)
        return u,ce,ce_,eu,eu_,w,ev,sd,perceived_ev,perceived_sd,x,p,minx,maxx,minp,maxp,minu,maxu
    else:
        #print(np.shape(u))
        return u,ce,eu,w
        
#__________________________________________________________________        
def mav(x,p):
    ex=np.sum(p*x,axis=1)         #Expected values
    exs=np.sum(p*(x**2.0),axis=1) #Expected squaredvalues
    va=exs-ex**2.0                #Variance
    sd=va**0.5                    #Standard deviation
    return ex,sd
#__________________________________________________________________     
class lottery():
    def __init__(self,x,p,alpha=1.0,beta=1.0,lamda=1.0,gama=1.0,delta=1.0,uform='power',wform='k&t',usym=False,expscale=1,compress=True,processed=False,fast=False):    
        self.x=x
        self.p=p
        self.alpha=alpha
        self.beta=beta
        self.lamda=lamda
        self.gama=gama
        self.delta=delta
        self.wform=wform
        self.uform=uform
        self.usym=usym
        self.processed=processed
        self.compress=compress
        
        if self.compress==False:
            print("\nWARNING compress=False is set and will invalidate the PrHeuristic and FOSDOM if there are repeated payoffs in the GAMBLES\n" )
        
        if fast==False:
            self.u,self.ce,self.ce_,self.eu,self.eu_,self.w,self.ev,self.sd,self.perceived_ev,self.perceived_sd,self.X,self.P,self.minx,self.maxx,self.minp,self.maxp,self.minu,self.maxu=pcalc(x,p,alpha,beta,lamda,gama,delta,uform,wform,usym,expscale,compress,processed,fast)
        else:
            self.u,self.ce,self.eu,self.w=pcalc(x,p,alpha,beta,lamda,gama,delta,uform,wform,usym,expscale,compress,processed,fast)
            
        if fast==False:
            self.ce=twodmca(self.ce)
            self.RESULTS=pd.DataFrame(np.array(self.ce),columns=["CE_PT"])
            self.RESULTS["CE_EUT"]=self.ce_
            self.RESULTS["EU_PT"]=self.eu
            self.RESULTS["EU_EUT"]=self.eu_
            self.RESULTS["EV"]=self.ev
            self.RESULTS["EV_PT"]=self.perceived_ev
            self.RESULTS["Stdv"]=self.sd
            self.RESULTS["Stdv_PT"]=self.perceived_sd
            self.RESULTS["MinPayoff"]=self.minx
            self.RESULTS["MaxPayoff"]=self.maxx
            self.RESULTS["MinUtility"]=self.minu
            self.RESULTS["MaxUtility"]=self.maxu
            self.RESULTS["ProbMinPayoff"]=self.minp
            self.RESULTS["ProbMaxPayoff"]=self.maxp
        
        
        
            self.description=["The following are inputs and outputs for the object and can be called using objectname.term, where the terms are ","","alpha=utility parameter in gain domain","beta=utility parameter in loss domain","lambda=utility parameter relative weighting of gains and losses","gama=prob weighting parameter in gain domain","delta=prob weighting parameter in loss domain","wform =   weighting form, 'k&t', 'power', 'prelecI', 'prelecII', 'g&h' 'beta' (the last three are two parameter forms so gama and delta need to have two parameters each)","uform =   utility form, 'power' (constant relative RA) or 'exp' (constand Abs RA)","","u=matrix of utilities (sorted ascending payoff expanded to equal dimension)","ce=vector of certainty equivalents","ce_=vector of certaining under exected utility","eu=vector of expected utilities","eu_=vector or expected utility under expected utility","w=transformed probabilities (sorted ascending payoff expanded to equal dimension)","ev =vector of expected values","sd=vector of standard deviaions","perceived_ev=perceived expected value under transformed probabilities", "perceived_sd=perceived standard deviation of transformed probabilities","X=sorted payoffs expanded to equal dimension","P=sorted probabilities expanded to equal dimension","x=original payoffs (unsorted, orginal dimension)","p=original probabilities (unsorted, orginal dimension)","RESULTS= a data frame of results for CE,EU, and EV etc","PXWU= Data frame of probs, transformed probs, payoffs and Utilities"]
        
    

            self.labsP=[]
            self.labsW=[]
            self.labsX=[]
            self.labsU=[]
            self.dim=cols(self.X)
            for i in range(self.dim):
                self.labsP.append("P" + str(i+1))
                self.labsX.append("X" + str(i+1))
                self.labsW.append("W" + str(i+1))
                self.labsU.append("U" + str(i+1))
    

            self.PXWU=pd.DataFrame(self.P,columns=self.labsP)
            for i in range(self.dim):
                self.PXWU[self.labsX[i]]=self.X[:,i]
            for i in range(self.dim):
                    self.PXWU[self.labsW[i]]=self.w[:,i]
            for i in range(self.dim):
                self.PXWU[self.labsU[i]]=self.u[:,i]    
    
    
    def descript(self):        
            for i in range(len(self.description)):
                print(self.description[i])
        
    
#_____________________________________________________________________________             
def lotteries(x1,p1,x2,p2,alpha=1.0,beta=1.0,lamda=1.0,gama=1.0,delta=1.0,uform='power',wform='k&t',usym=False,expscale=1,compress=True,processed=False,fast=False,sod=1000):
                              
    lot1=lottery(x1,p1,alpha,beta,lamda,gama,
                           delta,uform,wform,usym,expscale,compress,processed,fast)
    lot2=lottery(x2,p2,alpha,beta,lamda,gama,
                           delta,uform,wform,usym,expscale,compress,processed,fast)                  
    
    #Constructing contextual Utility
    maxus=cc([lot1.maxu,lot2.maxu])
    minus=cc([lot1.minu,lot2.minu])
    maxus=(np.max(maxus,axis=1))
    minus=(np.min(minus,axis=1))
    mdiff=twodmca(maxus-minus)
    cu1=lot1.u/mdiff
    cu2=lot2.u/mdiff    
    ecu1=twodmca(np.sum(lot1.w*cu1,axis=1))      
    ecu2=twodmca(np.sum(lot2.w*cu2,axis=1)) 
    
    ecu_EU1=twodmca(np.sum(lot1.P*cu1,axis=1))      
    ecu_EU2=twodmca(np.sum(lot2.P*cu2,axis=1)) 
    #print(ecu1,ecu2)
    lot1.RESULTS['CNTXU_PT']=ecu1
    lot2.RESULTS['CNTXU_PT']=ecu2
    lot1.RESULTS['CNTXU_EUT']=ecu_EU1
    lot2.RESULTS['CNTXU_EUT']=ecu_EU2
    
    T=rows(lot1.X)
    FODOM=np.array([])
    SODOM=np.array([])
    SODOM2=np.array([])
    for t in range(T):                      
        x1=lot1.X[t]
        x2=lot2.X[t]
        w1=lot1.P[t]
        w2=lot2.P[t]
        w_1=lot1.w[t]
        w_2=lot2.w[t]
        dom,som=FOD(x1,w1,x2,w2,sod)
        dom,som2=FOD(x1,w_1,x2,w_2,sod)
        
        FODOM=np.append(FODOM,dom)
        SODOM=np.append(SODOM,som)
        SODOM2=np.append(SODOM2,som2)
             
    
    if rows(lot1.eu) != rows(lot2.eu):
        print(rows(lot1.eu))
        print(rows(lot2.eu))
        print("The two sets of lotteries are not the same length \n therefore cannot be treated as pairs")
        #sys.exit()                       
    if fast==False:                       
        lot1.DIFF=lot1.RESULTS-lot2.RESULTS
        lot2.DIFF=lot2.RESULTS-lot1.RESULTS
        
        max_x1=twodmca(lot1.RESULTS['MaxPayoff'])
        max_x2=twodmca(lot2.RESULTS['MaxPayoff'])
        min_x1=twodmca(lot1.RESULTS['MinPayoff'])
        min_x2=twodmca(lot2.RESULTS['MinPayoff'])
        prob_min1=twodmca(lot1.RESULTS['ProbMinPayoff'])
        prob_min2=twodmca(lot2.RESULTS['ProbMinPayoff'])
        
        mx=nobool(max_x1 >= max_x2)  # one has highest payoff
        max_x=mx*max_x1+(1-mx)*max_x2 #The highest payoff across te lotteries
        
        threshold1=(min_x1-min_x2)/max_x
        threshold2=(prob_min1-prob_min2)
        
        passedx=np.array([np.abs(threshold1) >=0.1]) #passes the threshold to choose 1 or other
        passedp=np.array([np.abs(threshold2) >=0.1]) #passes the threshold to choose 1 or other
        
        one_has_biggest_max_x=twodmca(np.squeeze(np.array([max_x1 > max_x2])))
        two_has_biggest_max_x=twodmca(np.squeeze(np.array([max_x1 < max_x2])))
        one_has_biggest_min_x=twodmca(np.squeeze(np.array([min_x1 > min_x2])))
        two_has_biggest_min_x=twodmca(np.squeeze(np.array([min_x1 < min_x2])))
        one_has_the_smaller_probability_of_min_x=twodmca(np.squeeze(np.array([prob_min1 < prob_min2])))        
        two_has_the_smaller_probability_of_min_x=twodmca(np.squeeze(np.array([prob_min1 > prob_min2])))        
        
        passedx=twodmca(np.squeeze(passedx))
        passedp=twodmca(np.squeeze(passedp))
        
        one=[]
        two=[]
        
        for i in range(len(max_x1)):
            decision=False
            if passedx[i]:
                #print("Passed the first threshold")
                if one_has_biggest_min_x[i]:
                 #   print("One has biggest min payoff")
                    one.append('One')
                    two.append('One')
                    decision=True
                elif two_has_biggest_min_x[i]:
                  #  print("Two has biggest min payoff")
                    one.append('Two')
                    two.append('Two')
                    decision=True
            else: 
                if passedp[i]:
                   # print("Passed the second threshold")
                    if one_has_the_smaller_probability_of_min_x[i]:
                    #    print("One has smaller probability of getting the minimum")
                        one.append('One')
                        two.append('One')
                        decision=True
                    elif two_has_the_smaller_probability_of_min_x[i]:
                     #   print("Two has smaller probability of getting the minimum")
                        one.append('Two')
                        two.append('Two')
                        decision=True
                else:
                    
                    if one_has_biggest_max_x[i]:  
                      #  print("Passed by biggest Max")                  
                        one.append('One')
                        two.append('One')
                        decision=True
                    elif two_has_biggest_max_x[i]:   
                       # print("Passed by biggest Max")
                        one.append('Two')
                        two.append('Two')
                        decision=True
            if decision==False:
                    one.append("Draw")
                    two.append("Draw")
        lot1.DIFF['PrHeuristic']=(one)
        lot2.DIFF['PrHHeristic']=(two)
        lot1.DIFF['FOSDOM']=FODOM
        lot2.DIFF['FOSDOM']=FODOM
        lot1.DIFF['SOSDOM']=SODOM
        lot2.DIFF['SOSDOM']=SODOM
        lot1.DIFF['SOSDOM_PT']=SODOM2
        lot2.DIFF['SOSDOM_PT']=SODOM2
        
        #lot1.DIFF['Cntxt Diff']=lot1.DIFF
    else:
        diff.RESULTS=[]
    #print(np.shape(nobool(one)))
   # print(nobool(one))
              
    return lot1,lot2                       
                           

def augment(X,x):
    X=np.append(X,x[0])
    x=twodmca(np.delete(x,0))
    n=rows(x)-1
    return X,x,n

def FOD(x1,p1,x2,p2,sod=1000):    
    x1=twodmca(x1)
    x2=twodmca(x2)
    p1=twodmca(p1)
    p2=twodmca(p2)    
        
    x=[]
    P1=[]
    P2=[]
    n1=rows(x1)-1
    n2=rows(x2)-1
    
    while n1>=0 or n2>=0:
        if (n1>=0 and n2>=0):
            if x1[0] < x2[0]:
                x,x1,n1=augment(x,x1)
                P1,p1,m1,=augment(P1,p1)
                P2=np.append(P2,0)
            elif x2[0] < x1[0]:
                x,x2,n2=augment(x,x2)
                P2,p2,m2,=augment(P2,p2)
                P1=np.append(P1,0)
            elif x2[0] == x1[0]:
                x,x2,n2=augment(x,x2)
                x1=twodmca(np.delete(x1,0))
                n1=n1-1
                P1,p1,m1,=augment(P1,p1)
                P2,p2,m2,=augment(P2,p2)
        elif n2>=0:
            x,x2,n2=augment(x,x2)
            P2,p2,m2,=augment(P2,p2)
            P1=np.append(P1,0)
        elif n1>=0:
            x,x1,n1=augment(x,x1)
            P1,p1,m1,=augment(P1,p1)
            P2=np.append(P2,0)
    x=twodmca(x,False)
    mx=np.max(x)
    mn=np.min(x)
    z=np.linspace(mn,mx,sod)##Accurate sod increase 5 to 500
    
    P1=twodmca(P1,False)
    P2=twodmca(P2,False)
    
    W1=twodmca(dcumdist(P1),False)
    W2=twodmca(dcumdist(P2),False)
    Q1=twodmca((cumdist(P1)),False)
    Q2=twodmca((cumdist(P2)),False)

    I1=np.zeros([rows(z),1])
    I2=np.zeros([rows(z),1])
    
    Z1=z.copy()
    Z2=z.copy()
    for j in range(rows(z)-1):
        for k in range(cols(x)-1):
            
            if z[j]<x[:,k+1] and z[j]>=x[:,k]:
                Z1[j]=Q1[:,k]
                Z2[j]=Q2[:,k]
                
    for j in range(rows(z)-1):    
        I1[j+1]=I1[j]+Z1[j+1]*(z[j+1]-z[j])
        I2[j+1]=I2[j]+Z2[j+1]*(z[j+1]-z[j])
    
    dominant1=True
    dominant2=True
    do1=False
    do2=False

    for i in range(rows(I1)):
        if I1[i] < I2[i]:
             do1=True
        if I2[i] < I1[i]:    
             do2=True
    
    for i in range(rows(I1)):             
        if I1[i] > I2[i]:
            do1=False
        if I2[i] > I1[i]:
            do2=False         

    for i in range(cols(W1)):
        
        if W1[:,i] < W2[:,i]:
            dominant1=False
        if W2[:,i] < W1[:,i]:    
            dominant2=False
     
            
    if dominant1==True and dominant2==True:
        dom="none"
    elif dominant1==False and dominant2==False:
        dom="none"    
    elif dominant1==True and dominant2==False:
        dom="One"
    elif dominant1==False and dominant2==True:
        dom="Two"
    
    if do1==True and do2==True:
        som="none"
    elif do1==False and do2==False:
        som="none"    
    elif do1==True and do2==False:
        som="One"
    elif do1==False and do2==True:
        som="Two"    
    return dom,som     

                           
#__________________________________________________________________ 
#Some helper functions
#By default returns a column vector, but if True is a row vector


def multiplybyscalar(x,scalar):
    noshape=True
    x_=np.array(x)
    t=np.shape(x)
    x_=x.copy()
    
    for i in range(t[0]):
        s=np.shape(x[i])
        try:
           s=s[0]
           shape=True
           noshape=False
           x_[i]=np.multiply(x[i],scalar)
        except:
           shape=False 
        
    if noshape:
        x_=np.multiply(x,scalar)
    return(x_)

def nobool(x):
    z=twodmca(np.squeeze(twodmca(x).astype(int)))
    return z    


def runandsave(dta,sigrange,alprange,phirange,outputname,numberofsets):    
    

    x1=dta[['x11','x12']]
    x2=dta[['x21','x22']]
    p1=dta[['p11','p12']]
    p2=dta[['p21','p22']]
    
    z1=dta[['z11','z12']]
    z2=dta[['z21','z22']]
    w1=dta[['w11','w12']]
    w2=dta[['w21','w22']]    
    
    if numberofsets>=3:
        X1=dta[['X11','X12']]
        X2=dta[['X21','X22']]
        P1=dta[['P11','P12']]
        P2=dta[['P21','P22']]
    
    
    numalp=np.round(10*(alprange[1]-alprange[0])+1)
    numsig=np.round(10*(sigrange[1]-sigrange[0])+1)
    numphi=np.round(10*(phirange[1]-phirange[0])+1)

    num_gamble=np.shape(np.array(x1))
    num_gamble=num_gamble[0]
    
    changepoints=[]
    
    count=1
    phi=phirange[0]
    for m in range(int(numphi)):    
        alp=alprange[0] 
        for k in range(int(numalp)):
            sigma=sigrange[0] 
            for i in range(int(numsig)):
                #Setting the utility parameters
                a=sigma  #Curvature in the gain domain
                b=sigma  #Curvature in the loss domain

                #The weighting parameters, (will need two parameter pairs for some  forms)
                g=[alp,phi]   #Curvature in the gain domain 
                d=[1,1]   #Curvature in the loss domain

                #Loss aversion
                lamda=1
                weight='beta'
                lot1,lot2=lotteries(x1,p1,x2,p2,alpha=a,beta=b,lamda=lamda,gama=g,delta=d,uform='power',wform=weight,compress=True,processed=False,fast=False,sod=5)
        
                lot3,lot4=lotteries(z1,w1,z2,w2,alpha=a,beta=b,lamda=lamda,gama=g,delta=d,uform='power',wform=weight,compress=True,processed=False,fast=False,sod=5)
                
                if numberofsets >=3:
                    lot5,lot6=lotteries(X1,P1,X2,P2,alpha=a,beta=b,lamda=lamda,gama=g,delta=d,uform='power',wform=weight,compress=True,processed=False,fast=False,sod=5)

                    

                #I Will put us some components in the data frame below

                RESa=pd.DataFrame()
                RESa['GAMBLE']=np.arange(num_gamble)+1
                RESa['EV_DIFF']=lot1.DIFF['EV']
                RESa['CE_PT1']=lot1.RESULTS['CE_PT']
                RESa['CE_PT2']=lot2.RESULTS['CE_PT']
                RESa['CE_DIFF']=lot1.DIFF['CE_PT']
                RESa['SOSDOM']=lot1.DIFF['SOSDOM']
                RESa['SOSDOM_PT']=lot1.DIFF['SOSDOM_PT']
        
                RESb=pd.DataFrame()
                RESb['GAMBLE']=np.arange(num_gamble)+1
                RESb['EV_DIFF']=lot3.DIFF['EV']
                RESb['CE_PT1']=lot3.RESULTS['CE_PT']
                RESb['CE_PT2']=lot4.RESULTS['CE_PT']
                RESb['CE_DIFF']=lot3.DIFF['CE_PT']
                RESb['SOSDOM']=lot3.DIFF['SOSDOM']
                RESb['SOSDOM_PT']=lot3.DIFF['SOSDOM_PT']
 
                if numberofsets>=3:
                    RESc=pd.DataFrame()
                    RESc['GAMBLE']=np.arange(num_gamble)+1
                    RESc['EV_DIFF']=lot5.DIFF['EV']
                    RESc['CE_PT1']=lot5.RESULTS['CE_PT']
                    RESc['CE_PT2']=lot6.RESULTS['CE_PT']
                    RESc['CE_DIFF']=lot5.DIFF['CE_PT']
                    RESc['SOSDOM']=lot5.DIFF['SOSDOM']
                    RESc['SOSDOM_PT']=lot5.DIFF['SOSDOM_PT']
                       
                changepointa=0
                atone=1
                changepointb=0
                btone=1
                changepointc=0
                ctone=1
                
                
        
                for i in range(num_gamble-1):
                    if RESa.ix[i,'CE_DIFF']>0 and RESa.ix[i+1,'CE_DIFF']<0:
                        print("sigma=",np.round(sigma*1000)/1000,"alpha=",np.round(alp*1000)/1000,"phi=",np.round(phi*1000)/1000, "changepointa",i+1," to",i+2)
                        changepointa=(i+2)
                    if RESb.ix[i,'CE_DIFF']>0 and RESb.ix[i+1,'CE_DIFF']<0:
                        print("sigma=",np.round(sigma*1000)/1000,"alpha=",np.round(alp*1000)/1000,"phi=",np.round(phi*1000)/1000," changepointb",i+1," to",i+2)
                        changepointb=(i+2)
                    if numberofsets>=3:    
                        if RESc.ix[i,'CE_DIFF']>0 and RESc.ix[i+1,'CE_DIFF']<0:
                            print("sigma=",np.round(sigma*1000)/1000,"alpha=",np.round(alp*1000)/1000,"phi=",np.round(phi*1000)/1000," changepointc",i+1," to",i+2)
                            changepointc=(i+2)    
                
                    if RESa.ix[i,'CE_DIFF']>0:
                        atone=0
                    if RESb.ix[i,'CE_DIFF']>0:
                        btone=0
                    if numberofsets>=3:    
                        if RESc.ix[i,'CE_DIFF']>0:
                            ctone=0
                
                    if atone==1:
                        changepointa=1
                    if btone==1:
                        changepointb=1
                    if numberofsets>=3:
                        if ctone==1:
                            changepointc=1    
        
                if changepointa==0 or changepointa==1:
                        print("sigma=",np.round(sigma*1000)/1000,"alpha=",np.round(alp*1000)/1000,"phi=",np.round(phi*1000)/1000,"No change Pointa") 
                if changepointb==0 or changepointb==1:
                        print("sigma=",np.round(sigma*1000)/1000,"alpha=",np.round(alp*1000)/1000,"phi=",np.round(phi*1000)/1000,"No change Pointb") 
                if numberofsets>=3:        
                    if changepointc==0 or changepointc==1:
                            print("sigma=",np.round(sigma*1000)/1000,"alpha=",np.round(alp*1000)/1000,"phi=",np.round(phi*1000)/1000,"No change Pointc")         
            
                v=np.array([sigma,alp,phi,changepointa,changepointb,changepointc])
                if numberofsets==2:
                    dim=6
                else:
                    dim=6
                    
                v=np.reshape(v,(1,dim))
                
                #print(v)
                if count==1:
                    changepoints=np.append(changepoints,v)
                    changepoints=np.reshape(changepoints,(1,dim))
                    print(np.shape(changepoints))
                    count=count+1
                else:
                    changepoints=np.append(changepoints,v,axis=0)
                    print(np.shape(changepoints))
                    count=count+1     
                       
                sigma=sigma+0.10
            alp=alp+0.10    
        phi=phi+0.10
    
    namea=outputname
    fa=open(namea,"wb")
    pickle.dump(changepoints,fa)
    fa.close()
    return
    
    
    
