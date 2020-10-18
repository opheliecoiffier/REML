# -*- coding: utf-8 -*-
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


#%%
#################################
#Difference betwween ML and REML
###################################

# We show the difference between an unbiased Normal distribution and 
#a biased Normal distribution.
#The formula used in sigma of the y2 distribution is explained in the report

def loi_normale(x, mu, sigma):
    return sp.stats.norm.pdf(x,loc = mu, scale=sigma)

x = np.arange(-5,5,0.5)
y1 = loi_normale(x, mu=1, sigma=1.5)
y2 = loi_normale(x, mu=np.mean(x), sigma=1.5*((len(x)-1)/len(x)))
plt.plot(x,y1, label='N(1, 1.5) : unbiased')
plt.plot(x, y2, label="N(-0.25,1.43) : biased")
plt.axvline(np.mean(x), color='lightgreen', label="mean(x)", linestyle='--')
plt.axvline(1, linestyle='--', label='mu', color='lightblue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.title("Gaussian biased and unbiased")
plt.tight_layout()

#%%
#############################
#An example with real data
#############################

#The data used
#-------------
df = []
df = pd.DataFrame({'Treat' : [0, 1, 0, 1], 'Resp' : [10, 25, 3, 6], 'Ind' : [1, 1, 2, 2]})
print(df)

#%%
#The maximum of the log-likelihood
#-----------------------------------
def log_vraissemblance(data, sigma, mu):
    L = []
    for x in data:
        y =  loi_normale(x, mu = mu, sigma = sigma)
        L.append(np.log(y))
    return np.sum(L)  

data = sorted(df.Resp)
mu = np.mean(df.Resp)
sig = np.var(data)
x = range(int(np.floor(sig)))
print(sig)
z = []
for sigma in range(int(np.floor(sig))):
    z.append(log_vraissemblance(data=data, sigma=sigma, mu=mu))
    
plt.plot(x,z)
plt.title("Log-likelihood with mu=mean(Resp) depending on sigma")
plt.xlabel('Sigma')
plt.ylabel('Log-likelihood')
d = pd.DataFrame({'x' : x, 'z' : z})
print(d)
print("The log_likelihood's maximum coordinates are ",d.iloc[d['z'].idxmax()])

#%%
#Linear regression with our data
#---------------------------------------
linear = ols('Resp~Treat',data=df).fit()
print(linear.summary())
#We have the value of alpha = 9 and mu = 6.5

linear_reg = sm.OLS(df.Resp, df.Treat)
linear_reg_fit = linear_reg.fit()
print(linear_reg_fit.summary())

#We find the same coordinates of the log_likelihood when we use the linear regression
#than when we use the graph
#So we see that the variance is biased with linear regression
#Besides, We have the fixed value for our alpha and mu : respectively 15.5 =(6.5+9) and 6.5

#%%
#Linear mixed regression with our data
#---------------------------------------
mixed_random = smf.mixedlm("Resp ~ Treat", df, groups = 'Ind')
mixed_fit = mixed_random.fit()
print(mixed_fit.summary())

#We obtain the Fixed effects : Intercept & Treat
#And the Random effects : Ind Var, the Residual : Scale
#The log-likelihood has changed 
