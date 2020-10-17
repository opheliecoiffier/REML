# -*- coding: utf-8 -*-
import statsmodels.formula.api as smf
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import rlm
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

df = []
df = pd.DataFrame({'Treat' : [0, 1, 0, 1], 'Resp' : [10, 25, 3, 6], 'Ind' : [1, 1, 2, 2]})
print(df)

#%%
#linear regression with our data
#----------------------------------------
linear_reg = sm.OLS(df.Resp, df.Treat)
linear_reg_fit = linear_reg.fit()
print(linear_reg_fit.summary())

#%%
#linear mixed regression with our data
#---------------------------------------
mixed_random = smf.mixedlm("Resp ~ Treat", df, groups = 'Ind')
mixed_fit = mixed_random.fit()
print(mixed_fit.summary())

#We obtain the Fixed effects : Intercept & Treat
#And the Random effects : Ind Var, the Residual : Scale

#%%
def loi_normale(x, mu, sigma):
    return sp.stats.norm.pdf(x,loc = mu, scale=sigma)

def log_vraissemblance(data, sigma, mu):
    L = []
    for x in data:
        y =  loi_normale(x, mu = mu, sigma = sigma)
        L.append(np.log(y))
    return np.sum(L)  

x = sorted(df.Resp)
mu = np.mean(df.Resp)
z = []
for sigma in x:
    z.append(log_vraissemblance(data=x, sigma=sigma, mu=mu))

plt.plot(x,z)
plt.title("Log-likelihood with mu=mean(Resp) depending on sigma")
data = pd.DataFrame({'x' : x, 'z' : z})
print(data)
print("The log_likelihood's maximum coordinates are ",data.iloc[data['z'].idxmax()])