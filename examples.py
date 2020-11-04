# -*- coding: utf-8 -*-
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

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
fig = plt.figure()
plt.plot(x,y1, label='N(1, 1.5) : unbiased')
plt.plot(x, y2, label="N(-0.25,1.43) : biased")
plt.axvline(np.mean(x), color='lightgreen', label="mean(x)", linestyle='--')
plt.axvline(1, linestyle='--', label='mu', color='lightblue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.title("Gaussian biased and unbiased")
plt.tight_layout()
fig.savefig('Biased_normal_distri.pdf')

#############################
#An example with real data
#############################

#The data used
#-------------
df = []
df = pd.DataFrame({'Treat' : [0, 1, 0, 1], 'Resp' : [10, 25, 3, 6], 'Ind' : [1, 1, 2, 2]})
print(df)

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

fig2 = plt.figure()
plt.plot(x,z)
plt.title("Log-likelihood with mu=mean(Resp) depending on sigma")
plt.xlabel('Sigma')
plt.ylabel('Log-likelihood')
fig2.savefig('Log_likelihood.pdf')
d = pd.DataFrame({'x' : x, 'z' : z})
print(d)
print("The log_likelihood's maximum coordinates are ",d.iloc[d['z'].idxmax()])

#Linear regression with our data
#---------------------------------------
linear = ols('Resp~Treat',data=df).fit()
print(linear.summary())
#We have the value of alpha = 9 and mu = 6.5

linear_reg = sm.OLS(df.Resp, df.Treat)
linear_reg_fit = linear_reg.fit()
print(linear_reg_fit.summary())
# we have the value of beta2 = 15.5

#We find the same coordinates of the log_likelihood when we use 
#the linear regression and when we use the graph
#So we see that the variance is biased with linear regression
#Besides, We have the fixed value for our alpha and mu : 
#respectively 15.5 =(6.5+9) and 6.5

#Linear mixed regression with our data
#---------------------------------------
#With the REML method
mixed_random_REML = smf.mixedlm("Resp ~ Treat", df, groups = df['Ind'])
mixed_fit_REML = mixed_random_REML.fit()
print(mixed_fit_REML.summary())

#We obtain the Fixed effects : Intercept & Treat 
#(same coefficients and standard error using OLS)
#and the Random effects : Group Var, Residual : Scale
#The log-likelihood has changed 

#with the ML method
mixed_random_ML = smf.mixedlm("Resp ~ Treat", df, groups = df['Ind'])
mixed_fit_ML = mixed_random_ML.fit(reml=False)
print(mixed_fit_ML.summary())
#The fixed effects have the same coefficients
#but the standard deviation for random effects
#and residual standard deviation are differents, the log-likelihood too

#Comparison with our calculations
#-------------------------------------
def f(x):
   sigma = x[0]
   sigmas = x[1]
   beta1 = 6.5
   beta2 = 15.5
   y11 = 3
   y12 = 10
   y21 = 6
   y22 = 25
   return(-(1/2)*np.log(4*sigmas**4*sigma**4 +
                        4*sigmas**2*sigma**6 + sigma**8) -(1/2)*np.log(4/((sigma**2)*(sigma**2+2*sigmas**2)))- 
          (1/2)*(1/((sigma**2)*(sigma**2+2*sigmas**2)))*(((y11-beta1)**2)*(sigma**2+sigmas**2) - 
                                                         2*(y11-beta1)*(y21-beta2)*(sigmas**2) + 
                                                         ((y21-beta2)**2)*(sigma**2+sigmas**2) + 
                                                         ((y12-beta1)**2)*(sigma**2+sigmas**2) - 
                                                         2*(y12-beta1)*(y22-beta2)*(sigmas**2) + 
                                                         ((y22-beta2)**2)*(sigma**2+sigmas**2)))

nb = []
x=1.0
y=1.0
maxi = -10.0
y_liste = np.arange(1, 11, 0.01)
for x in range(1,11):
    for y in y_liste:
        nb = np.append(nb,f([x,y]))
        if maxi < f([x,y]) :
           compteur = x
           compteur2 = y
           maxi = f([x,y])
    

print("The log-likelihood's value found with REML",f([6.00, 8.15]))
print("The log-likelihood's value found with our handmade calculations",maxi)
print("sigma^2 is equal to",compteur, "sigma_s^2 is equal to",compteur2)

#We find the same values for sigma and sigma_s with our calculations and the REML method