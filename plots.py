# -*- coding: UTF8 -*-

from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import learnPoly as lp
from scipy import interpolate
from sklearn.linear_model import LinearRegression

P = 101320 # in Pa
R = 287.05 # in J/kgK
T = 298 # in K
rho = P/(R*T) # in kg/m3, change as needed
chord = 17.8e-2 # in m, change as needed
span = 57e-2 # in m, change as needed

# import the data
data = pd.read_csv('EXPdata0012.csv')

# define variables
datanp = pd.DataFrame(data).to_numpy()
alpha = pd.DataFrame(data['alpha']).to_numpy()
lift = pd.DataFrame(data['L']).to_numpy() #data['L'] # in N
drag = -1*pd.DataFrame(data['D']).to_numpy() #data['D'] # in N
velocity = pd.DataFrame(data['v']).to_numpy() # in m/s
nData,cols = shape(lift)

# dynamic pressure computation
S = chord
v_inf = mean(velocity) # in m/s
q_inf = 0.5*rho*(power(v_inf,2)) # dynamic pressure in Pa

# aerodynamic coefficients
cl = divide(lift,S*q_inf)
cd = divide(drag,S*q_inf)

# sklearn regression fit ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
reg = LinearRegression(fit_intercept=True).fit(alpha[:-1],cl[:-1])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# PLOTTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cl_0_reg = reg.intercept_ # y intercept
cl_0 = cl[where(alpha==0)]
alpha_stall = alpha[where(cl==max(cl))]
a_o = reg.coef_ # slope obtained from regression
alpha_L0 = -cl_0_reg/a_o
print('cl_max = %f \ncl_0 = %f \nalpha_stall = %f \na_o = %f \nalpha_L=0 = %f' 
	%(max(cl),cl_0,alpha_stall,a_o,alpha_L0))
print('\ncl_0 from regression = %f' %(cl_0_reg))

maxeff = max(divide(cl,cd))
alphaMaxEff = alpha[where(divide(cl,cd)==maxeff)]

print('Max aerodynamic efficiency = %f at %f degrees' %(maxeff,alphaMaxEff))




# plot parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color = 'y'
axwidth = 1

# cl vs alpha
plt.figure(1)
plt.scatter(alpha,cl,c=color)
plt.plot(alpha[:-1],a_o*alpha[:-1]+cl_0_reg,'g-',linewidth=1.2)
plt.legend(('Linear fit','Exp. data'))
plt.plot(alpha_stall,max(cl),'ro'); plt.plot(0,cl_0,'ro'); plt.plot(alpha_L0,0,'ro')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$C_l$')
plt.axvline(x=0,color='k',linewidth=axwidth)
plt.axhline(y=0,color='k',linewidth=axwidth)
plt.grid(True,linestyle='--')
# cd vs alpha
plt.figure(2)
plt.scatter(alpha,cd,c=color)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$C_d$')
plt.axvline(x=0,color='k',linewidth=axwidth)
plt.axhline(y=0,color='k',linewidth=axwidth)
plt.grid(True,linestyle='--')
#plt.gca().invert_yaxis()
#cl vs cd
plt.figure(3)
plt.scatter(cd,cl,c=color)
plt.xlabel(r'$C_d$')
plt.ylabel(r'$C_l$')
plt.axvline(x=0,color='k',linewidth=axwidth)
plt.axhline(y=0,color='k',linewidth=axwidth)
plt.grid(True,linestyle='--')
#plt.gca().invert_yaxis()
# cl/cd vs alpha
plt.figure(4)
plt.scatter(alpha,divide(cl,cd),c=color)
plt.plot(alphaMaxEff,maxeff,'ro'); plt.legend(('Max. Aero. Eff.','Exp. data'),loc='upper left')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\frac{C_l}{C_d}$')
plt.axvline(x=0,color='k',linewidth=axwidth)
plt.axhline(y=0,color='k',linewidth=axwidth)
plt.grid(True,linestyle='--')
#plt.gca().invert_yaxis()

plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~