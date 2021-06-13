# expanding-photosphere-method
# Author: Darshana Mehta (May 2021)
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.integrate import quad
from scipy import interpolate
from scipy import stats
import math
#coefficients from Hamuy et al. (2001)
c_B = np.array([-45.144,7.159,-4.301,2.639,-0.811,0.098])
c_V = np.array([-44.766,6.793,-4.523,2.695,-0.809,0.096])
c_R = np.array([-44.597,6.628,-4.693,2.770,-0.831,0.099])
c_I = np.array([-44.345,6.347,-4.732,2.739,-0.811,0.096])

'''
#total extinction
AB = np.mean([0.55068963,0.54900807,0.54373798,0.54352304])
AV = np.mean([0.41738099,0.41800779,0.40954067,0.42177598])
AR = np.mean([0.34789336,0.34462009,0.32577852,0.33126384])
AI = np.mean([0.24823369,0.25261019,0.2297512 ,0.22723341])
'''
#host extinction
AB = np.mean([0.49412497,0.49261614,0.48788737,0.4876945])
AV = np.mean([0.37450927,0.37507168,0.36747427,0.37845282])
AR = np.mean([0.31215913,0.30922208,0.29231584,0.29723773])
AI = np.mean([0.22273611,0.22666307,0.20615207,0.20389289])

#coefficients from Hamuy et al. (2001)
aBV = np.array([0.7557,-0.8997,0.5199])
aBVI = np.array([0.7336,-0.6942,0.3740])
aVI = np.array([0.7013,-0.5304,0.2646])

#importing the photometric data of the target SN
fname='interpolated_lcs_SN 2013he_BVRI.txt'
phase = np.genfromtxt(fname, delimiter=None, comments="#", dtype=None, usecols=(0))
new_x = phase*3600*24
m_B = np.genfromtxt(fname, delimiter=None, comments="#", dtype=None, usecols=(1))
m_Berr = np.genfromtxt(fname, delimiter=None, comments="#", dtype=None, usecols=(2))
m_V = np.genfromtxt(fname, delimiter=None, comments="#", dtype=None, usecols=(3))
m_Verr = np.genfromtxt(fname, delimiter=None, comments="#", dtype=None, usecols=(4))
m_I = np.genfromtxt(fname, delimiter=None, comments="#", dtype=None, usecols=(7))
m_Ierr = np.genfromtxt(fname, delimiter=None, comments="#", dtype=None, usecols=(8))

#interpolating the vphs of the available spectra, to match the epochs of the photometric data
vpha = 1000*np.array([8948.775972022,7658.270932607,7300.884955752,5919.907138712,5513.639001741])
vphb = 1000*np.array([9072.20736474,7658.270932607,7300.884955752,5861.86883343,5455.60069646])
phase1 = 3600*24*np.array([2.05,3.00,4.08,22.04,31.04])
vph = np.zeros(len(phase1))
verr = np.zeros(len(phase1))

for i in range (len(phase1)):
  vph[i] = np.mean([vpha[i],vphb[i]])
  verr[i] = np.std([vpha[i],vphb[i]])

new_y = interpolate.UnivariateSpline(phase1,vph)(new_x)
v_err = interpolate.UnivariateSpline(phase1,verr)(new_x)

#defining the blackbody fucntions for each of the BVRI bands 
def b_B(T):
  s=0
  for i in range(0,6):
    s+= c_B[i]*((1e4/T)**i) 
  return s

def b_V(T):
  s=0
  for i in range(0,6):
    s+= c_V[i]*((1e4/T)**i) 
  return s

def b_R(T):
  s=0
  for i in range(0,6):
    s+= c_R[i]*((1e4/T)**i) 
  return s

def b_I(T):
  s=0
  for i in range(0,6):
    s+= c_I[i]*((1e4/T)**i) 
  return s

#defining the fucntion to be minimized
def eta(x,mB,mV,mI):
  zt,T=x
  B = (mB + (5*np.log10(zt)) - AB - b_B(T))**2
  V = (mV + (5*np.log10(zt)) - AV - b_V(T))**2
  I = (mI + (5*np.log10(zt)) - AI - b_I(T))**2
  chi_sq = B + V + I 
  return chi_sq

#defining the dilution factor as a function of Temperature
def z(T,a):
  s=0
  for i in range (len(a)):
    s += a[i]*((1e4/T)**i)
  return s  

#defining the output parameters
zt = np.zeros(len(phase))
zt_err = np.zeros(len(phase))
T = np.zeros(len(phase)) 
T_err = np.zeros(len(phase))
the = np.zeros(len(phase))
the_err = np.zeros(len(phase))
ze = np.zeros(len(phase))
ze_err = np.zeros(len(phase))
vph_err = np.zeros(len(phase))
ratio = np.zeros(len(phase))
ratio_err = np.zeros(len(phase))


for i in range (len(phase)):
  init_guess=[1e-11,8000]
  #randomizing the magnitude values for error estimation
  mB = np.random.normal(m_B[i],m_Berr[i],1000)
  mV = np.random.normal(m_V[i],m_Verr[i],1000)
  mI = np.random.normal(m_I[i],m_Ierr[i],1000)

  zeta_theta = np.zeros(len(mB))
  temp = np.zeros(len(mB))
  zeta = np.zeros(len(mB))
  theta = np.zeros(len(mB))
  
  #minimizing the "eta" function and estimating the values of zeta, theta and temperature
  for j in range (len(mB)):
    res = minimize(eta,init_guess,args=(mB[j],mV[j],mI[j]),method='nelder-mead')
    zeta_theta[j] = res.x[0]
    temp[j] = res.x[1]
    zeta[j] = z(temp[j],aBVI)
    theta[j] = zeta_theta[j]/zeta[j]
  
  #calculating and printing the output 
  zt[i] = np.mean(zeta_theta)
  zt_err[i] = np.std(zeta_theta)
  T[i] = np.mean(temp)
  T_err[i] = np.std(temp)
  ze[i] = np.mean(zeta)
  ze_err[i] = np.std(zeta)
  the[i] = np.mean(theta)
  the_err[i] = np.std(theta)
   

  ratio[i] = the[i]/(new_y[i]*3.24)
  ratio_err[i] = ( ratio[i]*np.sqrt( (the_err[i]/the[i])**2 + (v_err[i]/new_y[i])**2 ) ) / 3.24
  print (phase[i],zt[i],zt_err[i],T[i],T_err[i],ze[i],ze_err[i],the[i],the_err[i],new_y[i]/1000,v_err[i]/1000,ratio[i],ratio_err[i])
  
#fitting the theta/vphs v/s days since discovery with a line, in order to estimate the distance(slope) and the explosion epoch (y-intercept)
res = np.polyfit(ratio,new_x,1)
print ("Distance estimate:",res[0]/(1e6*3.086e16),"Explosion Epoch:",res[1]/(3600*24))

plt.plot (ratio, ((ratio*res[0])+(res[1]))/(3600*24))
plt.errorbar(ratio,new_x/(3600*24),xerr=ratio_err,fmt='o',color='black',ecolor='gray')
plt.xlabel (r'$\theta$/$v_{ph}$ (rad $m^{-1}$ s)')
plt.ylabel ('Days since Discovery')
plt.show()
