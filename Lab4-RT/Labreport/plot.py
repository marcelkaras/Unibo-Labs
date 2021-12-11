import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy import stats
from uncertainties import ufloat
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

#--------------------------------------------------------------------------------------------------------
#Definition of Gaussian fit-Function
def gaussian(x, a, b, c):
    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))

#Definition of wavelength (nm) to E in (eV)
def energyConv(x):
    return 1240/x

#Definition of linar function
def metals(x,R_0,b):
    return R_0 + x*b

#Definition exponential function
def expo(x,A,B,C):
    return A*np.exp(-B/x)+C
#-------------------------------------------------------------------------------------------------------------------

#### how to use solve? -> finding f(x)=0
# solA = opt.fsolve(photonE,0.5,args=a_A_eV)

#--------------------------------------------------------------------------------------------------------------

### data plot with gaussian 
#
t,U1,U2,U3,U4,R1,R2,R3,T = np.genfromtxt("Data/RT-data.txt",delimiter="\t",skip_header=337, unpack = True)
#Resistance at T=0°C (t= 760)
R1_0=R1[76]
R2_0=R2[76]
R3_0=R3[76]
# #Metal fit
#Germanium
params1, cov1 = curve_fit(metals,T[:92],R1[:92],p0=[-10,1])
err1 = np.sqrt(np.diag(cov1))

print('\nGe:')
print('R0_1 = ', params1[0], r'\pm', err1[0])
print('b1 = ', params1[1], r'\pm', err1[1])

#Copper
params2, cov2 = curve_fit(metals,T,R2,p0=[-10,1])
err2 = np.sqrt(np.diag(cov2))

print('\nCopper:')
print('R0_2 = ', params2[0], r'\pm', err2[0])
print('b2 = ', params2[1], r'\pm', err2[1])

#Nickel
params3, cov3 = curve_fit(metals,T,R3,p0=[-10,1])
err3 = np.sqrt(np.diag(cov3))

print('\nNickel:')
print('R0_3 = ', params3[0], r'\pm', err3[0])
print('b3 = ', params3[1], r'\pm', err3[1])

#exponential fit for germanium
params4, cov4 = curve_fit(expo,T[120:],R1[120:],p0=[1,1,1])
err4 = np.sqrt(np.diag(cov4))
print('Size of T:',(T.size-110)*1000/T.size)
print('Size of T:',(T.size-92)*1000/T.size)
print('T(120)',T[120])

print('\nExp-Ge:')
print('A = ', params4[0], r'\pm', err4[0])
print('B = ', params4[1], r'\pm', err4[1])
print('B*2k= ',params4[1]*2*const.k/const.e, r'\pm', err4[1]*2*const.k/const.e)
print('k',const.Boltzmann/const.e)
print('C = ', params4[2], r'\pm', err4[2])

#logarithmic linear fit for germanium
params5, cov5 = curve_fit(metals,np.log(T[:92]),np.log(R1[:92]),p0=[1,1])
err5 = np.sqrt(np.diag(cov5))
print('\nLog-lin-Ge:')
print('A = ', params5[0], r'\pm', err5[0])
print('B = ', params5[1], r'\pm', err5[1])


#Plot of 

#Value-range in x
TT=np.linspace(T[0],T[T.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(T, R2/R2_0, 'g.', label='Copper')
ax.plot(T, R3/R3_0, 'b.', label='Nickel')
ax.plot(T, R1/R1_0, 'k.', label='Germanium')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$T \:/\: \si{\kelvin}$')
ax.set_ylabel(r'$R/R_0$')
leg1 = ax.legend(loc='best', fancybox=False, prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(T[0],T[T.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/compare.pdf',bbox_inches = "tight")


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#plotten germanium

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(T, R1, 'k.', label='Germanium')
ax.plot(TT[:420], metals(TT[:420],*params1), 'g-', label='Linear fit')
ax.plot(T[120:], expo(T[120:],*params4), 'r-', label='Exponential fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$T \:/\: \si{\kelvin}$')
ax.set_ylabel(r'$R \:/\: \si{\ohm}$')
leg1 = ax.legend(loc='best', fancybox=False, prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(T[0],T[T.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/R1.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#plotten Copper

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(T, R2, 'k.', label='Copper')
ax.plot(TT, metals(TT,*params2), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$T \:/\: \si{\kelvin}$')
ax.set_ylabel(r'$R \:/\: \si{\ohm}$')
leg1 = ax.legend(loc='best', fancybox=False, prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(T[0],T[T.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/R2.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#plotten Nickel

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(T, R3, 'k.', label='Nickel')
ax.plot(TT, metals(TT,*params3), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$T \:/\: \si{\kelvin}$')
ax.set_ylabel(r'$R \:/\: \si{\ohm}$')
leg1 = ax.legend(loc='best', fancybox=False, prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(T[0],T[T.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/R3.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------
#double logarithmic plot for germanium low temperature
#-------------------------------------------------------------------------------------------------------------
#plotten germanium

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(np.log(T), np.log(R1), 'k.', label='Germanium')
ax.plot(np.log(T[:92]), metals(np.log(T[:92]),*params5), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$ln(T) \:/\: a.u.$')
ax.set_ylabel(r'$ln(R) \:/\: a.u.$')
leg1 = ax.legend(loc='best', fancybox=False, prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(T[0],T[T.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/R1-log.pdf',bbox_inches = "tight")


#-------------------------------------------------------------------------------------------------------------
#Discussion











#-------------------------------------------------------------------------------------------------------------
#####Second part of the exoeriment
#-------------------------------------------------------------------------------------------------------------

