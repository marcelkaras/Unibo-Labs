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
#-------------------------------------------------------------------------------------------------------------------

#### how to use solve? -> finding f(x)=0
# solA = opt.fsolve(photonE,0.5,args=a_A_eV)

#--------------------------------------------------------------------------------------------------------------

### data plot with gaussian 
#
t,U1,U2,U3,U4,R1,R2,R3,T = np.genfromtxt("Data/RT-data.txt",delimiter="\t",skip_header=7, unpack = True)

# #Gaußfit
# params, cov = curve_fit(gaussian, T,R1,p0=[1,580,1])
# err = np.sqrt(np.diag(cov))

# print('\n:')
# print('a = ', params[0], r'\pm', err[0])
# print('b = ', params[1], r'\pm', err[1])
# print('c = ', params[2], r'\pm', err[2])

#Plot of 

#Value-range in x
# TT=np.linspace(T[0],T[a.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(T, R1, 'k-', label='Datapoints')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\T \:/\: \si{\degree\celsius}$')
ax.set_ylabel(r'$\text{R} \:/\: \si{\ohm}$')
leg1 = ax.legend(loc='best', fancybox=False, prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(T[0],T[T.size-1])
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
