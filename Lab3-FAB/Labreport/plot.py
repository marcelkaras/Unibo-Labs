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

# ### data plot with gaussian 
# #
# a, i = np.genfromtxt("Data/data.txt",delimiter=",",skip_header=0, unpack = True)

# #Gaußfit
# params, cov = curve_fit(gaussian, a,i,p0=[1,580,1])
# err = np.sqrt(np.diag(cov))

# print('\n:')
# print('a = ', params[0], r'\pm', err[0])
# print('b = ', params[1], r'\pm', err[1])
# print('c = ', params[2], r'\pm', err[2])

# #Plot of 

# #Value-range in x
# aa=np.linspace(a[0],a[a.size-1],1000)

# #plotten

# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# # for axis in ['top','bottom','left','right']:
# #   ax.spines[axis].set_linewidth(0.3)

# # Plot und Labels/ Legende
# ax.plot(a, i, 'k-', label='Datapoints')
# ax.plot(aa, gaussian(aa, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
# ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
# ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
# leg1 = ax.legend(loc='best', fancybox=False,  prop={'size':16}, edgecolor='k')
# leg1.get_frame().set_linewidth(0.3)

# # Einstellung der Achsen
# ax.set_xlim(a[0],a[a.size-1])
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# ax.yaxis.set_minor_locator(AutoMinorLocator())
# ax.tick_params(axis='both', direction='in')
# ax.tick_params(which='major', direction='in', length=7, width=0.3)
# ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# # in matplotlibrc leider (noch) nicht möglich
# #plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('plots/plot.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------

### data plot Sample A dark
#
V_Gate,I_Gate,t,V_Drain,I_Drain,TXX = np.genfromtxt("Data/scan_sampleA_dark_2415171.dat",delimiter="\t",skip_header=2, unpack = True)

#Plot of 

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(V_Gate, I_Gate, 'bx', label='Gate resistance')
#ax.plot(V_Drain, I_Drain, 'b-', label='Drain resistance')
#ax.plot(aa, gaussian(aa, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{V} \:/\: \si{\V}$')
ax.set_ylabel(r'$\text{I} \:/\: \si{\A}$')
leg1 = ax.legend(loc='best', fancybox=False,  prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(V_Gate[0],V_Gate[V_Gate.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/A_dark.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------

### data plot Sample A bright
#
V_Gate,I_Gate,t,V_Drain,I_Drain,TXX = np.genfromtxt("Data/scan_sampleA_light_2479718.dat",delimiter="\t",skip_header=2, unpack = True)

#Plot of 

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(V_Gate, I_Gate, 'bx', label='Gate resistance')
#ax.plot(V_Drain, I_Drain, 'b-', label='Drain resistance')
#ax.plot(aa, gaussian(aa, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{V} \:/\: \si{\V}$')
ax.set_ylabel(r'$\text{I} \:/\: \si{\A}$')
leg1 = ax.legend(loc='best', fancybox=False,  prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(V_Gate[0],V_Gate[V_Gate.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/A_light.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------

### data plot Sample B dark
#
V_Gate,I_Gate,t,V_Drain,I_Drain,TXX = np.genfromtxt("Data/scan_sampleB_dark_2706343.dat",delimiter="\t",skip_header=2, unpack = True)

#Plot of 

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(V_Gate, I_Gate, 'bx', label='Gate resistance')
#ax.plot(V_Drain, I_Drain, 'b-', label='Drain resistance')
#ax.plot(aa, gaussian(aa, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{V} \:/\: \si{\V}$')
ax.set_ylabel(r'$\text{I} \:/\: \si{\A}$')
leg1 = ax.legend(loc='best', fancybox=False,  prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(V_Gate[0],V_Gate[V_Gate.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/B_dark.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------

### data plot Sample B bright
#
V_Gate,I_Gate,t,V_Drain,I_Drain,TXX = np.genfromtxt("Data/scan_sampleB_light_2839421.dat",delimiter="\t",skip_header=2, unpack = True)

#Plot of 

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(V_Gate, I_Gate, 'bx', label='Gate resistance')
#ax.plot(V_Drain, I_Drain, 'b-', label='Drain resistance')
#ax.plot(aa, gaussian(aa, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{V} \:/\: \si{\V}$')
ax.set_ylabel(r'$\text{I} \:/\: \si{\A}$')
leg1 = ax.legend(loc='best', fancybox=False,  prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(V_Gate[0],V_Gate[V_Gate.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/B_light.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------
#time dynamic measurement (light is swiched on and off)

#
V_Gate,I_Gate,t,V_Drain,I_Drain,TXX = np.genfromtxt("Data/osci_sample_2_antisolvent_dynamic_measurement_3235875.dat",delimiter="\t",skip_header=2, unpack = True)

#Plot of 

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(t, I_Gate, 'b-', label='Gate current')
#ax.plot(V_Drain, I_Drain, 'b-', label='Drain resistance')
#ax.plot(aa, gaussian(aa, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{t} \:/\: \si{\s}$')
ax.set_ylabel(r'$\text{I} \:/\: \si{\A}$')
leg1 = ax.legend(loc='best', fancybox=False,  prop={'size':16}, edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
#ax.set_xlim(V_Gate[0],V_Gate[V_Gate.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/dynamic.pdf',bbox_inches = "tight")