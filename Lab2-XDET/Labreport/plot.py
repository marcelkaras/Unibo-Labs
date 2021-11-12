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

### data plot with gaussian for the SOURCE
a_1, i_1 = np.genfromtxt("Data/Spectrum_Source_133Ba_Cs.txt",delimiter="\t",skip_header=6, unpack = True)

#Gaußfit
params_1, cov_1 = curve_fit(gaussian, a_1,i_1,p0=[1,580,1])
err_1 = np.sqrt(np.diag(cov_1))

#peak
peak_pos_1= ufloat(params_1[1],err_1[1])

print('\nSource sprectrum fit:')
print('a = ', params_1[0], r'\pm', err_1[0])
print('b = ', peak_pos_1)
print('c = ', params_1[2], r'\pm', err_1[2])

#Plot of source spectrum

#Value-range in x
aa_1=np.linspace(a_1[0],a_1[a_1.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_1, i_1, 'k-', label='Datapoints')
ax.plot(aa_1, gaussian(aa_1, *params_1), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{bin} $') # \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{counts}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_1[0],a_1[a_1.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/source-spectrum.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------

### data plot with gaussian for the GE
a_2, i_2 = np.genfromtxt("Data/Spectrum_Ge.txt",delimiter="\t",skip_header=6, unpack = True)

#Gaußfit
params_2, cov_2 = curve_fit(gaussian, a_2,i_2,p0=[1,580,1])
err_2 = np.sqrt(np.diag(cov_2))

#peak
peak_pos_2= ufloat(params_2[1],err_2[1])

print('\nGE sprectrum fit:')
print('a = ', params_2[0], r'\pm', err_2[0])
print('b = ', peak_pos_2)
print('c = ', params_2[2], r'\pm', err_2[2])

#Plot of GE spectrum

#Value-range in x
aa_2=np.linspace(a_2[0],a_2[a_2.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_2, i_2, 'k-', label='Datapoints')
ax.plot(aa_2, gaussian(aa_2, *params_2), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{bin} $') # \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{counts}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_2[0],a_2[a_2.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/GE-spectrum.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------
### data plot with gaussian for the FE
a_3, i_3 = np.genfromtxt("Data/Spectrum_Fe.txt",delimiter="\t",skip_header=6, unpack = True)

#Gaußfit
params_3, cov_3 = curve_fit(gaussian, a_3,i_3,p0=[1,580,1])
err_3 = np.sqrt(np.diag(cov_3))

#peak
peak_pos_3= ufloat(params_3[1],err_3[1])

print('\nFE sprectrum fit:')
print('a = ', params_3[0], r'\pm', err_3[0])
print('b = ', peak_pos_3)
print('c = ', params_3[2], r'\pm', err_3[2])

#Plot of Fe spectrum

#Value-range in x
aa_3=np.linspace(a_3[0],a_3[a_3.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_3, i_3, 'k-', label='Datapoints')
ax.plot(aa_3, gaussian(aa_3, *params_3), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{bin} $') # \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{counts}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_3[0],a_3[a_3.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/FE-spectrum.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------
### data plot with gaussian for the Cu
a_4, i_4 = np.genfromtxt("Data/Spectrum_Cu.txt",delimiter="\t",skip_header=6, unpack = True)

#Gaußfit
params_4, cov_4 = curve_fit(gaussian, a_4,i_4,p0=[1,580,1])
err_4 = np.sqrt(np.diag(cov_4))

#peak
peak_pos_4= ufloat(params_4[1],err_4[1])

print('\nCu sprectrum fit:')
print('a = ', params_4[0], r'\pm', err_4[0])
print('b = ', peak_pos_4)
print('c = ', params_4[2], r'\pm', err_4[2])

#Plot of Cu spectrum

#Value-range in x
aa_4=np.linspace(a_4[0],a_4[a_4.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_4, i_4, 'k-', label='Datapoints')
ax.plot(aa_4, gaussian(aa_4, *params_4), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{bin} $') # \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{counts}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_4[0],a_4[a_4.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/Cu-spectrum.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------
### data plot with gaussian for the Ag
a_5, i_5 = np.genfromtxt("Data/Spectrum_Ag.txt",delimiter="\t",skip_header=6, unpack = True)

#Gaußfit
params_5, cov_5 = curve_fit(gaussian, a_5,i_5,p0=[1,580,1])
err_5 = np.sqrt(np.diag(cov_5))
 ####################################################bis hier
#peak
peak_pos_4= ufloat(params_4[1],err_4[1])

print('\nCu sprectrum fit:')
print('a = ', params_4[0], r'\pm', err_4[0])
print('b = ', peak_pos_4)
print('c = ', params_4[2], r'\pm', err_4[2])

#Plot of Cu spectrum

#Value-range in x
aa_4=np.linspace(a_4[0],a_4[a_4.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_4, i_4, 'k-', label='Datapoints')
ax.plot(aa_4, gaussian(aa_4, *params_4), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\text{bin} $') # \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{counts}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_4[0],a_4[a_4.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/Cu-spectrum.pdf',bbox_inches = "tight")

#-------------------------------------------------------------------------------------------------------------


