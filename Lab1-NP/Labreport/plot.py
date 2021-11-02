import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as con
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy import stats
from uncertainties import ufloat
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


#Definition of Gaussian fit-Function
def gaussian(x, a, b, c):
    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))


###green LED plot 
#wavelength in nm against intensity
a_G, i_G = np.genfromtxt("Data/green-LED-Spectrum.txt",delimiter=",",skip_header=5, unpack = True)
print(a_G)
print(i_G)

#Gaußfit for green spectrum
paramsG, covG = curve_fit(gaussian, a_G,i_G,p0=[1,580,1])
errG = np.sqrt(np.diag(covG))

print('\nGreen-LED-Gauß:')
print('a = ', paramsG[0], r'\pm', errG[0])
print('b = ', paramsG[1], r'\pm', errG[1])
print('c = ', paramsG[2], r'\pm', errG[2])

#Plot of green spectrum

#Value-range in a_G
print("a_G.size")
print(a_G.size)
aa_G=np.linspace(a_G[0],a_G[a_G.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_G, i_G, 'k-', label='Datapoints')
ax.plot(aa_G, gaussian(aa_G, *paramsG), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_G[0],a_G[a_G.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/LED-Green.pdf')
 

#-------------------------------------------------------------------------------------------------------------------

### red LED plot 
#wavelength in nm against intensity
a_R, i_R = np.genfromtxt("Data/red-LED-Spectrum.txt",delimiter=",",skip_header=5, unpack = True)
print(a_R)
print(i_R)

#Gaußfit for red spectrum
paramsR, covR = curve_fit(gaussian, a_R,i_R,p0=[1,580,1])
errR = np.sqrt(np.diag(covR))

print('\nRed-LED-Gauß:')
print('a = ', paramsR[0], r'\pm', errR[0])
print('b = ', paramsR[1], r'\pm', errR[1])
print('c = ', paramsR[2], r'\pm', errR[2])

#Plot of red spectrum

#Value-range in a_R
print("a_R.size")
print(a_R.size)
aa_R=np.linspace(a_R[0],a_R[a_R.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_R, i_R, 'k-', label='Datapoints')
ax.plot(aa_R, gaussian(aa_R, *paramsR), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_R[0],a_R[a_G.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/LED-Red.pdf')

#-------------------------------------------------------------------------------------------------------------------

### UV LED plot 
#wavelength in nm against intensity
a_UV, i_UV = np.genfromtxt("Data/UV-LED-Spectrum.txt",delimiter=",",skip_header=5, unpack = True)
print(a_UV)
print(i_UV)

#Gaußfit for red spectrum
paramsUV, covUV = curve_fit(gaussian, a_UV,i_UV,p0=[1,580,1])
errUV = np.sqrt(np.diag(covUV))

print('\nUV-LED-Gauß:')
print('a = ', paramsUV[0], r'\pm', errUV[0])
print('b = ', paramsUV[1], r'\pm', errUV[1])
print('c = ', paramsUV[2], r'\pm', errUV[2])

#Plot of UV spectrum

#Value-range in a_UV
print("a_UV.size")
print(a_UV.size)
aa_UV=np.linspace(a_UV[0],a_UV[a_UV.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_UV, i_UV, 'k-', label='Datapoints')
ax.plot(aa_UV, gaussian(aa_UV, *paramsUV), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_UV[0],a_UV[a_G.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/LED-UV.pdf')



































# x = np.linspace(0, 10, 1000)
# y = x ** np.sin(x)

# plt.subplot(1, 2, 1)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
# plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
# plt.legend(loc='best')

# plt.subplot(1, 2, 2)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
# plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
# plt.legend(loc='best')

# # in matplotlibrc leider (noch) nicht möglich
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/plot.pdf')
