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


#Definition of Gaussian fit-Function
def gaussian(x, a, b, c):
    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))

#Definition of wavelength (nm) to E in (eV)
def energyConv(x):
    return 1240/x

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
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/LED-Green.pdf',bbox_inches = "tight")
 

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
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/LED-Red.pdf',bbox_inches = "tight")

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
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/LED-UV.pdf',bbox_inches = "tight")

#------------------------------------------------------------------
#Sample spectra with LED light
#------------------------------------------------------------------

#Sample A_D (Sample A with LED)

###Sample A_D plot 
#wavelength in nm against intensity
a_A_D, i_A_D = np.genfromtxt("Data/sampleA-Copy.txt",delimiter=",",skip_header=5, unpack = True)

#Gaußfit for green spectrum
paramsA_D, covA_D = curve_fit(gaussian, a_A_D,i_A_D,p0=[1,470,1])
errA_D = np.sqrt(np.diag(covA_D))

#a-peak in eV instead of nm
a_A_eV=energyConv(paramsA_D[1]) ######### HIER funktioniert es (ohne ufloat) bei der Berechnung von x in Zeile 367
#a_A_eV=ufloat(energyConv(paramsA_D[1]),energyConv(errA_D[1])) ##und hier nicht mehr

print('\nSample_A-LED-Gauß:')
print('a = ', paramsA_D[0], r'\pm', errA_D[0])
print('b = ', paramsA_D[1], r'\pm', errA_D[1])
print('b_eV = ', a_A_eV, r'\pm', errA_D[1])
print('c = ', paramsA_D[2], r'\pm', errA_D[2])

#Plot of Sample A LED spectrum

#Value-range in a_A_D
print("a_A_D.size")
print(a_A_D.size)
aa_A_D=np.linspace(a_A_D[0],a_A_D[a_A_D.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_A_D, i_A_D, 'k-', label='Datapoints')
ax.plot(aa_A_D, gaussian(aa_A_D, *paramsA_D), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_A_D[0],a_A_D[a_A_D.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/Samp_A_D.pdf',bbox_inches = "tight")
 

#-------------------------------------------------------------------------------------------------------------------

#Sample B_D (Sample B with LED)

###Sample B_D plot 
#wavelength in nm against intensity
a_B_D, i_B_D = np.genfromtxt("Data/sampleB-Copy.txt",delimiter=",",skip_header=0, unpack = True)

#Gaußfit for green spectrum
paramsB_D, covB_D = curve_fit(gaussian, a_B_D,i_B_D,p0=[1,550,1])
errB_D = np.sqrt(np.diag(covB_D))

#a-peak in eV instead of nm
a_B_eV=energyConv(paramsB_D[1])
#a_B_eV=ufloat(energyConv(paramsB_D[1]),errB_D[1])

print('\nSample_B-LED-Gauß:')
print('a = ', paramsB_D[0], r'\pm', errB_D[0])
print('b = ', paramsB_D[1], r'\pm', errB_D[1])
print('b_eV = ', a_B_eV, r'\pm', errB_D[1])
print('c = ', paramsB_D[2], r'\pm', errB_D[2])

#Plot of Sample B LED spectrum

#Value-range in a_B_D
print("a_B_D.size")
print(a_B_D.size)
aa_B_D=np.linspace(a_B_D[0],a_B_D[a_B_D.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_B_D, i_B_D, 'k-', label='Datapoints')
ax.plot(aa_B_D, gaussian(aa_B_D, *paramsB_D), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_B_D[0],a_B_D[a_B_D.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/Samp_B_D.pdf',bbox_inches = "tight")
 

#-------------------------------------------------------------------------------------------------------------------
#Sample C_D (Sample C with LED)

###Sample C_D plot 
#wavelength in nm against intensity
a_C_D, i_C_D = np.genfromtxt("Data/sampleC-Copy.txt",delimiter=",",skip_header=0, unpack = True)

#Gaußfit for green spectrum
paramsC_D, covC_D = curve_fit(gaussian, a_C_D,i_C_D,p0=[1,670,1])
errC_D = np.sqrt(np.diag(covC_D))

#a-peak in eV instead of nm
a_C_eV=energyConv(paramsC_D[1])
#a_C_eV=ufloat(energyConv(paramsC_D[1]),errC_D[1])

print('\nSample_C-LED-Gauß:')
print('a = ', paramsC_D[0], r'\pm', errC_D[0])
print('b = ', paramsC_D[1], r'\pm', errC_D[1])
print('b_eV = ', a_C_eV, r'\pm', errC_D[1])
print('c = ', paramsC_D[2], r'\pm', errC_D[2])

#Plot of Sample C LED spectrum

#Value-range in a_C_D
print("a_C_D.size")
print(a_C_D.size)
aa_C_D=np.linspace(a_C_D[0],a_C_D[a_C_D.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_C_D, i_C_D, 'k-', label='Datapoints')
ax.plot(aa_C_D, gaussian(aa_C_D, *paramsC_D), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $C(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_C_D[0],a_C_D[a_C_D.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/Samp_C_D.pdf',bbox_inches = "tight")
 

#-------------------------------------------------------------------------------------------------------------------
#finding the concentracion x

#defining parameters
b=0.24
r = 6e-9 #nm
E_g_CdSe = 1.74 #eV
E_g_CdS  = 2.45 #eV
E_g_ZnS  = 3.45 #eV

m_e_CdSe = 0.13
m_e_CdS  = 0.21

m_h_CdSe = 0.45
m_h_CdS  = 0.8

#a_A_eV sind die photonenenergien hv in eV

#defining the function
def photonE(x,E_photon):
    return x * E_g_CdS + (1-x) * E_g_CdSe - b * x * (1-x) + const.h**2/(8 * const.e * r**2)*(1/( x * m_e_CdS + (1-x) * m_e_CdSe )+1/(x * m_h_CdS + (1-x) * m_h_CdSe)) -E_photon

solA = opt.fsolve(photonE,0.5,args=a_A_eV)
#solA = opt.fmin_slsqp(photonE,0.5,args=a_A_eV)
print('Solutions for x')
print('Sample A concentration',solA)

solB = opt.fsolve(photonE,0.5,args=a_B_eV)
print('Sample B concentration',solB)

solC = opt.fsolve(photonE,0.5,args=a_C_eV)
print('Sample C concentration',solC)
print('h :', const.h)
print('e :',const.e)





















#------------------------------------------------------------------
#Sample spectra with laser light
#------------------------------------------------------------------

#Sample A_D (Sample A with Laser)

###Sample A_D plot 
#wavelength in nm against intensity
a_A_D, i_A_D = np.genfromtxt("Data/sampleA.txt",delimiter=",",skip_header=6, unpack = True)

#Gaußfit for spectrum
paramsA_D, covA_D = curve_fit(gaussian, a_A_D,i_A_D,p0=[1,470,1])
errA_D = np.sqrt(np.diag(covA_D))

#a-peak in eV instead of nm
a_A_eV_L=energyConv(paramsA_D[1]) ######### HIER funktioniert es (ohne ufloat) bei der Berechnung von x in Zeile 367
#a_A_eV=ufloat(energyConv(paramsA_D[1]),energyConv(errA_D[1])) ##und hier nicht mehr

print('\nSample_A-UV-Gauß:')
print('a = ', paramsA_D[0], r'\pm', errA_D[0])
print('b = ', paramsA_D[1], r'\pm', errA_D[1])
print('b_eV = ', a_A_eV_L, r'\pm', energyConv(errA_D[1]))
print('c = ', paramsA_D[2], r'\pm', errA_D[2])

#Plot of Sample A UV spectrum

#Value-range in a_A_D
print("a_A_D.size")
print(a_A_D.size)
aa_A_D=np.linspace(a_A_D[0],a_A_D[a_A_D.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_A_D, i_A_D, 'k-', label='Datapoints')
ax.plot(aa_A_D, gaussian(aa_A_D, *paramsA_D), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_A_D[0],a_A_D[a_A_D.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/Samp_A_D_UV.pdf',bbox_inches = "tight")
 

#-------------------------------------------------------------------------------------------------------------------

#Sample B_D (Sample B with UV)

###Sample B_D plot 
#wavelength in nm against intensity
a_B_D, i_B_D = np.genfromtxt("Data/sampleB.txt",delimiter=",",skip_header=6, unpack = True)

#Gaußfit for spectrum
paramsB_D, covB_D = curve_fit(gaussian, a_B_D,i_B_D,p0=[1,550,1])
errB_D = np.sqrt(np.diag(covB_D))

#a-peak in eV instead of nm
a_B_eV_L=energyConv(paramsB_D[1])
#a_B_eV=ufloat(energyConv(paramsB_D[1]),energyConv(errB_D[1]))

print('\nSample_B-UV-Gauß:')
print('a = ', paramsB_D[0], r'\pm', errB_D[0])
print('b = ', paramsB_D[1], r'\pm', errB_D[1])
print('b_eV = ', a_B_eV_L, r'\pm', energyConv(errB_D[1]))
print('c = ', paramsB_D[2], r'\pm', errB_D[2])

#Plot of Sample B UV spectrum

#Value-range in a_B_D
print("a_B_D.size")
print(a_B_D.size)
aa_B_D=np.linspace(a_B_D[0],a_B_D[a_B_D.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_B_D, i_B_D, 'k-', label='Datapoints')
ax.plot(aa_B_D, gaussian(aa_B_D, *paramsB_D), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_B_D[0],a_B_D[a_B_D.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/Samp_B_D_UV.pdf',bbox_inches = "tight")
 

#-------------------------------------------------------------------------------------------------------------------
#Sample C_D (Sample C with UV)

###Sample C_D plot 
#wavelength in nm against intensity
a_C_D, i_C_D = np.genfromtxt("Data/sampleC.txt",delimiter=",",skip_header=6, unpack = True)

#Gaußfit for spectrum
paramsC_D, covC_D = curve_fit(gaussian, a_C_D,i_C_D,p0=[1,670,1])
errC_D = np.sqrt(np.diag(covC_D))

#a-peak in eV instead of nm
a_C_eV_L=energyConv(paramsC_D[1])
#a_C_eV=ufloat(energyConv(paramsC_D[1]),energyConv(errC_D[1]))

print('\nSample_C-UV-Gauß:')
print('a = ', paramsC_D[0], r'\pm', errC_D[0])
print('b = ', paramsC_D[1], r'\pm', errC_D[1])
print('b_eV = ', a_C_eV_L, r'\pm', energyConv(errC_D[1]))
print('c = ', paramsC_D[2], r'\pm', errC_D[2])

#Plot of Sample C UV spectrum

#Value-range in a_C_D
print("a_C_D.size")
print(a_C_D.size)
aa_C_D=np.linspace(a_C_D[0],a_C_D[a_C_D.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_C_D, i_C_D, 'k-', label='Datapoints')
ax.plot(aa_C_D, gaussian(aa_C_D, *paramsC_D), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $C(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_C_D[0],a_C_D[a_C_D.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/Samp_C_D_UV.pdf',bbox_inches = "tight")
 

#-------------------------------------------------------------------------------------------------------------------
#finding the concentracion x

#defining parameters
b=0.24
r = 6e-9 #nm
E_g_CdSe = 1.74 #eV
E_g_CdS  = 2.45 #eV
E_g_ZnS  = 3.45 #eV

m_e_CdSe = 0.13
m_e_CdS  = 0.21

m_h_CdSe = 0.45
m_h_CdS  = 0.8

#a_A_eV sind die photonenenergien hv in eV

#defining the function
def photonE(x,E_photon):
    return x * E_g_CdS + (1-x) * E_g_CdSe - b * x * (1-x) + const.h**2/(8 * const.e * r**2)*(1/( x * m_e_CdS + (1-x) * m_e_CdSe )+1/(x * m_h_CdS + (1-x) * m_h_CdSe)) -E_photon

solA = opt.fsolve(photonE,0.5,args=a_A_eV_L)
#solA = opt.fmin_slsqp(photonE,0.5,args=a_A_eV)
print('Solutions for x')
print('Sample A concentration',solA)

solB = opt.fsolve(photonE,0.5,args=a_B_eV_L)
print('Sample B concentration',solB)

solC = opt.fsolve(photonE,0.5,args=a_C_eV_L)
print('Sample C concentration',solC)




#------------------------------------------------------------------
#Perovskite spectra with laser light
#------------------------------------------------------------------

#Sample P 

###Sample P plot 
#wavelength in nm against intensity
a_P, i_P = np.genfromtxt("Data/perovskite-sample.txt",delimiter=",",skip_header=6, unpack = True)

#Gaußfit for spectrum
paramsP, covP = curve_fit(gaussian, a_P,i_P,p0=[1,570,1])
errP = np.sqrt(np.diag(covP))

#a-peak in eV instead of nm
a_P_eV=energyConv(paramsP[1]) ######### HIER funktioniert es (ohne ufloat) bei der Berechnung von x in Zeile 367
#a_P_eV=ufloat(energyConv(paramsP[1]),energyConv(errP[1])) ##und hier nicht mehr

print('\nSample_P_UV-Gauß:')
print('a = ', paramsP[0], r'\pm', errP[0])
print('b = ', paramsP[1], r'\pm', errP[1])
print('b_eV = ', a_P_eV, r'\pm', energyConv(errP[1]))
print('c = ', paramsP[2], r'\pm', errP[2])

#Plot of Sample P UV spectrum

#Value-range in a_P
print("a_P.size")
print(a_P.size)
aa_P=np.linspace(a_P[0],a_P[a_P.size-1],1000)

#plotten

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(a_P, i_P, 'k-', label='Datapoints')
ax.plot(aa_P, gaussian(aa_P, *paramsP), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$\lambda \:/\: \si{\nano\meter}$')
ax.set_ylabel(r'$\text{Intensity} \:/\: \text{a.u.}$')
leg1 = ax.legend(loc='best', fancybox=False, fontsize='small', edgecolor='k')
leg1.get_frame().set_linewidth(0.3)

# Einstellung der Achsen
ax.set_xlim(a_A_D[0],a_A_D[a_A_D.size-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', direction='in')
ax.tick_params(which='major', direction='in', length=7, width=0.3)
ax.tick_params(which='minor', direction='in', length=4, width=0.3)

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plots/Samp_P_UV.pdf',bbox_inches = "tight")

##Discussion

#deviations between LED and Laser

d_A = (a_A_eV - a_A_eV_L) / a_A_eV
d_B = (a_B_eV-a_B_eV_L) / a_B_eV
d_C = (a_C_eV-a_C_eV_L) / a_C_eV

print("Discussion")
print("delta A:", d_A)
print("delta B:", d_B)
print("delta C:", d_C)