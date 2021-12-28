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
E_g = ufloat(-params4[1]*2*const.k/const.e,err4[1]*2*const.k/const.e)
#logarithmic linear fit for germanium
params5, cov5 = curve_fit(metals,np.log(T[:92]),np.log(R1[:92]),p0=[1,1])
err5 = np.sqrt(np.diag(cov5))
print('\nLog-lin-Ge:')
print('A = ', params5[0], r'\pm', err5[0])
print('B = ', params5[1], r'\pm', err5[1])
alpha=ufloat(params5[1],err5[1])

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
print('\n Discussion')
#error in alpha
err_alpha = (1.5-alpha)/1.5
print('err_alpha',err_alpha)
#error in E_g
err_E = (0.67-E_g)/0.67
print('err_E',err_E)







#-------------------------------------------------------------------------------------------------------------
#####Second part of the exoeriment
#-------------------------------------------------------------------------------------------------------------
#samples A,B,C,D
# length (cm), diameter(mm)
scale=np.array([[13.5,1],[18.7,0.6],[23.9,0.6],[13.2,0.23]])
scale[:,0]=scale[:,0]*10**(-2) #length
scale[:,1]=scale[:,1]*10**(-3) #diameter

#2 wire resistivity(ohm)
twireR=np.array([[0.3554,0.034,0.034,0.032], #sample A
[1.0452,0.072,0.062,0.051], #B
[0.3814,0.036,0.037,0.036], #C
[0.43,0.034,0.062,0.036]])  #D

twireL=np.array([[13.5,7.6,4.6,2.3], #A
[18.7,9.3,7,3.9],   #B
[23.9,3.2,7.1,12.9],    #C
[13.2,1.6,3.7,6.5]])    #D
twireL[:,:]=twireL[:,:]*10**(-2)

#4 wire resitivity
fwireR=np.array([[0.067,0.031,0.043,0.078],
[0.754,0.18,0.311,0.429],
[0.127,0.161,0.049,0.039],
[0.148,0.075,0.049,0.033]])

fwireL=np.array([[13.5,6.5,4.7,2.3],
[18.7,3.1,6.7,9.4],
[23.9,12.8,7.5,2.2],
[13.2,7.1,4.8,1.9]])
fwireL[:,:]=fwireL[:,:]*10**(-2)

#define formula for resitance : to find the fitting resistivity as a fit parameter
def RA(L,rho):
    return rho*L/(const.pi*(scale[0][1]/2)**2)
def RB(L,rho):
    return rho*L/(const.pi*(scale[1][1]/2)**2)
def RC(L,rho):
    return rho*L/(const.pi*(scale[2][1]/2)**2)
def RD(L,rho):
    return rho*L/(const.pi*(scale[3][1]/2)**2)
    
#2 wires
#resistivity fit for sample A
params2A, cov2A = curve_fit(RA,twireL[0],twireR[0])
err2A = np.sqrt(np.diag(cov2A))
print('twireL[0]',twireL[0])
print('\nA:')
print('rho2A = ', params2A[0], r'\pm', err2A[0])
rho2A=ufloat(params2A[0],err2A[0])
#resistivity fit for sample B
params2B, cov2B = curve_fit(RB,twireL[1],twireR[1])
err2B = np.sqrt(np.diag(cov2B))

print('\nB:')
print('rho2B = ', params2B[0], r'\pm', err2B[0])
rho2B=ufloat(params2B[0],err2B[0])
#resistivity fit for sample C
params2C, cov2C = curve_fit(RC,twireL[2],twireR[2])
err2C = np.sqrt(np.diag(cov2C))

print('\nC:')
print('rho2C = ', params2C[0], r'\pm', err2C[0])
rho2C=ufloat(params2C[0],err2C[0])
#resistivity fit for sample D
params2D, cov2D = curve_fit(RD,twireL[3],twireR[3])
err2D = np.sqrt(np.diag(cov2D))

print('\nD:')
print('rho2D = ', params2D[0], r'\pm', err2D[0])
rho2D=ufloat(params2D[0],err2D[0])
#4 wires###############################
#resistivity fit for sample A
params4A, cov4A = curve_fit(RA,fwireL[0],fwireR[0])
err4A = np.sqrt(np.diag(cov4A))

print('\nA:')
print('rho4A = ', params4A[0], r'\pm', err4A[0])
rho4A=ufloat(params4A[0],err4A[0])
#resistivity fit for sample B
params4B, cov4B = curve_fit(RB,fwireL[1],fwireR[1])
err4B = np.sqrt(np.diag(cov4B))

print('\nB:')
print('rho4B = ', params4B[0], r'\pm', err4B[0])
rho4B=ufloat(params4B[0],err4B[0])
#resistivity fit for sample C
params4C, cov4C = curve_fit(RC,fwireL[2],fwireR[2])
err4C = np.sqrt(np.diag(cov4C))

print('\nC:')
print('rho4C = ', params4C[0], r'\pm', err4C[0])
rho4C=ufloat(params4C[0],err4C[0])
#resistivity fit for sample D
params4D, cov4D = curve_fit(RD,fwireL[3],fwireR[3])
err4D = np.sqrt(np.diag(cov4D))

print('\nD:')
print('rho4D = ', params4D[0], r'\pm', err4D[0])
rho4D=ufloat(params4D[0],err4D[0])

##Disscussion of wire-part 
#relative discrepancy

RRA=(rho2A-rho4A)/rho4A
RRB=(rho2B-rho4B)/rho4B
RRC=(rho2C-rho4C)/rho4C
RRD=(rho2D-rho4D)/rho4D

print('\nErrors of resistivity 2 vs4 wires:')
print('error A: ',RRA)
print('error B: ',RRB)
print('error C: ',RRC)
print('error D: ',RRD)

#plotten resistivity A

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(twireL[0],twireR[0], 'k.', label='Sample A')
ax.plot(twireL[0], RA(twireL[0],*params2A), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$L \:/\: \si{\meter}$')
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
plt.savefig('plots/resitvity-check-A2.pdf',bbox_inches = "tight")

#plotten resistivity B

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(twireL[1],twireR[1], 'k.', label='Sample B')
ax.plot(twireL[1], RB(twireL[1],*params2B), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$L \:/\: \si{\meter}$')
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
plt.savefig('plots/resitvity-check-B2.pdf',bbox_inches = "tight")

#plotten resistivity C

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(twireL[2],twireR[2], 'k.', label='Sample C')
ax.plot(twireL[2], RC(twireL[2],*params2C), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$L \:/\: \si{\meter}$')
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
plt.savefig('plots/resitvity-check-C2.pdf',bbox_inches = "tight")

#plotten resistivity D

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(twireL[3],twireR[3], 'k.', label='Sample D')
ax.plot(twireL[3], RD(twireL[3],*params2D), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$L \:/\: \si{\meter}$')
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
plt.savefig('plots/resitvity-check-D2.pdf',bbox_inches = "tight")

#-------------------4 wires resistivity fit

#plotten resistivity A

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(fwireL[0],fwireR[0], 'k.', label='Sample A')
ax.plot(fwireL[0], RA(fwireL[0],*params4A), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$L \:/\: \si{\meter}$')
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
plt.savefig('plots/resitvity-check-A4.pdf',bbox_inches = "tight")

#plotten resistivity B

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(fwireL[1],fwireR[1], 'k.', label='Sample B')
ax.plot(fwireL[1], RB(fwireL[1],*params4B), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$L \:/\: \si{\meter}$')
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
plt.savefig('plots/resitvity-check-B4.pdf',bbox_inches = "tight")

#plotten resistivity C

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(fwireL[2],fwireR[2], 'k.', label='Sample C')
ax.plot(fwireL[2], RC(fwireL[2],*params4C), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$L \:/\: \si{\meter}$')
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
plt.savefig('plots/resitvity-check-C4.pdf',bbox_inches = "tight")

#plotten resistivity D

fig = plt.figure()
ax = fig.add_axes([0.1, 0.13, 0.8, 0.83])
# for axis in ['top','bottom','left','right']:
#   ax.spines[axis].set_linewidth(0.3)

# Plot und Labels/ Legende
ax.plot(fwireL[3],fwireR[3], 'k.', label='Sample D')
ax.plot(fwireL[3], RD(fwireL[3],*params4D), 'r-', label='Linear fit')
#ax.plot(TT, gaussian(TT, *params), '-', color='#EC0000', label='Gaussian fit') #label=r'Fit $B(z) = az^4 + bz^3 + cz^2 + dz + e$')
ax.set_xlabel(r'$L \:/\: \si{\meter}$')
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
plt.savefig('plots/resitvity-check-D4.pdf',bbox_inches = "tight")

#####Discussion of second part
print('\nDiscussion part two')
B_lit=144e-8
C_lit=22e-8
D_lit=1.68e-8

err_B=(rho4B-B_lit)/B_lit
print('err_B: ',err_B)

err_C=(rho4C-C_lit)/C_lit
print('err_C: ',err_C)

err_D=(rho4D-D_lit)/D_lit
print('err_D: ',err_D)