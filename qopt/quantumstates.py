# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:39:17 2010

@author: Jonas
"""


import scipy as sp
import numpy as np
from numpy import pi, exp, log, sqrt, tan, sin, cos, \
        tanh, sinh, cosh, arccos, angle
from scipy.special import factorial
#from scipy import sp.np.conj, real, imag, real_if_close
#from scipy import arange, array, sum, np.outer, mgrid, dot
from scipy.special import laguerre, genlaguerre
import scipy.linalg as la
#from scipy.linalg import det, logm
#from enthought.mayavi import mlab

#import quantumoptics as qo


def chop(a):
    """
    Chop an ndarray to maximum precision.
    """
    return a.round(sp.finfo(float).precision)

def even(n):
    return ((n+1)%2)
    
def odd(n):
    return n%2

# =============================================================================
#  pure quantum state definitions
# =============================================================================

# ==== vacuum ====

def c_vac(n):
    """
    c_n for vacuum, |0>.
    """
    if n == 0:
        c = 1
    else:
        c = 0
        
    return c
    
def rho_vac(photons=15):
    """
    Density matrix for vacuum, |0>.
    
    photons = 15 : photon number cut-off
    """
    cvec = [c_vac(n) for n in range(photons + 1)]
    return np.outer(cvec, np.conj(cvec))
    
def wig_vac():
    """
    Wigner function W(x, p) for vacuum, |0>.
    """
    return lambda x, p: 1/pi * exp(-x*x - p*p)
    
    
# ==== Fock states ====

def c_fock(m, n):
    """
    c_n for the m-photon Fock state, |m>.
    """
    if m==n:
        c = 1
    else:
        c = 0
        
    return c

def rho_fock(m, photons=15):
    """
    Density matrix for the m-photon Fock state, |m>.
    
    photons = 15 : photon number cut-off
    """
    cvec = [c_fock(m, n) for n in range(photons + 1)]
    return np.outer(cvec, np.conj(cvec))
    
def wig_fock(m):
    """
    Wigner function W(x, p) for the m-photon Fock state, |m>.
    """    
    return lambda x, p: 1/pi * exp(-x*x - p*p) * ((-1)**m * 
                         laguerre(m)(2*x*x + 2*p*p))
    
    
# ==== coherent state ====

def c_coh(alpha, n):
    """
    c_n for coherent state, |alpha>.
    """
    c = exp(-abs(alpha)**2 / 2) * alpha**n / sqrt(factorial(n))
        
    return c
    
def rho_coh(alpha, photons=15):
    """
    Density matrix for coherent state, |alpha>.
    
    alpha : coherent (complex) amplitude
    photons = 15 : photon number cut-off
    """
    cvec = [c_coh(alpha, n) for n in range(photons + 1)]
    return np.outer(cvec, np.conj(cvec))
    
def wig_coh(alpha):
    """
    Wigner function W(x, p) for coherent state, |alpha>.
    
    alpha : coherent (complex) amplitude
    """
    return lambda x, p: wig_vac()(x - sqrt(2) * alpha.real, 
                                p - sqrt(2) * alpha.imag)


# ==== coherent state qubit ====

def c_coqu(alpha, theta, phi, n):
    """
    c_n for an arbitrary superposition of opposite-phase coherent states
    c_coqu(alpha, theta, phi, n)
    |alpha> for theta=0, |-alpha> for theta=pi.
    """
    c = 1/sqrt(1 + 2*cos(theta/2)*sin(theta/2)*cos(phi)*exp(-2*alpha**2)) * (
            cos(theta/2) * c_coh(alpha, n) +
            exp(1j*phi) * sin(theta/2) * c_coh(-alpha, n))
    return c
    
def rho_coqu(alpha, theta, phi, photons=15):
    """
    Density matrix for an arbitrary superposition of opposite-phase coherent states
    rho_coqu(alpha, theta, phi, n)
    |alpha> for theta=0, |-alpha> for theta=pi.
    """    
    cvec = c_coqu(alpha, theta, phi, np.arange(photons + 1))
    
    return np.outer(cvec, np.conj(cvec))
    
def cat2coqu(alpha, theta, phi):
    """
    Get coherent state qubit coordinates from cat qubit coordinates
    """
    Np = sqrt(2+2*exp(-2*alpha**2))
    Nm = sqrt(2-2*exp(-2*alpha**2))
    Nc = sqrt((1-exp(-4*alpha**2)) / (1-exp(-2*alpha**2)*cos(theta)))
    a = cos(theta/2) / Np
    b = cos(phi) * sin(theta/2) / Nm
    c = sin(phi) * sin(theta/2) / Nm
    
    theta_coh = 2*arccos(Nc * sqrt((a+b)**2+c**2))
    phi_coh = angle(a-b-c*1j) - angle(a+b+c*1j)
        
    return theta_coh, phi_coh

def coqu2cat(alpha, theta, phi):
    """
    Get cat qubit coordinates from coherent state qubit coordinates
    """
    Np = sqrt(2+2*exp(-2*alpha**2))
    Nm = sqrt(2-2*exp(-2*alpha**2))
    Na = sqrt(1 + 2*cos(phi)*cos(theta/2)*sin(theta/2)*exp(-2*alpha**2))
    a = cos(theta/2) / (2*Na)
    b = cos(phi) * sin(theta/2) / (2*Na)
    c = sin(phi) * sin(theta/2) / (2*Na)
    
    theta_cat = 2*arccos(Np * sqrt((a+b)**2+c**2))
    phi_cat = angle(Nm*(a-b-c*1j)) - angle(Np*(a+b+c*1j))
    
    return theta_cat, phi_cat   
    
# ==== squeezed vacuum ====

def c_sqv(r, n):
    """ 
    c_n for squeezed vacuum, S(r)|0>, with squeezing along p-direction
    for real positive r and squeezing along p-direction for negative r.
    """
    c = ((1 - tanh(abs(r))**2)**.25 * (exp(angle(r)*1j)*tanh(abs(r))/2)**(n/2) *
            sqrt(factorial(n)) / factorial(n/2)) * ((n+1) % 2)
    
    return chop(c)
    

def rho_sqv(r, phi=0, photons=15):
    """
    Density matrix for squeezed vacuum, S(r)|0>, with squeezing along 
    p-direction.
    
    r : squeezing parameter
    photons = 15 : photon number cut-off
    """
    cvec = [c_sqv(r, n) for n in range(photons + 1)]
    return np.outer(cvec, np.conj(cvec))
    
def wig_sqv(r):
    """
    Wigner function W(x,p) for squeezed vacuum, S(r)|0>, with squeezing
    along p-direction.
    
    r : squeezing parameter
    """
    return lambda x, p: wig_vac()(exp(-r)*x, exp(r)*p)
    

# ==== 1-photon subtracted squeezed vacuum ====

def c_1ps(r, n):
    """
    c_n for 1-photon subtracted squeezed vacuum (squeezed single photon),
    a S(r)|0> ~ S(r)|1>.
    """
    c = sqrt(n+1) / sinh(r) * c_sqv(r, n+1)
    
    return c
    
def rho_1ps(r, photons=15):
    """
    Density matrix for 1-photon subtracted squeezed vacuum (squeezed single 
    photon), a S(r)|0> ~ S(r)|1>.
    
    r : squeezing parameter
    photons = 15 : photon number cut-off
    """
    cvec = [c_1ps(r, n) for n in range(photons + 1)]
    
    return np.outer(cvec, np.conj(cvec))
    
def wig_1ps(r):
    """
    Wigner function W(x,p) for 1-photon subtracted squeezed vacuum (squeezed 
    single photon), a S(r)|0> ~ S(r)|1>.
    
    r : squeezing parameter
    """
    return lambda x, p: wig_fock(1)(exp(-r)*x, exp(r)*p)


# ==== 2-photon subtracted squeezed vacuum ====

def c_2ps(r, n):
    """
    c_n for 2-photon subtracted squeezed vacuum,
    a^2 S(r)|0>.
    """
    c = sqrt(2*(n+1)*(n+2)) / (sinh(r)*sqrt(3*cosh(2*r)-1)) * c_sqv(r, n+2)
    
    return c
    
def rho_2ps(r, photons=15):
    """
    Density matrix for 2-photon subtracted squeezed vacuum
    a^2 S(r)|0>.
    
    r : squeezing parameter
    photons = 15 : photon number cut-off
    """
    cvec = [c_2ps(r, n) for n in range(photons + 1)]
    
    return np.outer(cvec, np.conj(cvec))


# ==== squeezed m-photon Fock state ====

def c_sqFock(r, m, n):
    """
    DOES NOT YET WORK FOR VECTOR INPUT
    c_n for a squeezed m-photon state, S(r)|m>.
    """
    
    if (m-n) % 2 == 1:
        c = 0
    else:
        k = np.arange(max(m,n)/2)
        s = (-1)**k
        if n<m:
            m,n = n,m
            s = (-1)**((m-n)/2 + k)
            
        c = (sqrt(factorial(m)*factorial(n) / cosh(r)) * 
             (s * (tanh(r)/2)**((n-m)/2 + 2*k) / cosh(r)**(m-2*k) /
              (factorial(k) * factorial((n-m)/2 + k) * factorial(m-2*k))).sum())
              
    return c

# ==== squeezed qubit ====

def c_sqq(r, theta, phi, n):
    """
    c_n for a squeezed qubit -- the superposition of S(r)|0>
    (North pole) and S(r)|1> (South pole).
    """
    c = cos(theta/2) * c_sqv(r, n) + sin(theta/2) * exp(1j*phi) * c_1ps(r, n)
    
    return c

def rho_sqq(r, theta, phi, photons=15):
    """
    Density matrix for a squeezed qubit -- the superposition of S(r)|0>
    (North pole) and S(r)|1> (South pole).
    
    r : squeezing parameter
    theta : superposition weight - Bloch sphere latitude from North pole
    phi : superposition phase - Bloch sphere longitude from x-axis
    photons = 15 : photon number cut-off
    """
    cvec = [c_sqq(r, theta, phi, n) for n in range(photons + 1)]
    
    return np.outer(cvec, np.conj(cvec))

def wig_sqq(r, theta, phi):
    """
    Wigner function W(x,p) for a squeezed qubit -- the superposition of S(r)|0>
    (North pole) and S(r)|1> (South pole).
    
    r : squeezing parameter
    theta : superposition weight - Bloch sphere latitude from North pole
    phi : superposition phase - Bloch sphere longitude from x-axis
    """
    return lambda x, p: ( cos(theta/2)**2 * wig_sqv(r)(x, p) 
                         + sin(theta/2)**2 * wig_1ps(r)(x, p)
                         + sqrt(2) * sin(theta) * (cos(phi) * 
                           exp(-r)*x + sin(phi) * exp(r)*p) * wig_sqv(r)(x, p) )


# ==== squeezed 1-2 PS qubit ====

def c_sq12q(r, theta, phi, n):
    """
    c_n for a squeezed 1 or 2 photon subtracted qubit -- the superposition of 
    a^2 S(r)|0> (North pole) and a S(r)|0> (South pole).
    """
    c = cos(theta/2) * c_2ps(r, n) + sin(theta/2) * exp(1j*phi) * c_1ps(r, n)
    
    return c

def rho_sq12q(r, theta, phi, photons=15):
    """
    Density matrix for a squeezed 1 or 2 photon subtracted qubit -- the 
    superposition of a^2 S(r)|0> (North pole) and a S(r)|0> (South pole).
    
    r : squeezing parameter
    theta : superposition weight - Bloch sphere latitude from North pole
    phi : superposition phase - Bloch sphere longitude from x-axis
    photons = 15 : photon number cut-off
    """
    cvec = [c_sq12q(r, theta, phi, n) for n in range(photons + 1)]
    
    return np.outer(cvec, np.conj(cvec))



# ==== cat state superposition ====

def c_cat(alpha, theta, phi, n):
    """
    c_n for a cat state superposition:
    c_cat(alpha, theta, phi, n)
    Even cat for theta=0, odd for theta=pi.
    """
    c = alpha**n / sqrt(factorial(n)) * (
          even(n) * cos(theta/2) / sqrt(cosh(abs(alpha)**2)) +
          odd(n) * exp(1j*phi) * sin(theta/2) / sqrt(sinh(abs(alpha)**2)))
    return c
    
def rho_cat(alpha, theta, phi, photons=15):
    """
    Density matrix for a cat state superposition:
    rho_cat(alpha, theta, phi, n)
    Even cat for theta=0, odd for theta=pi.
    """    
    cvec = c_cat(alpha, theta, phi, np.arange(photons + 1))
    
    return np.outer(cvec, np.conj(cvec))

def wig_cat(alpha, theta, phi):
    """
    Wigner function for a cat state superposition:
    wig_cat(alpha, theta, phi, x, p)
    Even cat for theta=0, odd for theta=pi.
    alpha should be real.
    """
    norm = 1/(1-exp(-4*alpha**2))
    Ccoh = lambda sig: 1/2 * (norm -
                cos(theta) / (2*sinh(2*alpha**2)) +
                sp.sign(sig) * sqrt(norm) * cos(phi) * sin(theta))
    Cint = lambda p: (
                norm * (cos(theta)-exp(-2*alpha**2)) * cos(2*sqrt(2)*alpha*p) +
                sqrt(norm) * sin(phi) * sin(theta) * sin(2*sqrt(2)*alpha*p))
    
    return lambda x, p: (
                Ccoh(1) * wig_coh(alpha)(x, p) +
                Ccoh(-1) * wig_coh(-alpha)(x, p) +
                Cint(p) * wig_vac()(x, p))
                


# =============================================================================
#  quantum optics with Wigner functions
# =============================================================================


#def bs_wig(wig_in, T):
#    """
#    wig_after_bs = bs_wig(wig_in, T)
#    
#    Beamsplitter acting on two input (single mode) Wigner functions
#    """
#    return lambda
    

# =============================================================================
#  plotting and related functions
# =============================================================================

#def rho2wig_table(rho, endx=4.0, endp=4.0, res=0.1, ret_xp=True):
#    """
#    [X, P,] W = rho2wig_table(rho[, endx, endp, res, ret_xp])
#    
#    Convert density matrix to an array of Wigner function values.
#    
#    rho : density matrix
#    endx = 4.0 : endpoints for the x-grid [-endx ; endx]
#    endp = 4.0 : endpoints for the p-grid [-endp ; endp]
#    res = 0.1 : x- and p-resolution
#    ret_xp = True : return x- and p-grids for plotting
#    """
#    #X = arange(-endx, endx + res, res)
#    #P = arange(-endp, endp + res, res)
#    X, P = np.mgrid[-endx:endx + res:res, -endx:endx + res:res]
#    W = qo.rho2wig(rho, X, P)
#    
#    if ret_xp:
#        return X, P, W
#    else:
#        return W
        

def plotWig(*XPW, **kwargs):
    """
    3D surface plot of Wigner function.
    Parameters are either W (Wigner function) only, or X, P, W
    """
    from mayavi import mlab
    defaultPlottype = 'mpl'    
    
    if 'plottype' in kwargs:
        plottype = kwargs['plottype']
    else:
        plottype = defaultPlottype
        
    if plottype == 'mpl': 
        mlab.surf(*XPW, warp_scale='auto')
    elif plottype == 'qwt':
        import guidata
        from guiqwt.plot import ImageWidget
        from guiqwt.builder import make
        app = guidata.qapplication()
        win = ImageWidget()
        win.resize(600, 600)
    
        wig = make.image(XPW[2])
        win.plot.add_item(wig)
        
        win.show()
        app.exec_()
    

  

def wig2photonnumber(wig):
    x,dx = sp.linspace(-20,20,num=1001,retstep=True)
    X,P = sp.meshgrid(x,x)
    return (sp.array([(lambda x,p: 
        2*pi * wig(x,p) * wig_fock(m)(x,p))(X,P).sum() * dx**2 
        for m in range(20)])*np.arange(20)).sum()
        
# =============================================================================
#  constants and stuff
# =============================================================================

deg = 360/2/pi
