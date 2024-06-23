# -*- coding: utf-8 -*-
"""
Created on Mon May 09 15:24:26 2011

@author: Jonas S. Neergaard-Nielsen

General quantum optics formulas

ToDo:
* Vectorize xn, wig_mn
"""


import numpy as np
import scipy.linalg as la
import scipy.special as sf
import scipy.stats as stats
from numpy import pi, exp, log, sqrt, angle
from numpy import asmatrix, asarray, finfo, zeros, mgrid, meshgrid, array, \
                  arange, linspace, real, imag, real_if_close, dot, tensordot, \
                  nonzero, outer, tile, sum, arccos, arctan2, conj, \
                  mat, isfinite, ogrid, eye
from numpy.random import rand
import scipy.optimize as optimize
import qopt.quantumstates as qs

# ================
#  misc functions
# ================

#hermite = vectorize(lambda n, x: sf.hermite(n)(x))
#hermite2 = lambda n, x: vectorize(lambda n: sf.hermite(n)(x), otypes=[object])(n)
#genlaguerre = vectorize(lambda m, n, x: sf.genlaguerre(m, n, x))

numeps = finfo(float).eps

fact = lambda k: sf.gamma(k+1)

    
# =========================================
#  functions for building Wigner functions
# =========================================    
    

def xn(x, n, theta=0):
    """
    xn(x, n, theta=0)
    ---------------
    Number state wave function <x|n>, including phase factor.
    """
    return exp(-1j * n * theta) * sf.hermite(n)(x) * exp(-x*x/2.) / \
            sqrt(sqrt(pi) * 2**n * fact(n))


def wig_mn(m, n, x, p):
    if n > m:
        m, n = n, m
        p = -p
    
    return (-1)**n * (x-p*1j)**(m-n) * 1/pi * exp(-x*x - p*p) * \
            sqrt(2**(m-n) * fact(n) / fact(m)) * \
            sf.genlaguerre(n, m-n)(2*x*x + 2*p*p)
        

Wmn_table = []

def getWmn(X, P, photons):
    for d in Wmn_table:
        try:
            if not (d['X']-X).any() and not (d['P']-P).any() \
                   and d['photons']==photons:
                return d['Wmn']
        except ValueError:
            pass
            
    Wmn = zeros((photons+1, photons+1, len(X), len(P[0])), dtype=np.complex64)
    for m in range(photons+1):
        for n in range(m+1):
            wmn = wig_mn(m, n, X, P)
            Wmn[m,n] = wmn
            Wmn[n,m] = wmn.conjugate()
            
    Wmn_dict = dict(X=X, P=P, photons=photons, Wmn=Wmn)
    Wmn_table.append(Wmn_dict)
    
    return Wmn
    

def makeGrid(xlim, nx, plim=None, np=None, return_res=False):
    try:
        if len(xlim) == 1:
            xmax = abs(xlim[0])
            xmin = -xmax
        elif len(xlim) == 2:
            xmin, xmax = xlim
        else:
            raise ValueError("xlim should be one or two elements long.")
    except TypeError:
        xmax = abs(xlim)
        xmin = -xmax    
        
    if plim != None:
        try:
            if len(plim) == 1:
                pmax = abs(plim[0])
                pmin = -pmax
            elif len(plim) == 2:
                pmin, pmax = plim
            else:
                raise ValueError("plim should be one or two elements long.")
        except TypeError:
            pmax = abs(plim)
            pmin = -pmax        
    else:
        pmin = xmin
        pmax = xmax
        
    if np == None:
        np = nx  
        
    X, P = mgrid[xmin:xmax:nx*1j, pmin:pmax:np*1j]
    dx = (xmax-xmin)/(nx-1)
    dp = (pmax-pmin)/(np-1)
    
    if return_res:
        return X, P, dx, dp
    else:
        return X, P
    

def makeGrid_alt(xlim, xres, plim=None, pres=None):
    """
    Build X and P grids for Wigner function plotting.
    xlim: xmin or [xmin, xmax]    
    If xlim is a scalar, xmin and xmax will be equal.
    xres: Resolution or number of points.
    If xres is a float, it will be the resolution. If it is an integer,
    the number of points.
    If plim and/or pres are not set, they will be the same as for x.
    ------
    Returns X, P, dx, dp
    dx/dp are resolution if xres/pres were number of points -- if xres/pres
    were resolution, dx/dp will be number of points.
    """
    try:
        if len(xlim) == 1:
            xmax = abs(xlim[0])
            xmin = -xmax
        elif len(xlim) == 2:
            xmin, xmax = xlim
        else:
            raise ValueError("xlim should be one or two elements long.")
    except TypeError:
        xmax = abs(xlim)
        xmin = -xmax
        
    if plim != None:
        try:
            if len(plim) == 1:
                pmax = abs(plim[0])
                pmin = -pmax
            elif len(plim) == 2:
                pmin, pmax = plim
            else:
                raise ValueError("plim should be one or two elements long.")
        except TypeError:
            pmax = abs(plim)
            pmin = -pmax        
    else:
        pmin = xmin
        pmax = xmax
        
    if pres == None:
        pres = xres        
        
    if type(xres) == type(0) or \
       (type(xres) == type(array(0)) and xres.dtype == array(0).dtype):
        x, dx = linspace(xmin, xmax, xres, retstep=True)
    elif type(xres) == type(0.) or \
       (type(xres) == type(array(0.)) and xres.dtype == array(0.).dtype):
        x = arange(xmin, xmax+xres, xres)
        dx = len(x)
        
    if type(pres) == type(0) or \
       (type(pres) == type(array(0)) and pres.dtype == array(0).dtype):
        p, dp = linspace(pmin, pmax, pres, retstep=True)
    elif type(pres) == type(0.) or \
       (type(pres) == type(array(0.)) and pres.dtype == array(0.).dtype):
        p = arange(pmin, pmax+pres, pres)
        dp = len(p)
        
    X,P = meshgrid(x,p)
    
    return X,P,dx,dp
    
    
def hermitify(rho):
    """
    Make matrix Hermitian by averaging it and its Hermite conjugate.
    """    
    return (rho + rho.T.conj()) / 2
    
# ============================
#  various quantum optics functions
# ============================


def entropyGaussian(C):
    """
    entropyGaussian(C)
    ------------------
    von Neumann entropy of a Gaussian state with covariance matrix C.
    Source: Genoni et al., arXiv:1101.5777
    """
    x = sqrt(la.det(C))
    ent = 0 if x == 0.5 else (x + 0.5) * log(x + 0.5) - (x - 0.5) * log(x - 0.5)
    return ent



        
# =============================
#  single mode quantum state class
# =============================

class QuantumState:
    """
    QuantumStateDM(rho)
    -------------------
    Quantum state defined by its density matrix (in Fock state representation).
    """
    def __init__(self, rho=None, initialWigner=False):
        self.rho = asarray(rho)
        self.nmax = rho.shape[0] - 1
        self.update(initialWigner)
        
               
    def update(self, includeWigner=False):
        self.mean = self._mean()
        self.phaseSpaceAngle = self._phaseSpaceAngle()
        self.covariance = self._covariance()
        self.purity = self._purity()
        self.photon_number = self._photonnumber()
        self.entropy = self._entropy()
        self.nonGaussianity = self._nonGaussianity()
        self.superness = self._superness()
        # It may be wrong to use hermitify here!!!
        # self._sqrtrho = hermitify(asmatrix(la.sqrtm(self.rho + numeps)))
        self._sqrtrho = asmatrix(la.sqrtm(self.rho + numeps))
        if includeWigner: self.buildWigner()        
        

    def wig(self, x, p):
        """
        wig(x, p)
        ---------
        Get Wigner function W(x,p).
        """     
        diagsum = sum([self.rho[m, m] * wig_mn(m, m, x, p) for m in
                range(self.nmax+1)], axis=0)
    
        offdiagsum = sum([2 * real(self.rho[m, n] * wig_mn(m, n, x, p)) 
                for m in range(self.nmax+1) for n in range(m)], axis=0)
    
        return real_if_close(diagsum + offdiagsum)
     
    
    def marginal(self, x, theta=0):
        XN = array([xn(x, n, theta) for n in arange(self.nmax+1)])
        return real_if_close(sum(dot(XN.T, self.rho)*XN.T.conj(), axis=1))
        
           
        
#    def buildWigner(self, xlim=(-4,4), plim=(-4,4), xres=.1, pres=.1):
#        self.X, self.P = mgrid[xlim[0]:xlim[1]+xres:xres,
#                                  plim[0]:plim[1]+pres:pres]
#        self.W = self.wig(self.X, self.P)
#    def buildWigner(self, xlim=(-4,4), nx=81, plim=(-4,4), np=81):
#        self.X, self.P, self.dx, self.dp = makeGrid(xlim, nx, plim, np, True)
#        self.W = self.wig(self.X, self.P)
    
    def buildWigner(self, xlim=4, nx=81, plim=None, np=None):
        self.X, self.P, self.dx, self.dp = makeGrid(xlim, nx, plim, np, True)
        Wmn = getWmn(self.X, self.P, self.nmax)
        self.W = tensordot(self.rho, Wmn, 2).real
        
        
    def _mean(self):
        """
        mean()
        ------
        Mean vector [E(x), E(p)]  (displacement for a Gaussian state).
        """
        k = arange(len(self.rho))
        Ex = (sqrt(k[1:]/2.) * self.rho.diagonal(1)).sum().real * 2
        Ep = -(sqrt(k[1:]/2.) * self.rho.diagonal(1)).sum().imag * 2

        return array([Ex, Ep])
        
    def _phaseSpaceAngle(self):
        return angle(dot(self._mean(), [1,1j]), deg=True)

    def _covariance(self):
        """
        covariance()
        ------------
        Covariance matrix [[Var(x), Cov(x,p)], [Cov(x,p), Var(p)]].
        """
        k = arange(len(self.rho))
        Ex, Ep  = self._mean()        
        Exx = 0.5 * (((2*k+1) * self.rho.diagonal(0)).sum() + 
                (sqrt(k[1:-1] * k[2:]) * self.rho.diagonal(2)).sum().real * 2)
        Epp = 0.5 * (((2*k+1) * self.rho.diagonal(0)).sum() - 
                (sqrt(k[1:-1] * k[2:]) * self.rho.diagonal(2)).sum().real * 2)
        Exp = 0.5j * (1 + 
                (sqrt(k[1:-1] * k[2:]) * self.rho.diagonal(2)).sum().imag * 2j)
                
        Vxx = Exx - Ex**2
        Vpp = Epp - Ep**2
        Vxp = Exp.real - Ex * Ep
                
        return real_if_close([[Vxx, Vxp], [Vxp, Vpp]])
        
        
    def _purity(self):
        """
        purity()
        --------
        State purity.
        """
        return real_if_close(dot(self.rho, self.rho).trace())
        
        
    def _photonnumber(self):
        """
        _photon_number()
        ----------------
        Mean photon number.
        """
        return (arange(self.nmax+1) * self.rho.diagonal().real).sum()
        

    def fidelity(self, sigma):
        """
        fidelity(sigma)
        ---------------
        Fidelity between state rho and a second state sigma.
        sigma can be a QuantumStateDM instance or a simple DM array.
        The two density matrices should be same dimension.        
        """
        
        if hasattr(sigma, 'rho'):        
            sigma = asmatrix(sigma.rho)
        else:
            sigma = asmatrix(sigma)
            
        size = min([len(self._sqrtrho), len(sigma)])
        
        if self._sqrtrho.shape != sigma.shape:
            print("The two density matrices have unequal dimensions.")
            print(("Cropping down to %d photons" % (size - 1)))
            
#        return real_if_close(la.sqrtm(sqrtrho * sigma * sqrtrho).trace())
        return real(la.sqrtm(self._sqrtrho[:size,:size] * sigma[:size,:size] *
                                         self._sqrtrho[:size,:size]).trace()**2)


    def tracedistance(self, sigma):
        """
        tracedistance(sigma)
        ---------------
        Trace distance between state rho and a second state sigma.
        sigma can be a QuantumStateDM instance or a simple DM array.
        The two density matrices should be same dimension.        
        """
        
        try:        
            sigma = asmatrix(sigma.rho)
        except AttributeError:
            sigma = asmatrix(sigma)
            
        if self._sqrtrho.shape != sigma.shape:
            print("The two density matrices have unequal dimensions.")            
            return None
            
#        return real_if_close(la.sqrtm(sqrtrho * sigma * sqrtrho).trace())
        return la.svdvals(self.rho - sigma).sum()/2
        
   
#    def _entropy(self):
#        """
#        S = entropy(rho)
#        von Neumann entropy of a given state with density matrix rho.
#        """
#        return real_if_close(-dot(self.rho, 
#                                        logmALT(self.rho + numeps)).trace())
                                        
    def _entropy(self):
        """                                    
        S = entropy(rho)
        von Neumann entropy of a given state with density matrix rho.
        Method borrowed from QuTiP's entropy_vn.
        """
        eigvals = la.eig(self.rho, right=False)
        eigvals = eigvals[eigvals!=0]
        return -(eigvals * log(eigvals)).sum().real

    def _nonGaussianity(self):
        """
        delta = nonGaussianity(rho)
        Non-Gaussianity of density matrix rho, defined as difference in von
        Neumann entropy to the Gaussian state with same covariance.
        """
        return entropyGaussian(self._covariance()) - self._entropy()
        
    def _superness(self):
        k = arange(len(self.rho))
        superness = -sum(self.rho[:-1,:-1] * \
                (self.rho.T * sqrt(outer(k, k)))[1:, 1:]) + \
                sum(self.rho * self.rho.T * tile(k, [len(self.rho), 1]))
        return real_if_close(superness)
        


    def generateSamples(self, number=10000, theta=0., res=.01, edge=10.):
        """
        """
        x = arange(-edge, edge, res)
        cdf = self.marginal(x, theta).cumsum() * res
        indices = [nonzero(r > cdf)[0][-1] for r in rand(number)*cdf[-1]]
        return x[indices] + res/2.
        
    def loss(self, loss):
        """
        s2 = s.loss(loss)
        Imposes a loss (intensity, 0 <= loss <= 1) on the DM and returns a new 
        instance.
        Described by the generalized Bernoulli transformation (beamsplitter).
        """
        K, N = mgrid[:self.nmax+1, :self.nmax+1]
        bino = stats.binom.pmf(K, N, 1-loss)
        sigma = self.rho.copy()  # shape only
        for m in range(len(sigma)):
            for n in range(len(sigma)):
                maxj = self.nmax - max(m,n)
                sigma[m,n] = (self.rho.diagonal(n-m)[min(m,n):] * 
                              sqrt(bino.diagonal(m)[:maxj+1] *
                                      bino.diagonal(n)[:maxj+1])).sum()
        return self.__class__(sigma)
        
    def rotate(self, angle):
        """
        s2 = s.rotate(angle)
        Rotates the state in phase space and returns a new instance.
        """
        M, N = mgrid[:self.nmax+1, :self.nmax+1]
        R = exp(1j * (M-N) * angle)
        sigma = self.rho * R
        return self.__class__(sigma)
            


# =============================================================================
#  Specialized quantum state classes with extra methods
# =============================================================================

#class QuantumStateW(QuantumState):
#    """
#    Quantum state defined in terms of its Wigner function.
#    """
#    def __init__(self, wigf, gridparam=[4,41], photons=20):
##        self.wig = wigf
##        self.wig = types.MethodType(lambda self, x, p: wigf(x,p), self)
##        self.wig = lambda x, p: wigf(x, p)
#        def wig(self, x, p):
#            return wigf(x, p)
#        self.nmax = photons
#        self.buildWignerFromWig(*gridparam)
#        self.rho = 2*pi * tensordot(getWmn(self.X,self.P,photons), self.W) \
#                       * self.dx * self.dp
#        self.rho = hermitify(self.rho / self.rho.trace())
#        self.update(False)
#        
#    def buildWignerFromWig(self, xlim=4, nx=81, plim=None, np=None):
#        self.X, self.P, self.dx, self.dp = makeGrid(xlim, nx, plim, np, True)
#        self.W = self.wig(self.X, self.P)       

class QuantumStateW(QuantumState):
    """
    Quantum state defined in terms of its Wigner function.
    """
    def __init__(self, wig, gridparam=[4,81], photons=20):
        self.X, self.P, self.dx, self.dp = makeGrid(*gridparam, return_res=True)
        self.W = wig(self.X, self.P)
        self.nmax = photons
        self.rho = 2*pi * tensordot(getWmn(self.X,self.P,photons).conj(), self.W) \
                       * self.dx * self.dp
        self.rho = hermitify(self.rho / self.rho.trace())
        self.update(False)
        

class CVQubitState(QuantumState):
    def __init__(self, rho=None, initialWigner=False):
        QuantumState.__init__(self, rho, initialWigner)
        
    def getSuperposition(self, up, down):
        """
        Calculate the Bloch vector of rho as a superposition of states up and down.
        up and down are vectors of Fock state components, same length as dim of rho.  
        ----
        Returns r, theta, phi
        """
        uu = dot(conj(up), dot(self.rho, up))
        dd = dot(conj(down), dot(self.rho, down))
        ud = dot(conj(up), dot(self.rho, down))
        x = 2 * real(ud)
        y = -2 * imag(ud)
        z = real_if_close(uu - dd)
        r = sqrt(x*x + y*y + z*z)
        theta = arccos(z / r)
        phi = arctan2(y, x)
        
        return r, theta, phi  

    def cat_overlap(self, a, theta, phi):
        return self.fidelity(qs.rho_cat(a,theta,phi,self.nmax))

    def getCatSuperposition(self, a):
        return self.getSuperposition(qs.c_cat(a,0,0,arange(self.nmax+1)), 
                                     qs.c_cat(a,pi,0,arange(self.nmax+1)))


class CVQubitStateW(QuantumStateW, CVQubitState):
    def __init__(self, wig, gridparam=[4,41], photons=20):
        QuantumStateW.__init__(self, wig, gridparam, photons)
    
    
    
# =============================================================================
#  Wigner function <--> density matrix transformations
# =============================================================================


            
            
#def rho2wig(rho, x, p):
#    """
#    Convert density matrix to Wigner function:
#    Get W(x, p) for a given rho.
#    """
#    dim = len(rho)
#    grid = array([[m, n] for m in range(dim) for n in range(m)]).transpose()
#    diag = arange(dim)
#    
#    diagsum = sum([rho[m, m] * wig_mn(m, m, x, p) for m in range(dim)], axis=0)
#
#    offdiagsum = sum([2 * real(rho[m, n] * wig_mn(m, n, x, p)) 
#                      for m in range(dim) for n in range(m)], axis=0)
#    
#    return real_if_close(diagsum + offdiagsum)
#    
#
#def wig2rho(X, P, W, photons):
#    dx = (X[1:,0]-X[:-1,0]).mean()
#    dp = (P[0,1:]-P[0,:-1]).mean()
#    return 2*pi * tensordot(getWmn(X,P,photons), W) * dx*dp


# =============================================================================
#  Some fitting functions
# =============================================================================

def find_cat(s, alpha=None, printing=False):
    """
    find_cat(s, alpha=None, printing=False)
    ---------------------------------------
    Finds cat with best overlap with state s.
    If alpha is provided, that value is fixed and only theta and phi are
    free parameters. Otherwise alpha is also fitted.
    If printing=True, writes various messages.
    """

    if alpha:
        a_bounds = (alpha, alpha)
    else:
        a_bounds = (.01, None)
    
    if printing: print('Trying phi_ini=0...')
    best_cat1, nfeval1, rc1 = optimize.fmin_tnc(
            lambda atp: 1-s.cat_overlap(*atp), array([.5,pi/2,0]),  
            bounds=[a_bounds,(0,pi),(-pi,pi)], approx_grad=True, disp=0)
    if rc1 != 1 and printing:
        print((optimize.tnc.RCSTRINGS[rc1]))
        
    if printing: print('Trying phi_ini=pi... ')
    best_cat2, nfeval2, rc2 = optimize.fmin_tnc(
            lambda atp: 1-s.cat_overlap(*atp), array([.5,pi/2,pi]),  
            bounds=[a_bounds,(0,pi),(0,2*pi)], approx_grad=True, disp=0)
    if rc2 != 1 and printing:
        print((optimize.tnc.RCSTRINGS[rc2]))
    
    if s.cat_overlap(*best_cat1) > s.cat_overlap(*best_cat2):
        best_cat = best_cat1
    else:
        best_cat = best_cat2

    if printing:
        print(('Best fit with cat (alpha,theta,phi) = (%.3f, %.1f\xb0, %.1f\xb0)' %
        (best_cat[0], best_cat[1]*180/pi, best_cat[2]*180/pi)))
        print(('Fidelity =', s.cat_overlap(*best_cat)))
        print('')
        
    return best_cat, s.cat_overlap(*best_cat)
    
    
#####
# alternative logm - only change from the one in scipy.linalg is to remove
# mat( ) from the first line. With that the function would sometimes randomly
# give a NaN error
#
def logmALT(A, disp=True):
    """Compute matrix logarithm.

    The matrix logarithm is the inverse of expm: expm(logm(A)) == A

    Parameters
    ----------
    A : array, shape(M,M)
        Matrix whose logarithm to evaluate
    disp : boolean
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)

    Returns
    -------
    logA : array, shape(M,M)
        Matrix logarithm of A

    (if disp == False)
    errest : float
        1-norm of the estimated error, ||err||_1 / ||A||_1

    """
    # Compute using general funm but then use better error estimator and
    #   make one step in improving estimate using a rotation matrix.
    A = asarray(A)
    F, errest = la.funm(A,log,disp=0)
    errtol = 1000*numeps
    # Only iterate if estimate of error is too large.
    if errest >= errtol:
        # Use better approximation of error
        errest = la.misc.norm(la.expm(F)-A,1) / la.misc.norm(A,1)
        if not isfinite(errest) or errest >= errtol:
            N,N = A.shape
            X,Y = ogrid[1:N+1,1:N+1]
            R = mat(la.orth(eye(N,dtype='d')+X+Y))
            F, dontcare = la.funm(R*A*R.H,log,disp=0)
            F = R.H*F*R
            if (la.misc.norm(imag(F),1)<=1000*errtol*la.misc.norm(F,1)):
                F = mat(real(F))
            E = mat(la.expm(F))
            temp = mat(la.solve(E.T,(E-A).T))
            F = F - temp.T
            errest = la.misc.norm(la.expm(F)-A,1) / la.misc.norm(A,1)
    if disp:
        if not isfinite(errest) or errest >= errtol:
            print("Result may be inaccurate, approximate err =", errest)
        return F
    else:
        return F, errest
