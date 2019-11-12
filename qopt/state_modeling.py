# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:42:06 2011

@author: Jonas

Change from _2 to _3: Decided to get rid of mode names and instead only accept
multiplication etc. of states with same number of modes.
Then we need to have an "insert mode" function.

"""



from numpy import pi, sqrt, exp, array, arange, zeros
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import scipy.special as sf
from scipy.misc import factorial
import itertools

pi2 = 2*sp.pi
numeps = sp.finfo(float).eps
fact = lambda k: sf.gamma(k+1)

def chop(a):
    """
    Chop an ndarray to maximum precision.
    """
    return a.round(sp.finfo(float).precision)


def beamsplitter_matrix(theta, modes=2, splitmodes=[0,1]):
    """
    Generate beamsplitter matrix for arbitrary number of modes.
    
    theta : splitting angle, cos(theta) = sqrt(T)
    modes = 2 : number of dimensions
    splitmodes = [1,2] : which two modes to split
    """
    splitmodes = sp.array(splitmodes)
    bsm = sp.diag([1. for i in range(2*modes)])
    bsm[sp.ix_(2 * splitmodes, 2 * splitmodes)] = [
                                        [sp.cos(theta), sp.sin(theta)],
                                        [-sp.sin(theta), sp.cos(theta)]]
    bsm[sp.ix_(2 * splitmodes + 1, 2 * splitmodes + 1)] = [
                                        [sp.cos(theta), sp.sin(theta)],
                                        [-sp.sin(theta), sp.cos(theta)]]

    return sp.matrix(bsm)
   
   
def modes2indices(modes, retgrid=True):
    """
    Convert mode numbers to indices in covariance matrix, 
    e.g. [0,1] -> [0,1,2,3].
    If retgrid==True, returns (indices, grid) where grid can be used for matrix
    indexing.
    """
    indices = sp.array([[2*m, 2*m+1] 
                       for m in modes], dtype=sp.integer).flatten()
    grid = sp.ix_(indices, indices)
    return (indices, grid) if retgrid else indices
    
    
class Gaussian:
    """
    Class for multivariate Gaussian states in x-p phase space.
    
    The vacuum covariance matrix has diagonal entries 0.5.
    """
    #def __init__(self, covariance, disp=None, modeNames=None, prefactor=None):
    def __init__(self, covariance, disp=[], prefactor=None, emptymodes=[]):
        
        self.emptymodes = emptymodes        
        
        cShape = sp.array(covariance).shape
        if len(cShape) == 1 and cShape[0] % 2 == 0:
            self.covariance = sp.matrix(sp.diag(covariance))
        elif len(cShape) == 2 and cShape[0] == cShape[1] and \
                                               cShape[0] % 2 == 0:
            self.covariance = sp.matrix(covariance)
        else:
            raise ValueError('covariance must be 2M x 2M matrix or 2M array')
                    
        self.numModes = cShape[0]//2
        
        if not sp.any(disp):
            self.disp = sp.zeros(2*self.numModes)
        elif len(disp) == 2*self.numModes:
            self.disp = sp.array(disp)
        else:
            raise ValueError('disp must be array of same length as covariance')
            
        self.nonemptymodes = [m for m in range(self.numModes) \
                if m not in emptymodes]
        ne_ind, ne_grid = modes2indices(self.nonemptymodes)

        self.ne_covariance = self.covariance[ne_grid]
        self.ne_disp = self.disp[ne_ind]
        
        self.wigCoeff = sp.matrix(sp.zeros([2*self.numModes, 2*self.numModes]))
        self.wigCoeff[ne_grid] = self.ne_covariance.I
        self.ne_wigCoeff = sp.array(self.ne_covariance.I)
        
        
        if prefactor:
            self.prefactor = prefactor      
        elif len(self.nonemptymodes) > 0:
            self.prefactor = 1/(pi2**len(self.nonemptymodes) * 
                                sp.sqrt(linalg.det(self.ne_covariance)))
        else:
            self.prefactor = 1.
            
        self.norm = self._norm()
        self.photonnumbers = dict(list(zip(['noise','amplitude'], 
                                      self._photonnumber())))
                                
    
    def __str__(self):
        out = self.covariance.__str__() + '\n' + self.disp.__str__()
        return out
            
            
    def __mul__(self, G2):
        """
        G = gaussian_multiply(G1, G2)
        Multiply two Gaussians with same number of modes.
        """
        if not hasattr(G2, 'numModes'):
            prefactor = self.prefactor * G2
            return Gaussian(self.covariance, self.disp, prefactor, 
                            self.emptymodes)
                            
        if not self.numModes == G2.numModes:
            print("G1 and G2 do not have same number of modes")
            return

        ne_modesEither = sorted(list(set(self.nonemptymodes + G2.nonemptymodes)))
        ne_ind, ne_grid = modes2indices(ne_modesEither)

        V1 = self.wigCoeff[ne_grid]
        V2 = G2.wigCoeff[ne_grid]
        d1 = sp.matrix(self.disp[ne_ind]).T  
        d2 = sp.matrix(G2.disp[ne_ind]).T
        
        K = (V1 + V2).I
        d = K * (V1 * d1 + V2 * d2)

        # Take note of the 1/2 in the exp here - I spent
        # 2-3 days trying to find that missing factor of 2!
        prefactor = float(self.prefactor * G2.prefactor * 
                     sp.exp(-1/2 * (d1.T * V1 * d1 +
                                    d2.T * V2 * d2 -
                                    d.T * (V1 + V2) * d)))

        covariance = self.covariance.copy()
        covariance[ne_grid] = K

        disp = self.disp.copy()
        disp[ne_ind] = np.array(d).T[0]
        
        return Gaussian(covariance, disp, prefactor, self.emptymodes)
                
                
    def __rmul__(self, G2):
        return self * G2

       
    def wig(self, *co):
        """Evaluates the Gaussian functional expression in coordinates co."""
        vec = co - self.ne_disp
        return self.prefactor * sp.exp(-0.5 * 
                sp.dot(vec, sp.dot(self.ne_wigCoeff, vec)))
                
    def wig_grid(self, X, P):
        """Evaluates the Gaussian functional expression in grid coords X, Y."""
        XP = sp.dstack((X,P))
        vec = XP - self.ne_disp
        Vdotvec = sp.tensordot(self.ne_wigCoeff, vec, (1,2)).transpose((1,2,0))
        vecdotVdotvec = sp.sum(vec * Vdotvec, axis=2)
        return self.prefactor * sp.exp(-0.5 * vecdotVdotvec)
        
        
    def _norm(self):
        return ( self.prefactor * pi2**len(self.nonemptymodes) / 
                 sp.sqrt(linalg.det(linalg.inv(self.ne_covariance))) )
                 
                 
    def _photonnumber(self):
        n_noise = []
        n_amplitude = []
        for m in range(self.numModes):
            if m in self.emptymodes:
                n_noise.append(None)
                n_amplitude.append(None)
            else:
                ind = modes2indices([m])[0]
                n_noise.append((self.covariance.diagonal()[0, ind].sum() - 1)/2)
                n_amplitude.append((self.disp[ind]**2).sum()/2)
        
        return n_noise, n_amplitude
                 
                 
                 
    def _traceSingleMode(self, mode):
        """
        Trace out a mode and return a new Gaussian with one mode less, or a
        scalar if single mode.
        """
        t_ind, t_grid = modes2indices([mode])   # indices of traced mode
        i = self.nonemptymodes.index(mode)
        r_ind, r_grid = modes2indices(self.nonemptymodes[:i] + 
                                      self.nonemptymodes[i+1:])
        
        # if final mode, return scalar instead of new Gaussian
        if len(r_ind) == 0:
            return self.prefactor * pi2 / sp.sqrt(
                                             linalg.det(self.wigCoeff[t_grid]))

        W0 = self.wigCoeff[t_grid]
        U0 = self.wigCoeff[r_grid]
        V  = self.wigCoeff[sp.ix_(r_ind, t_ind)]
 
        prefactor = self.prefactor * pi2 / sp.sqrt(linalg.det(W0))
        covariance = self.covariance.copy() # only using the shape
        disp = self.disp.copy()
        
        covariance[r_grid] = (U0 - V * W0.I * V.T).I
        covariance[t_ind] = covariance[:, t_ind] = 0
        
        disp[t_ind] = 0
        emptymodes = self.emptymodes + [mode]
        
        return Gaussian(covariance, disp, prefactor, emptymodes)
        
        
    def xHomodyne(self, mode, q):
        if len(self.nonemptymodes) < 2:
            print("Not multimode state")
            return
        
        
        
    def traceMode(self, modes):
        """
        Trace out one or multiple modes and return a new reduced Gaussian or a
        scalar if initially single mode.
        modes: integer or list of integers
        """        
        modes = sp.array(modes, ndmin=1)
        Gtraced = self
        for m in modes:
            Gtraced = Gtraced._traceSingleMode(m)
        return Gtraced       
          
        
    def _vacProjSingleMode(self, mode):
        """
        Project a mode onto vacuum - multiply with vacuum Wigner function
        and trace over that mode.
        """
        cov_vac = [0.5 if i in modes2indices([mode])[0] else 0 
                   for i in range(2*self.numModes)]
        e_modes = [m for m in range(self.numModes) if m != mode]
        Gvac = Gaussian(cov_vac, emptymodes=e_modes)
        return pi2 * (self * Gvac).traceMode(mode)
        
        
    def vacProj(self, modes):
        """
        Project one or more modes onto vacuum - multiply with vacuum Wigner
        function and trace over that mode.
        modes: integer or list of integers
        """
        modes = sp.array(modes, ndmin=1)
        Gproj = self
        for m in modes:
            Gproj = Gproj._vacProjSingleMode(m)
        return Gproj          
        
        
    def xProj(self, mode, r, x):
        """
        Project a mode onto vacuum - multiply with vacuum Wigner function
        and trace over that mode.
        """
        cov_sqz = sp.zeros(2*self.numModes)
        cov_sqz[2*mode:2*(mode+1)] = [0.5*sp.exp(-2*r), 0.5*sp.exp(2*r)]
        disp = sp.zeros(2*self.numModes)
        disp[2*mode] = x
        e_modes = [m for m in range(self.numModes) if m != mode]
        Gsqz = Gaussian(cov_sqz, disp, emptymodes=e_modes)
        return pi2 * (self * Gsqz).traceMode(mode)
        
        
    def pProj(self, mode, r, p):
        """
        Project a mode onto vacuum - multiply with vacuum Wigner function
        and trace over that mode.
        """
        cov_sqz = sp.zeros(2*self.numModes)
        cov_sqz[2*mode:2*(mode+1)] = [0.5*sp.exp(2*r), 0.5*sp.exp(-2*r)]
        disp = sp.zeros(2*self.numModes)
        disp[2*mode+1] = p
        e_modes = [m for m in range(self.numModes) if m != mode]
        Gsqz = Gaussian(cov_sqz, disp, emptymodes=e_modes)
        return pi2 * (self * Gsqz).traceMode(mode)
        
        
    def rotate(self, *rot):
        """
        Rotation of multimode Gaussian (beam splitting the modes).
        
        gaussian_rotate([theta, modes, [theta, modes, ...]])
        rot : rotation angle and modes for each rotation in turn, e.g.
              gaussian_rotate(G, pi/4, [0,1], arccos(sqrt(.9)), [1,2])
        """
        rot = list(rot)
        rot.reverse()
        bigbsm = sp.identity(2*self.numModes)
    
        while rot:
            angle = rot.pop()
            modes  = rot.pop()
            bsm = beamsplitter_matrix(angle, self.numModes, modes)
            bigbsm = sp.dot(bsm, bigbsm)
        
        covariance = chop(bigbsm * self.covariance * bigbsm.I)
        disp = chop(sp.dot(sp.asarray(bigbsm), self.disp))
        
        return Gaussian(covariance, disp, self.prefactor, self.emptymodes)
        
        

#####################################
# OPO specific formulas             #
####

        
# OPO correlations
XX = lambda eps, t: eps/(1-eps) * sp.exp(-(1-eps)*abs(t))
PP = lambda eps, t: -eps/(1+eps) * sp.exp(-(1+eps)*abs(t))

## mode functions
# mf0: kappa-filtered double-sided exponential with gamma=1
mf0 = lambda kap, t: sp.sqrt(kap * (1+kap) / (1+kap-4*kap**2+kap**3+kap**4)) * \
        (kap * sp.exp(-abs(t)) - sp.exp(-kap*abs(t)))
# mf1: kappa-filtered step function
mf1 = lambda kap, t: sp.sqrt(2*kap) * sp.exp(kap*t) * (1+sp.sign(-t))/2.


# analytically calculated expressions for filtered correlations
XX00 = lambda eps, kap: \
        (2*eps * ((2-eps)**2 * (1-eps + (5-3*eps) * kap + (7-eps) * kap**2) +
                  (3-eps) * ((5-2*eps) * kap**3 + kap**4))) /   \
        ((1-eps) * (2-eps)**2 * (1-eps+kap)**2 * (1+kap*(3+kap)))
XX01 = lambda eps, kap: \
        (sp.sqrt(2)*eps * (2 - 3*eps + eps**2 + 2*kap*(2-eps)**2 +
                           4*kap**2*(2-eps) + 2*kap**3) * sp.sign(-1+kap)) /   \
        ((1-eps) * (2-eps) * (1-eps+kap)**2 * sp.sqrt(1+4*(kap+kap**2)+kap**3))
XX11 = lambda eps, kap: 2*eps / ((1-eps) * (1-eps+kap))

# analytically calculated expressions for integrated mode functions,
# relevant for filtering of displacement with a narrow-band cw field
imf0 = lambda kap: (2 * (1+kap)**(3/2) * sp.sign(kap-1)) /   \
                   sp.sqrt(kap * (1 + 3*kap + kap**2))
imf1 = lambda kap: sp.sqrt(2/kap)


def buildG_singlemode(r=[.3,0], eta=[1,1], beamsplitters=[(1,0,1)], disp=None):
    numModes = len(r)
    if (len(eta) != numModes):
        print("Incorrect number of modes.")
        return None

    rotlist = []
    for bs in beamsplitters:
        rotlist.append(sp.arccos(sp.sqrt(bs[0])))
        rotlist.append(bs[1:])
        
    Mvac = sp.diag(0.5 * sp.ones(2*numModes))   
    Meta = sp.diag(sp.array(sp.sqrt(eta)).repeat(2))     

    # Get rotated disp vector
    disp = Gaussian(Mvac, disp).rotate(*rotlist).disp        
    disp = sp.dot(Meta, disp)
    
    r = sp.array(r)
    eta = sp.array(eta)
    M = 0.5 * sp.diag(sp.array([(1-eta) + eta * sp.exp(2*r), 
                                (1-eta) + eta * sp.exp(-2*r)]).T.flatten())
    M = Gaussian(M).rotate(*rotlist).covariance
    
    return Gaussian(M, disp)
    

def buildG(eps=[.3, 0], kap=5, eta=[1, 1], beamsplitters=[(1, 0, 1)],
           modetypes=[0, 1], disp=None):
    """
    Build covariance matrix based on OPO in one or more modes and our standard
    filter functions.
    eps, eta and modetypes should all be of length equal to number of modes.
    For modetypes, 0 is filtered double-sided exponential,
                   1 is filtered delta-function.
    """
    
    numModes = len(eta)
    if (len(modetypes) != numModes) or (len(eps) != numModes):
        print("Incorrect number of modes.")
        return None
    
    rotlist = []
    for bs in beamsplitters:
        rotlist.append(sp.arccos(sp.sqrt(bs[0])))
        rotlist.append(bs[1:])
        
    Mvac = sp.diag(0.5 * sp.ones(2*numModes))   
    Meta = sp.diag(sp.array(sp.sqrt(eta)).repeat(2))     

    # Get rotated, filtered disp vector
    disp = Gaussian(Mvac, disp).rotate(*rotlist).disp
    imf = [imf0(kap), imf1(kap)]
    dispFilter = sp.zeros(2*numModes)
    for i in range(numModes):
        dispFilter[2*i:2*(i+1)] = imf[modetypes[i]]
    disp = sp.dot(Meta, disp) * dispFilter
    
    # mode distribution (beamsplitting) matrix   
    # - index 1 where squeezed, otherwise 0. Take advantage of the rotation
    # feature in the Gaussian class, but need to add vacuum and subtract later
    Mcorr = []
    Mrot = []    
    
    for m,e in enumerate(eps):
        # distribution matrix
        distributionDiag = [1 if i//2==m else 0 for i in range(2*numModes)]
        rotState = Gaussian(sp.diag(distributionDiag) + Mvac).rotate(*rotlist)
        Mrot = rotState.covariance - Mvac
        
        # correlation matrix
        xcorr = [[XX00( e, kap), XX01( e, kap)],
                 [XX01( e, kap), XX11( e, kap)]]
        pcorr = [[XX00(-e, kap), XX01(-e, kap)],
                 [XX01(-e, kap), XX11(-e, kap)]]

        M = sp.zeros((2*numModes, 2*numModes))
        for i in range(numModes):
            for j in range(numModes):
                M[2*i, 2*j] = xcorr[modetypes[i]][modetypes[j]]
                M[2*i+1, 2*j+1] = pcorr[modetypes[i]][modetypes[j]]            

        M = sp.dot(Meta, sp.dot(M, Meta))
        
        Mcorr.append(sp.asarray(Mrot) * M)
    
    # create Gaussian state with the combined (distributed, filtered) correlation 
    # matrix    
    return Gaussian(Mvac + sp.array(Mcorr).sum(axis=0), disp=disp)




def buildVac(modelist=[0,1]):
    """
    Make Gaussian vacuum state for the modes listed as 1 in modelist.
    E.g. buildVac([0,1,0]) -> Wid(x0,p0) Wvac(x1,p1) Wid(x2,p2)
    """
    return Gaussian((0.5*sp.array(modelist)).repeat(2), 
            emptymodes = [m for m in range(len(modelist)) if modelist[m]==0])
            
            
            
            
def makeWig(G, APDs, HD, ret_G=False):
    """
    makeWig(eps, kap, eta, beamsplitters, modetypes, disp, APDs, HD)
    ----------------------------------------------------------------
    Calculates the conditional Wigner function for a general Gaussian state +
    on/off detection network.
    
    eps, eta, modetypes, disp should all have same length = number of modes.
    kap is the APD filtering bandwidth.
    beamsplitters should be of the form
        [(.95, 0, 1), (.5, 0, 2), (.5, 3, 2)],
    where first value is transmission, and the two next are the contributing modes.
    modetypes should be 0 (filtered dblexp) or 1 (filtered single-sided exp).
    APDs should be a list of the modes with conditioning APDs.
    HD is the output mode to be homodyned.
    
    Returns wig(x,p), already vectorized.
    """
    
#    G = buildG(eps, kap, eta, beamsplitters, modetypes, disp)    

    Gc = []
    for i in range(len(APDs)+1):
        tmp = []
        m_vacs = itertools.combinations(APDs, i)  # vacuum projection combinations
        for m_vac in m_vacs:
            m_id = list(range(G.numModes))  # m_id is the modes with identity trace out
            m_id.remove(HD)
            for m in m_vac:
                m_id.remove(m)
            tmp.append(G.vacProj(m_vac).traceMode(m_id))
        Gc.append(tmp)
        
    def wig(x,p):
        num = denom = 0
        for k, Gk in enumerate(Gc):
            for Gj in Gk:
#                former method:                
#                num += (-1)**k * Gj.wig(x,p)
                num += (-1)**k * Gj.wig_grid(x,p)
                denom += (-1)**k * Gj.norm
        return num/denom    
        
    # success probability
    prob = sum([(-1)**k * sum([g.norm for g in gg]) for k,gg in enumerate(Gc)])

#    vectorization was necessary with the former method:                
#    wig = sp.vectorize(wig)        

    if ret_G:
        return G, Gc, wig, prob
    else:
        return wig 
        
        
        
        
def makeWigOnOff(G, APD_on, APD_off, HD, ret_G=False):
    """
    makeWig(eps, kap, eta, beamsplitters, modetypes, disp, APDs, HD)
    ----------------------------------------------------------------
    Calculates the conditional Wigner function for a general Gaussian state +
    on/off detection network.
    
    eps, eta, modetypes, disp should all have same length = number of modes.
    kap is the APD filtering bandwidth.
    beamsplitters should be of the form
        [(.95, 0, 1), (.5, 0, 2), (.5, 3, 2)],
    where first value is transmission, and the two next are the contributing modes.
    modetypes should be 0 (filtered dblexp) or 1 (filtered single-sided exp).
    APDs should be a list of the modes with conditioning APDs.
    HD is the output mode to be homodyned.
    
    Returns wig(x,p), already vectorized.
    """
    
#    G = buildG(eps, kap, eta, beamsplitters, modetypes, disp)    

    Gc = []
    for i in range(len(APD_on)+1):
        tmp = []
        m_vacs = itertools.combinations(APD_on, i)  # vacuum projection combinations
        for m_vac in m_vacs:
            m_id = list(range(G.numModes))  # m_id is the modes with identity trace out
            m_id.remove(HD)
            for m in m_vac:
                m_id.remove(m)
            for m in APD_off:
                m_id.remove(m)
            tmp.append(G.vacProj(APD_off + list(m_vac)).traceMode(m_id))
        Gc.append(tmp)
        
    def wig(x,p):
        num = denom = 0
        for k, Gk in enumerate(Gc):
            for Gj in Gk:
                num += (-1)**k * Gj.wig_grid(x,p)
                denom += (-1)**k * Gj.norm
        return num/denom    

    # success probability
    prob = sum([(-1)**k * sum([g.norm for g in gg]) for k,gg in enumerate(Gc)])

    #wig = sp.vectorize(wig)        

    if ret_G:
        return G, Gc, wig, prob
    else:
        return wig 
        
        

        
        
def rhoSingleModeGaussian(covariance, disp=[0,0], N=20):
    """
    Calculate density matrix (up to N photons) for the single-mode Gaussian
    state with given covariance matrix and displacement vector.
    """
    covariance = sp.array(covariance)
    disp = sp.array(disp)
    assert covariance.shape == (2,2)
    assert disp.shape == (2,)
        
    # Using Marian & Marian, PRA 47, 4474:
    # See also Zhang & van Loock, PRA 84, 062309 (2011).
    a, b, c = covariance.take([0,3,1])  # covariance matrix entries
    a += numeps
    b += numeps
    
    A = (a+b-1)/2
    B = -((a-b)/2 + 1j*c)
    C = (disp[0] + 1j*disp[1])/sqrt(2)
    
    det = (1+A)**2 - abs(B)**2
    Aa = (A*(1+A) - abs(B)**2) / det
    Bb = B / det
    Cc = ((1+A)*C + B*C.conj()) / det
    Q0 = 1/(pi*sqrt(det)) * exp(-((1+A) * abs(C)**2 + 
                                  (B*C.conj()**2 + B.conj()*C**2)/2) / det)
                                  
    rho = sp.zeros((N+1,N+1), dtype=sp.complex128)
    
    herm_table = zeros(N+1, dtype=sp.complex128)
    hermc_table = zeros(N+1, dtype=sp.complex128)
    bin_table = zeros((N+1,N+1))
    Aa_table = zeros(N+1)
    Bb_table = zeros((N+1,N+1), dtype=sp.complex128)
    Bbc_table = zeros((N+1,N+1), dtype=sp.complex128)
    
    for m in arange(N+1):
        herm_table[m] = sf.hermite(m)(Cc / sqrt(2*Bb))[0]
        bin_table[m,:m+1] = array([sf.binom(m,k) for k in range(m+1)])
        Aa_table[m] = Aa**m
        Bb_table[m,:m+1] = array([(Bb/2)**((m-k)/2) for k in range(m+1)])
        
    hermc_table = herm_table.conj()
    Bbc_table = Bb_table.conj()
        
    
    for m in range(N+1):
        for n in range(m+1):
            k = sp.arange(min(m,n)+1)
        
            rho[m,n] = ( pi*Q0 / sqrt(fact(m) * fact(n)) *
                        fact(k) * bin_table[m,:len(k)] * bin_table[n,:len(k)] *
                        Aa_table[k] * Bb_table[m,:len(k)] * Bbc_table[n,:len(k)] *
                        herm_table[m-k] * hermc_table[n-k] ).sum() 
            
    return rho + rho.T.conj() - sp.diagflat(rho.diagonal())
    
    
if __name__=='__main__':
    from . import quantumoptics as qo
    cov = [[.625,.375],[.375,.625]]
    disp = [.5,.5]
    N = 15
    rho = rhoSingleModeGaussian(cov, disp, N)
    G = Gaussian(cov, disp)
    q1 = qo.QuantumState(rho)
    q2 = qo.QuantumStateW(G.wig_grid, [8,81], photons=N)
    print((abs(rho-q2.rho).max()))