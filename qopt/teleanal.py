# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 17:50:06 2012

@author: jsne
"""

import os
from scipy import pi, array, arange, zeros, log, sqrt, arccos, linspace, \
                  meshgrid, exp, e, cos, sin, finfo, diag, ones, sinc
from numpy import linalg as nla
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .tomography import HomodyneTomogram
from .quantumoptics import QuantumState, CVQubitStateW, QuantumStateW
from .quantumstates import rho_coh, rho_cat, rho_coqu, coqu2cat, cat2coqu
from .tomography_plotting import wignerContour, quadratureTrace, countRate
from .state_modeling import imf0, buildG, makeWig, makeWigOnOff
import qutip as qt
from functools import reduce

numeps = finfo(float).eps

# It is important to choose a wide area in phase space for the Wigner
# function sampling for wig -> rho conversion - otherwise rho get negative
# eigenvalues with dire consequences, such as fidelity > 1 with itself

defaults = dict(datadir = 'c:/lab/data',
                savedir = 'c:/Dropbox/Work/Teleportation/analysis4',
                gridparam = [8,161,8,161],
                N = 12
                )

recon_defaults_common = dict(mftype = 'dblexpfilt',
                             mfparameters = [2*pi*4.5e06, 2*pi*25e06, -70e-09],
                             offsetcorr = 'tail',
                             n_bins = 100,
                             n_iter = 100,
                             eta = .9999)
recon_defaults_tele  = dict(fnfilter = 'tele',
                            channel = 1,
                            eta = .94)
recon_defaults_sqz   = dict(fnfilter = 'tele',
                            channel = 1,
                            mfparameters = [2*pi*4.5e06, 2*pi*25e06, 70e-09],
                            offsetcorr = 'tail',
                            eta = .94)
recon_defaults_input = dict(fnfilter = 'input',
                            channel = 2,
                            offsetcorr = 'totalmean',
                            eta = .88)




class Tele:
    """
    Tele(parameters)
    ----------------
    Container for teleportation data, reconstructed state, simulations etc.
    """
    def __init__(self, setup_params={}):
        params = defaults.copy()
        params.update(setup_params)
        for key, val in params.items():
            setattr(self, key, val)
        
        
    def settings_from_string(self, varnames, settings):
        """Set the settings from strings.
        
        Args:
            varnames: tab-separated string of variable names
            settings: tab-separated string of experimental settings
            
        Returns:
            sets self.settings as a dict
        """
        sett = dict(list(zip(varnames.split(), settings.split())))
        for key, val in sett.items():
            try:
                sett[key] = float(val)
            except ValueError:
                pass
        self.settings = sett
        self.path = os.path.join(self.datadir, sett['series'])
        self.path_input = os.path.join(self.datadir, sett['input_state'])
        

    def reconstruction(self, path, params, rot):
        ht = HomodyneTomogram(offsetcorr=params['offsetcorr'])
        ht.set_filenames(path, fnfilter=params['fnfilter'], channel=params['channel'])
        ht.mftype = params['mftype']
        ht.mfparameters = params['mfparameters']
        ht.load_data()
        ht.reconstruct(n_bins=params['n_bins'], n_iter=params['n_iter'],
                       rho0=self.N, eta=params['eta'])
                       
        s = QuantumState(ht.rho)
        s = s.rotate(rot * pi/180.)
        s.buildWigner(*self.gridparam)
        
        return ht, s
                   
                   
    def reconstruct_tele(self, recon_params={}):
        params = recon_defaults_common.copy()
        params.update(recon_defaults_tele)
        params.update(recon_params)
        
        self.ht, self.s = self.reconstruction(self.path, params,
                                              self.settings['rot'])
                                              
        self.ht.trigrates = self.ht.trigtimes.shape[1] /  \
                            (self.ht.trigtimes[:,-1] - self.ht.trigtimes[:,0])
                                              
    def reconstruct_sqz(self, recon_params={}):
        params = recon_defaults_common.copy()
        params.update(recon_defaults_sqz)
        params.update(recon_params)
        
        self.hts, self.ss = self.reconstruction(self.path, params,
                                                self.settings['rot'])
                                              
    def reconstruct_input(self, recon_params={}):
        params = recon_defaults_common.copy()
        params.update(recon_defaults_input)
        params.update(recon_params)
        
        self.hti, self.si = self.reconstruction(self.path_input, params,
                                                self.settings['rot'])
                                                
                                                
#    def qutip_model(self, squeezed=False):
#        N = self.N
#        eps = self.settings['eps']
#        delta = self.settings['alphatg']    # target amplitude        
#        alpha = -delta/self.settings['Gtg'] # input amplitude
#
#        RB = 1 - self.settings['Tcat']/100  # Bob's R
#        RA = self.settings['Tbell']/100     # Alice's R (towards APD from Bob's side)
#        RE = self.settings['Rloss']/100
#        
#        splitloss = True
#        # THE FOLLOWING SHOULD BE .87, I THINK!!! .96*.95*.95
#        # outcoupling, tapping, 5% propagation
#        etaCAT = .79 # .91*.96*.95*.95 (5% prop loss before BS)
#        etaB = .96   # 4% prop loss after BS, before HD
#        etaHD = etaCAT * etaB
#        etaAPD = .1
#        
#        r = log(1 + 4*eps/((1-eps)**2 + .42**2))/2.  # effective squeezing parameter
#        
#        a = qt.destroy(N)
#        ad = qt.create(N)
#        I = qt.qeye(N)
#        
##        ssv = a * q.squeez(N, -r) * q.basis(N)
##        ssv = ssv.unit()
#        rhovac = qt.fock_dm(N)
#        
#        bs = lambda T: (arccos(sqrt(T)) * (qt.tensor([a, ad]) - qt.tensor([ad, a]))).expm()
#        VAC = lambda T: (arccos(sqrt(T)) * (qt.tensor([a, I, ad]) - qt.tensor([ad, I, a]))).expm()
#        VBC = lambda T: (arccos(sqrt(T)) * (qt.tensor([I, a, ad]) - qt.tensor([I, ad, a]))).expm()
#               
#        rhoA = qt.coherent_dm(N, alpha)
#        if squeezed:
#            rhoB = (qt.squeez(N, -r) * rhovac * qt.squeez(N, -r).dag()).unit()
#        else:
#            rhoB = (a * qt.squeez(N, -r) * rhovac * qt.squeez(N, -r).dag() * ad).unit()
#        if splitloss:
#            rhoB = qt.ptrace(bs(etaCAT) * qt.tensor([rhoB, rhovac]) * 
#                            bs(etaCAT).dag(), 0).unit()
#        rhoC = qt.fock_dm(N)
#        
#        rhoABC = VAC(1-RA) * VBC(1-RB) * qt.tensor([rhoA, rhoB, rhoC]) * \
#                 VBC(1-RB).dag() * VAC(1-RA).dag()
#        
#        poff = reduce(lambda x,y: x+y,
#                      map(lambda m: qt.fock_dm(N, m) * (1-etaAPD)**m, range(N)))
#        proj = qt.tensor([I-poff, I, I])              
#        
#        rho_on = rhoABC if squeezed else proj * rhoABC * proj.dag()
#        rho_out = qt.ptrace(rho_on, 1).unit()
#   
#        rho = rho_out.data.toarray()
#        rank = nla.matrix_rank(rho)
#        state = QuantumState(rho[:rank+1,:rank+1]).loss(etaB if splitloss else etaHD)
#        state.buildWigner(*defaults['gridparam'])
#        
#        if squeezed:
#            self.mqs = state
#        else:
#            self.mq = state
#
#
#    def gaussian_model(self):
#        N = self.N
##        eps = self.settings['eps']
#        delta = self.settings['alphatg']    # target amplitude        
#        alpha = -delta/self.settings['Gtg'] # input amplitude
#        
#        eps = [self.settings['eps'],0,0,0]
#        kap = 5
#        eta = [.87, .1, .1, 1]  # not including detection efficiency
#        beamsplitters = [(.95, 0, 1), 
#                         (self.settings['Tcat']/100., 0, 2),
#                         (self.settings['Tbell']/100., 2, 3)]
#        modetypes = [0,1,1,0] # homodyne, APD, APD
#        disp = zeros(8)
#        disp[6] = sqrt(2) * alpha / imf0(kap)
#        
#        gridparam = defaults['gridparam'] # x={-5,..,5}, p={-3,..,3}
#        
#        Go = buildG(eps, kap, eta, beamsplitters, modetypes, disp)
#        beamsplitters.pop()
#        Gi = buildG(eps, kap, eta, beamsplitters, modetypes, disp) 
#        wigo = makeWig(Go, [1,2], 0)
#        wigi = makeWig(Gi, [], 3)
#        
#        self.mg = CVQubitStateW(wigo, gridparam, N)
#        self.mgi = CVQubitStateW(wigi, gridparam, N)
#        
#    def changwoo_model(self):
#        delta = self.settings['alphatg']    # target amplitude        
#        alpha = -delta/self.settings['Gtg'] # input amplitude
#        theta = coh_theta(alpha)
#        phi = 0
#        RHD = 1 - 0.96*0.95*.91  # times 0.92 for broadband squeezing!!!!!!!!
#        RAPD = 1 - .1
#        RB = 1 - self.settings['Tcat']/100  # Bob's R
#        RA = self.settings['Tbell']/100     # Alice's R (towards APD from Bob's side)
#        RE = self.settings['Rloss']/100
#        S = Seff(self.settings['eps'])
#        
#        wig = lambda x,p: qWOut(theta, phi, RHD, RAPD, RA, RB, RE, alpha, S, x, p)
#        self.mc = CVQubitStateW(wig, defaults['gridparam'], self.N)



################################################################################
## Models ######################################################################
################################################################################

def gaussian_model(alpha, etaHD, etaAPD, RA, RB, RE, eps, RT=.05, kappa=5,
                   N=defaults['N'], gridparam=defaults['gridparam'],
                   off_detection=False):    
    """
    Gaussian model.
    Parameters: alpha, etaHD, etaAPD, RA, RB, RE, eps
    Optional: RT (tapping R), kappa, N, gridparam
    
    Returns (tele state, unconditioned state, input state)
    """
    eps = [eps,0,0,0,0]
#    eta = [etaHD, etaAPD, 1, etaAPD, 1]  # not including detection efficiency
    eta = [etaHD, etaAPD, etaAPD, etaAPD, 1]  # not including detection efficiency
    etai = [1, 1, 1, 1, 1]               # not including detection efficiency
    beamsplitters = [(1-RT, 0, 1), 
                     (1-RB, 0, 2),
                     (1-RE, 2, 4),
                     (1-RA, 2, 3)]
#    modetypes = [0,1,0,1,0] # homodyne, APD, irrelevant, APD, irrelevant
    modetypes = [0,1,1,1,0] # homodyne, APD, APD-OFF, APD, irrelevant
    modetypesi = [0,1,0,0,0] # homodyne, APD, irrelevant, homodyne, irrelevant
    disp = zeros(10)
    disp[6] = sqrt(2) * alpha / imf0(kappa) # x displacement of input, corrected
                                          # for experimental mode-function
    
    # Gaussian state for teleportation setting
    Go = buildG(eps, kappa, eta, beamsplitters, modetypes, disp)
    # Gaussian state for input homodyning setting
    Gi = buildG(eps, kappa, etai, beamsplitters[:-1], modetypesi, disp) 
    if off_detection: # ON/OFF detection in mode C?
        wigo = makeWigOnOff(Go, [1,3], [2], 0)
    else:
        wigo = makeWig(Go, [1,3], 0)  # teleported state
    wigs = makeWig(Go, [], 0)    # state without APD clicks
    wigi = makeWig(Gi, [], 3)     # input homodyned state (no APD, HD on mode 3)
    
    return (CVQubitStateW(wigo, gridparam, N),
            CVQubitStateW(wigs, gridparam, N),
            CVQubitStateW(wigi, gridparam, N))
           

def changwoo_model(alpha, etaHD, etaAPD, RA, RB, RE, r, theta=None, phi=0,
                   N=defaults['N'], gridparam=defaults['gridparam']):
    """
    Changwoo's model.
    Parameters: alpha, etaHD, etaAPD, RA, RB, RE, r
    Optional: theta, phi (for qubit, otherwise coherent), N, gridparam
    
    Returns (tele state)
    """
    if theta == None:
        theta = coh_theta(alpha)
    S = exp(2*r)
    RHD = 1-etaHD
    RAPD = 1-etaAPD
    
    wig = lambda x,p: qWOut(theta, phi, RHD, RAPD, RA, RB, RE, alpha, S, x, p)
    return CVQubitStateW(wig, gridparam, N)
    
    
def qutip_model(alpha, etaCAT, etaHD, etaAPD, RA, RB, RE, r, theta=None, phi=0,
                N=defaults['N'], gridparam=defaults['gridparam']):
    """
    Qutip model.
    Parameters: alpha, etaCAT, etaHD, etaAPD, RA, RB, RE, r
    Optional: theta, phi (for qubit, otherwise coherent), N, gridparam
    
    Returns (tele state, unconditioned state)
    """
    if not theta:
        theta = coh_theta(alpha)
        
    a = qt.destroy(N+1)
    ad = qt.create(N+1)
    I = qt.qeye(N+1)
    
    rhovac = qt.fock_dm(N+1)
    
    bs = lambda T: (arccos(sqrt(T)) * \
                     (qt.tensor([a, ad]) - qt.tensor([ad, a]))).expm()
    VAC = lambda T: (arccos(sqrt(T)) * \
                      (qt.tensor([a, I, ad]) - qt.tensor([ad, I, a]))).expm()
    VBC = lambda T: (arccos(sqrt(T)) * \
                      (qt.tensor([I, a, ad]) - qt.tensor([I, ad, a]))).expm()
    
    evencat = (qt.coherent(N+1, alpha) + qt.coherent(N+1, -alpha)).unit()
    oddcat = (qt.coherent(N+1, alpha) - qt.coherent(N+1, -alpha)).unit()
    ketA = cos(theta/2) * evencat + sin(theta/2) * exp(1j*phi) * oddcat
    rhoA = ketA * ketA.dag()
    
    rhoBsq = (qt.squeez(N+1, -r) * rhovac * qt.squeez(N+1, -r).dag()).unit()
    rhoB = (a * qt.squeez(N+1, -r) * rhovac * qt.squeez(N+1, -r).dag() * ad).unit()
    rhoBsq = qt.ptrace(bs(etaCAT) * qt.tensor([rhoBsq, rhovac]) * 
                        bs(etaCAT).dag(), 0).unit()
    rhoB = qt.ptrace(bs(etaCAT) * qt.tensor([rhoB, rhovac]) * 
                        bs(etaCAT).dag(), 0).unit()

    rhoC = rhovac
    
    rhoABCsq = VAC(1-RA) * VBC(1-RB) * qt.tensor([rhoA, rhoBsq, rhoC]) * \
               VBC(1-RB).dag() * VAC(1-RA).dag()
    rhoABC = VAC(1-RA) * VBC(1-RB) * qt.tensor([rhoA, rhoB, rhoC]) * \
             VBC(1-RB).dag() * VAC(1-RA).dag()
    
    poff = reduce(lambda x,y: x+y,
                  [qt.fock_dm(N+1, m) * (1-etaAPD)**m for m in range(N+1)])
    proj = qt.tensor([I-poff, I, I])              
    
    rho_onsq = rhoABCsq
    rho_on = proj * rhoABC * proj.dag()
    rho_outsq = qt.ptrace(rho_onsq, 1).unit()
    rho_out = qt.ptrace(rho_on, 1).unit()
   
    rhosq = rho_outsq.data.toarray() + numeps*diag(ones(N+1))
    rho = rho_out.data.toarray() + numeps*diag(ones(N+1))
    
    statesq = QuantumState(rhosq).loss(etaHD)    
    state = QuantumState(rho).loss(etaHD)
    statesq.buildWigner(*gridparam)    
    state.buildWigner(*gridparam)
    
    return state, statesq


### Helper functions ###########################################################

def coh_theta(alpha):
    """Get cat qubit theta for coherent state |alpha>"""
    return 2 * arccos(sqrt((1 + exp(-2*alpha**2))/2))

def reff(eps):
    return log(1 + 4*eps / ((1-eps)**2 + .42**2)) / 2.
    
def Seff(eps):
    return exp(2 * reff(eps))

E = e

def se(beta):
    """  From Branczyk2008, p.3  """
    return log(sqrt(2 * beta**2 + sqrt(1 + 4 * beta**4)))
    
def so(beta):
    return se(beta / sqrt(3))
    
def So(alpha):
    return exp(2 * so(alpha))
    
def Se(alpha):
    return exp(2 * se(alpha))
    
def W0(ar, ai):
    return 2/pi * exp(-2 * (ar**2 + ai**2))
    
def WC(alpha, ar, ai):
    return W0(ar - alpha, ai)


################################################################################
## Chang-Woo's model ###########################################################
################################################################################

### Coherent input #############################################################
 
def WOutUN(RHD, RAPD, RA, RB, RE, alpha, S, gammaBr, gammaBi):
    return (-2*(E)**((2*S*(gammaBi)**(2))/(-1 + RHD + RB*(-1 + RHD)*(-1 + S) - \
        RHD*S) - (2*(gammaBr)**(2))/(RHD + RB*(-1 + RHD)*(-1 + S) + S - \
        RHD*S))*(S)**(1.5)*(-(((1 + 2*RB*(-1 + RHD) - 2*RHD)*(RB*(-1 + \
        S)**(2) - (RB)**(2)*(-1 + S)**(2) + S)**(2))/((-1 + RHD + RB*(-1 + \
        RHD)*(-1 + S) - RHD*S)*(RHD + RB*(-1 + RHD)*(-1 + S) + S - RHD*S))) - \
        (4*(-1 + RB)*(-1 + RHD)*(1 + RB*(-1 + S))**(2)*(RB + S - \
        RB*S)**(2)*(gammaBi)**(2))/(-1 + RHD + RB*(-1 + RHD)*(-1 + S) - \
        RHD*S)**(2) - (4*(-1 + RB)*(-1 + RHD)*(1 + RB*(-1 + S))**(2)*(RB + S \
        - RB*S)**(2)*(gammaBr)**(2))/(RHD + RB*(-1 + RHD)*(-1 + S) + S - \
        RHD*S)**(2)))/(pi*(1 + RB*(-1 + S))**(2)*(RB + S - \
        RB*S)**(2)*sqrt((RHD + RB*(-1 + RHD)*(-1 + S) + S - RHD*S)*(1 - RHD - \
        RB*(-1 + RHD)*(-1 + S) + RHD*S))) - (16*(E)**((-2*(-1 + RA)*(-1 + \
        RAPD)*(RHD + RB*(-1 + RHD)*(-1 + S) + S - \
        RHD*S)*(alpha)**(2))/(RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) + 2*(RHD + S - RHD*S)) - (2*(RA*(-1 + RAPD)*RB*(-1 + \
        RE)*(-1 + S) - 2*S)*(gammaBi)**(2))/(-2 + RB*(RA*(-1 + RAPD)*(-1 + \
        RE) + 2*(-1 + RHD))*(-1 + S) - 2*RHD*(-1 + S)) - (4*(-1 + \
        RAPD)*sqrt((-1 + RA)*RA*(-1 + RB)*RB*(-1 + RE)*(-1 + RHD))*(-1 + \
        S)*alpha*gammaBr)/(RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + \
        S) + 2*(RHD + S - RHD*S)) - (2*(2 + RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + \
        S))*(gammaBr)**(2))/(RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 \
        + S) + 2*(RHD + S - RHD*S)))*(S)**(1.5)*(((2*RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2)*(2*(-1 + 2*RHD)*(RHD*(-1 + S) - \
        S) + RB*(-2*(-1 + RHD)*(1 + 4*RHD*(-1 + S) - 3*S) - 2*(RA)**(2)*(-1 + \
        RAPD)**(2)*(-1 + RE)*(1 + RHD*(-1 + S))*(alpha)**(2) + RA*(-1 + \
        RAPD)*(-1 + RE)*(-1 + 3*S - 2*(alpha)**(2) + 2*RAPD*(alpha)**(2) + \
        2*RHD*(-1 + S)*(-2 + (-1 + RAPD)*(alpha)**(2)))) + (RB)**(2)*(-1 + \
        S)*(4*(-1 + RHD)**(2) + (RA)**(3)*(-1 + RAPD)**(3)*(-1 + \
        RE)**(2)*(alpha)**(2) - 2*RA*(-1 + RAPD)*(-1 + RE)*(-1 + RHD)*(-2 + \
        (-1 + RAPD)*(alpha)**(2)) - (RA)**(2)*(-1 + RAPD)**(2)*(-1 + RE)*(1 - \
        (-3 + RAPD + 2*RHD)*(alpha)**(2) + RE*(-1 + (-1 + \
        RAPD)*(alpha)**(2))))))/((-2 + RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) - 2*RHD*(-1 + S))*(RB*(RA*(-1 + RAPD)*(-1 + RE) + \
        2*(-1 + RHD))*(-1 + S) + 2*(RHD + S - RHD*S))**(2)) + (4*(-1 + \
        RB)*(-1 + RHD)*(-2 + RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + \
        S))**(2)*(RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + S) + \
        2*S)**(2)*(gammaBi)**(2))/(2 - RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) + 2*RHD*(-1 + S))**(2) - (4*(-1 + RAPD)*sqrt((-1 + \
        RA)*RA*(-1 + RB)*RB*(-1 + RE)*(-1 + RHD))*(2*RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2)*alpha*gammaBr)/(RB*(RA*(-1 + \
        RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + S) + 2*(RHD + S - RHD*S))**(2) \
        + (4*(-1 + RB)*(-1 + RHD)*(2*RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + \
        S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + RE))**(2)*(-1 + \
        S)**(2) - 4*S)**(2)*(gammaBr)**(2))/(RB*(RA*(-1 + RAPD)*(-1 + RE) + \
        2*(-1 + RHD))*(-1 + S) + 2*(RHD + S - RHD*S))**(2)))/(pi*(-2 + RB*(-2 \
        + RA*(-1 + RAPD)*(-1 + RE))*(-1 + S))**(2)*sqrt(-4*(1 + RHD*(-1 + \
        S))*(RHD*(-1 + S) - S) - (RB)**(2)*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 \
        + RHD))**(2)*(-1 + S)**(2) + 2*RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + 2*RHD)*(-1 + S)**(2))*(RB*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))*(-1 + S) + 2*S)**(2))

def Psuccess(RAPD, RA, RB, RE, alpha, S):
    return 1 - (8*(S)**(1.5)*(-2 + (RA)**(3)*(-1 + RAPD)**(3)*(RB)**(2)*(-1 + \
        RE)**(2)*(-1 + S)*(alpha)**(2) - (RA)**(2)*(-1 + RAPD)**(2)*RB*(-1 + \
        RE)*(2*S*(alpha)**(2) + RB*(-1 + RE)*(-1 + S)*(-1 + (-1 + \
        RAPD)*(alpha)**(2))) + RA*(-1 + RAPD)*RB*(-1 + RE)*(3 + S*(-1 + 2*(-1 \
        + RAPD)*(alpha)**(2)))))/((E)**((2*(-1 + RA)*(-1 + \
        RAPD)*(alpha)**(2))/(2 + RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + S)))*(2 + \
        RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + S))**(2)*(RA*(-1 + RAPD)*RB*(-1 + \
        RE)*(-1 + S) - 2*S)*sqrt(2*RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + S)**(2) \
        - (RA)**(2)*(-1 + RAPD)**(2)*(RB)**(2)*(-1 + RE)**(2)*(-1 + S)**(2) + \
        4*S))

def WOut(RHD, RAPD, RA, RB, RE, alpha, S, x, p):
    return  .5*WOutUN(RHD, RAPD, RA, RB, RE, alpha, S, x/sqrt(2), p/sqrt(2)) / \
            Psuccess(RAPD, RA, RB, RE, alpha, S)
            
### Qubit input ################################################################

def AbsSqA(alpha, theta, phi):
    return ((E)**(2*(alpha)**(2))*(-cos(theta) + (E)**(2*(alpha)**(2))*(1 + \
        sqrt(1 - (E)**(-4*(alpha)**(2)))*cos(phi)*sin(theta))))/(2.*(-1 + \
        (E)**(4*(alpha)**(2))))
        
def AbsSqB(alpha, theta, phi):
    return -((E)**(2*(alpha)**(2))*(cos(theta) + (E)**(2*(alpha)**(2))*(-1 + \
        sqrt(1 - (E)**(-4*(alpha)**(2)))*cos(phi)*sin(theta))))/(2.*(-1 + \
        (E)**(4*(alpha)**(2))))
        
def ReAcB(alpha, theta, phi):
    return ((E)**(2*(alpha)**(2)) - (E)**(4*(alpha)**(2))*cos(theta))/(2 - \
        2*(E)**(4*(alpha)**(2)))
    
def ImAcB(alpha, theta, phi):
    return (sin(theta)*sin(phi))/(2.*sqrt(1 - (E)**(-4*(alpha)**(2))))
    
def V(alpha, gammar, gammai):
    return W0(gammar,gammai)/(E)**(4*1j*alpha*gammai)
    
def X(alpha, gammaBr, gammaBi):
    return V(alpha, gammaBr, gammaBi) + V(-alpha, gammaBr, gammaBi)
    
def Y(alpha, gammaBr, gammaBi):
    return -1j * (V(alpha, gammaBr, gammaBi) + V(-alpha, gammaBr, gammaBi))
    
def WSCS(theta, phi, alpha, gammaBr, gammaBi):
    return AbsSqB(alpha,theta,phi)*WC(-alpha,gammaBr,gammaBi) + \
        AbsSqA(alpha,theta,phi)*WC(alpha,gammaBr,gammaBi) + \
        ReAcB(alpha,theta,phi)*X(alpha,gammaBr,gammaBi) + \
        ImAcB(alpha,theta,phi)*Y(alpha,gammaBr,gammaBi)
        
def WCUN(RHD, RAPD, RA, RB, RE, alpha, S, gammaBr, gammaBi):
    return (2*(E)**((2*S*(gammaBi)**(2))/(-1 + RHD + RB*(-1 + RHD)*(-1 + S) - \
        RHD*S) - (2*(gammaBr)**(2))/(RHD + RB*(-1 + RHD)*(-1 + S) + S - \
        RHD*S))*(S)**(1.5)*(((1 + 2*RB*(-1 + RHD) - 2*RHD)*(RB*(-1 + S)**(2) \
        - (RB)**(2)*(-1 + S)**(2) + S)**(2))/((-1 + RHD + RB*(-1 + RHD)*(-1 + \
        S) - RHD*S)*(RHD + RB*(-1 + RHD)*(-1 + S) + S - RHD*S)) + (4*(-1 + \
        RB)*(-1 + RHD)*(1 + RB*(-1 + S))**(2)*(RB + S - \
        RB*S)**(2)*(gammaBi)**(2))/(-1 + RHD + RB*(-1 + RHD)*(-1 + S) - \
        RHD*S)**(2) + (4*(-1 + RB)*(-1 + RHD)*(1 + RB*(-1 + S))**(2)*(RB + S \
        - RB*S)**(2)*(gammaBr)**(2))/(RHD + RB*(-1 + RHD)*(-1 + S) + S - \
        RHD*S)**(2)))/(pi*(1 + RB*(-1 + S))**(2)*(RB + S - \
        RB*S)**(2)*sqrt((RHD + RB*(-1 + RHD)*(-1 + S) + S - RHD*S)*(1 - RHD - \
        RB*(-1 + RHD)*(-1 + S) + RHD*S))) - (16*(E)**((-2*(-1 + RA)*(-1 + \
        RAPD)*(RHD + RB*(-1 + RHD)*(-1 + S) + S - \
        RHD*S)*(alpha)**(2))/(RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) + 2*(RHD + S - RHD*S)) - (2*(RA*(-1 + RAPD)*RB*(-1 + \
        RE)*(-1 + S) - 2*S)*(gammaBi)**(2))/(-2 + RB*(RA*(-1 + RAPD)*(-1 + \
        RE) + 2*(-1 + RHD))*(-1 + S) - 2*RHD*(-1 + S)) - (4*(-1 + \
        RAPD)*sqrt((-1 + RA)*RA*(-1 + RB)*RB*(-1 + RE)*(-1 + RHD))*(-1 + \
        S)*alpha*gammaBr)/(RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + \
        S) + 2*(RHD + S - RHD*S)) - (2*(2 + RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + \
        S))*(gammaBr)**(2))/(RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 \
        + S) + 2*(RHD + S - RHD*S)))*(S)**(1.5)*(((2*RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2)*(2*(-1 + 2*RHD)*(RHD*(-1 + S) - \
        S) + RB*(-2*(-1 + RHD)*(1 + 4*RHD*(-1 + S) - 3*S) - 2*(RA)**(2)*(-1 + \
        RAPD)**(2)*(-1 + RE)*(1 + RHD*(-1 + S))*(alpha)**(2) + RA*(-1 + \
        RAPD)*(-1 + RE)*(-1 + 3*S - 2*(alpha)**(2) + 2*RAPD*(alpha)**(2) + \
        2*RHD*(-1 + S)*(-2 + (-1 + RAPD)*(alpha)**(2)))) + (RB)**(2)*(-1 + \
        S)*(4*(-1 + RHD)**(2) + (RA)**(3)*(-1 + RAPD)**(3)*(-1 + \
        RE)**(2)*(alpha)**(2) - 2*RA*(-1 + RAPD)*(-1 + RE)*(-1 + RHD)*(-2 + \
        (-1 + RAPD)*(alpha)**(2)) - (RA)**(2)*(-1 + RAPD)**(2)*(-1 + RE)*(1 - \
        (-3 + RAPD + 2*RHD)*(alpha)**(2) + RE*(-1 + (-1 + \
        RAPD)*(alpha)**(2))))))/((-2 + RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) - 2*RHD*(-1 + S))*(RB*(RA*(-1 + RAPD)*(-1 + RE) + \
        2*(-1 + RHD))*(-1 + S) + 2*(RHD + S - RHD*S))**(2)) + (4*(-1 + \
        RB)*(-1 + RHD)*(-2 + RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + \
        S))**(2)*(RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + S) + \
        2*S)**(2)*(gammaBi)**(2))/(2 - RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) + 2*RHD*(-1 + S))**(2) - (4*(-1 + RAPD)*sqrt((-1 + \
        RA)*RA*(-1 + RB)*RB*(-1 + RE)*(-1 + RHD))*(2*RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2)*alpha*gammaBr)/(RB*(RA*(-1 + \
        RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + S) + 2*(RHD + S - RHD*S))**(2) \
        + (4*(-1 + RB)*(-1 + RHD)*(2*RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + \
        S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + RE))**(2)*(-1 + \
        S)**(2) - 4*S)**(2)*(gammaBr)**(2))/(RB*(RA*(-1 + RAPD)*(-1 + RE) + \
        2*(-1 + RHD))*(-1 + S) + 2*(RHD + S - RHD*S))**(2)))/(pi*(-2 + RB*(-2 \
        + RA*(-1 + RAPD)*(-1 + RE))*(-1 + S))**(2)*sqrt(-4*(1 + RHD*(-1 + \
        S))*(RHD*(-1 + S) - S) - (RB)**(2)*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 \
        + RHD))**(2)*(-1 + S)**(2) + 2*RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + 2*RHD)*(-1 + S)**(2))*(RB*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))*(-1 + S) + 2*S)**(2))
        
def XUN(RHD, RAPD, RA, RB, RE, alpha, S, gammaBr, gammaBi):
    return (-4*(E)**(-2*(alpha)**(2) + (2*S*(gammaBi)**(2))/(-1 + RHD + RB*(-1 \
        + RHD)*(-1 + S) - RHD*S) - (2*(gammaBr)**(2))/(RHD + RB*(-1 + \
        RHD)*(-1 + S) + S - RHD*S))*(S)**(1.5)*(-(((1 + 2*RB*(-1 + RHD) - \
        2*RHD)*(RB*(-1 + S)**(2) - (RB)**(2)*(-1 + S)**(2) + S)**(2))/((-1 + \
        RHD + RB*(-1 + RHD)*(-1 + S) - RHD*S)*(RHD + RB*(-1 + RHD)*(-1 + S) + \
        S - RHD*S))) - (4*(-1 + RB)*(-1 + RHD)*(1 + RB*(-1 + S))**(2)*(RB + S \
        - RB*S)**(2)*(gammaBi)**(2))/(-1 + RHD + RB*(-1 + RHD)*(-1 + S) - \
        RHD*S)**(2) - (4*(-1 + RB)*(-1 + RHD)*(1 + RB*(-1 + S))**(2)*(RB + S \
        - RB*S)**(2)*(gammaBr)**(2))/(RHD + RB*(-1 + RHD)*(-1 + S) + S - \
        RHD*S)**(2)))/(pi*(1 + RB*(-1 + S))**(2)*(RB + S - \
        RB*S)**(2)*sqrt((RHD + RB*(-1 + RHD)*(-1 + S) + S - RHD*S)*(1 - RHD - \
        RB*(-1 + RHD)*(-1 + S) + RHD*S))) + (32*(E)**((-2*(RA*(-1 + RAPD)*(1 \
        + RB*(RE - RHD)*(-1 + S) + RHD*(-1 + S)) + (1 + RAPD)*(-1 + RHD + \
        RB*(-1 + RHD)*(-1 + S) - RHD*S))*(alpha)**(2))/(-2 + RB*(RA*(-1 + \
        RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + S) - 2*RHD*(-1 + S)) - \
        (2*(RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + S) - 2*S)*(gammaBi)**(2))/(-2 + \
        RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + S) - 2*RHD*(-1 + \
        S)) - (2*(2 + RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + \
        S))*(gammaBr)**(2))/(RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 \
        + S) + 2*(RHD + S - RHD*S)))*(S)**(1.5)*((-(((1 + RB*(RA*(-1 + \
        RAPD)*(-1 + RE) + 2*(-1 + RHD)) - 2*RHD)*(2*RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2))/((-2 + RB*(RA*(-1 + RAPD)*(-1 + \
        RE) + 2*(-1 + RHD))*(-1 + S) - 2*RHD*(-1 + S))*(RB*(RA*(-1 + \
        RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + S) + 2*(RHD + S - RHD*S)))) + \
        ((-1 + RA)*RA*(-1 + RAPD)**(2)*RB*(-1 + RE)*(2*RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2)*(alpha)**(2))/(2 - RB*(RA*(-1 + \
        RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + S) + 2*RHD*(-1 + S))**(2) - \
        (4*(-1 + RB)*(-1 + RHD)*(2*RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + \
        S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + RE))**(2)*(-1 + \
        S)**(2) - 4*S)**(2)*(gammaBi)**(2))/(2 - RB*(RA*(-1 + RAPD)*(-1 + RE) \
        + 2*(-1 + RHD))*(-1 + S) + 2*RHD*(-1 + S))**(2) - (4*(-1 + RB)*(-1 + \
        RHD)*(-2 + RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + S))**(2)*(RB*(-2 \
        + RA*(-1 + RAPD)*(-1 + RE))*(-1 + S) + \
        2*S)**(2)*(gammaBr)**(2))/(RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) + 2*(RHD + S - RHD*S))**(2))*cos((4*(-1 + \
        RAPD)*sqrt((-1 + RA)*RA*(-1 + RB)*RB*(-1 + RE)*(-1 + RHD))*(-1 + \
        S)*alpha*gammaBi)/(-2 + RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) - 2*RHD*(-1 + S))) + (4*(-1 + RAPD)*sqrt((-1 + \
        RA)*RA*(-1 + RB)*RB*(-1 + RE)*(-1 + RHD))*(2*RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2)*alpha*gammaBi*sin((4*(-1 + \
        RAPD)*sqrt((-1 + RA)*RA*(-1 + RB)*RB*(-1 + RE)*(-1 + RHD))*(-1 + \
        S)*alpha*gammaBi)/(-2 + RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) - 2*RHD*(-1 + S))))/(2 - RB*(RA*(-1 + RAPD)*(-1 + RE) \
        + 2*(-1 + RHD))*(-1 + S) + 2*RHD*(-1 + S))**(2)))/(pi*(-2 + RB*(-2 + \
        RA*(-1 + RAPD)*(-1 + RE))*(-1 + S))**(2)*sqrt(-4*(1 + RHD*(-1 + \
        S))*(RHD*(-1 + S) - S) - (RB)**(2)*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 \
        + RHD))**(2)*(-1 + S)**(2) + 2*RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + 2*RHD)*(-1 + S)**(2))*(RB*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))*(-1 + S) + 2*S)**(2))
        
def YUN(RHD, RAPD, RA, RB, RE, alpha, S, gammaBr, gammaBi):
    return (32*(E)**((-2*(RA*(-1 + RAPD)*(1 + RB*(RE - RHD)*(-1 + S) + RHD*(-1 \
        + S)) + (1 + RAPD)*(-1 + RHD + RB*(-1 + RHD)*(-1 + S) - \
        RHD*S))*(alpha)**(2))/(-2 + RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) - 2*RHD*(-1 + S)) - (2*(RA*(-1 + RAPD)*RB*(-1 + \
        RE)*(-1 + S) - 2*S)*(gammaBi)**(2))/(-2 + RB*(RA*(-1 + RAPD)*(-1 + \
        RE) + 2*(-1 + RHD))*(-1 + S) - 2*RHD*(-1 + S)) - (2*(2 + RA*(-1 + \
        RAPD)*RB*(-1 + RE)*(-1 + S))*(gammaBr)**(2))/(RB*(RA*(-1 + RAPD)*(-1 \
        + RE) + 2*(-1 + RHD))*(-1 + S) + 2*(RHD + S - \
        RHD*S)))*(S)**(1.5)*((-4*(-1 + RAPD)*sqrt((-1 + RA)*RA*(-1 + \
        RB)*RB*(-1 + RE)*(-1 + RHD))*(2*RB*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2)*alpha*gammaBi*cos((4*(-1 + \
        RAPD)*sqrt((-1 + RA)*RA*(-1 + RB)*RB*(-1 + RE)*(-1 + RHD))*(-1 + \
        S)*alpha*gammaBi)/(-2 + RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) - 2*RHD*(-1 + S))))/(2 - RB*(RA*(-1 + RAPD)*(-1 + RE) \
        + 2*(-1 + RHD))*(-1 + S) + 2*RHD*(-1 + S))**(2) + (-(((1 + RB*(RA*(-1 \
        + RAPD)*(-1 + RE) + 2*(-1 + RHD)) - 2*RHD)*(2*RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2))/((-2 + RB*(RA*(-1 + RAPD)*(-1 + \
        RE) + 2*(-1 + RHD))*(-1 + S) - 2*RHD*(-1 + S))*(RB*(RA*(-1 + \
        RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + S) + 2*(RHD + S - RHD*S)))) + \
        ((-1 + RA)*RA*(-1 + RAPD)**(2)*RB*(-1 + RE)*(2*RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2)*(alpha)**(2))/(2 - RB*(RA*(-1 + \
        RAPD)*(-1 + RE) + 2*(-1 + RHD))*(-1 + S) + 2*RHD*(-1 + S))**(2) - \
        (4*(-1 + RB)*(-1 + RHD)*(2*RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + \
        S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + RE))**(2)*(-1 + \
        S)**(2) - 4*S)**(2)*(gammaBi)**(2))/(2 - RB*(RA*(-1 + RAPD)*(-1 + RE) \
        + 2*(-1 + RHD))*(-1 + S) + 2*RHD*(-1 + S))**(2) - (4*(-1 + RB)*(-1 + \
        RHD)*(-2 + RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + S))**(2)*(RB*(-2 \
        + RA*(-1 + RAPD)*(-1 + RE))*(-1 + S) + \
        2*S)**(2)*(gammaBr)**(2))/(RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) + 2*(RHD + S - RHD*S))**(2))*sin((4*(-1 + \
        RAPD)*sqrt((-1 + RA)*RA*(-1 + RB)*RB*(-1 + RE)*(-1 + RHD))*(-1 + \
        S)*alpha*gammaBi)/(-2 + RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + S) - 2*RHD*(-1 + S)))))/(pi*(-2 + RB*(-2 + RA*(-1 + \
        RAPD)*(-1 + RE))*(-1 + S))**(2)*sqrt(-4*(1 + RHD*(-1 + S))*(RHD*(-1 + \
        S) - S) - (RB)**(2)*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))**(2)*(-1 + S)**(2) + 2*RB*(RA*(-1 + RAPD)*(-1 + RE) + 2*(-1 + \
        RHD))*(-1 + 2*RHD)*(-1 + S)**(2))*(RB*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))*(-1 + S) + 2*S)**(2))
        
def qWOutUN(theta, phi, RHD, RAPD, RA, RB, RE, alpha, S, gammaBr, gammaBi):
    return AbsSqB(alpha,theta,phi)*WCUN(RHD,RAPD,RA,RB,RE,-alpha,S,gammaBr,\
        gammaBi) + \
        AbsSqA(alpha,theta,phi)*WCUN(RHD,RAPD,RA,RB,RE,alpha,S,gammaBr,\
        gammaBi) + \
        ReAcB(alpha,theta,phi)*XUN(RHD,RAPD,RA,RB,RE,alpha,S,gammaBr,gammaBi) \
        + ImAcB(alpha,theta,phi)*YUN(RHD,RAPD,RA,RB,RE,alpha,S,gammaBr,\
        gammaBi)
        
def PC(RAPD, RA, RB, RE, alpha, S):
    return 1 - (8*(S)**(1.5)*(-2 + (RA)**(3)*(-1 + RAPD)**(3)*(RB)**(2)*(-1 + \
        RE)**(2)*(-1 + S)*(alpha)**(2) - (RA)**(2)*(-1 + RAPD)**(2)*RB*(-1 + \
        RE)*(2*S*(alpha)**(2) + RB*(-1 + RE)*(-1 + S)*(-1 + (-1 + \
        RAPD)*(alpha)**(2))) + RA*(-1 + RAPD)*RB*(-1 + RE)*(3 + S*(-1 + 2*(-1 \
        + RAPD)*(alpha)**(2)))))/((E)**((2*(-1 + RA)*(-1 + \
        RAPD)*(alpha)**(2))/(2 + RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + S)))*(2 + \
        RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + S))**(2)*(RA*(-1 + RAPD)*RB*(-1 + \
        RE)*(-1 + S) - 2*S)*sqrt(2*RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + S)**(2) \
        - (RA)**(2)*(-1 + RAPD)**(2)*(RB)**(2)*(-1 + RE)**(2)*(-1 + S)**(2) + \
        4*S))
        
def PX(RAPD, RA, RB, RE, alpha, S):
    return 2/(E)**(2*(alpha)**(2)) + (16*(S)**(1.5)*(-(((-1 + RA*(-1 + \
        RAPD)*RB*(-1 + RE))*(2*RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + \
        S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + RE))**(2)*(-1 + \
        S)**(2) - 4*S)**(2))/((2 + RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + \
        S))*(RA*(-1 + RAPD)*RB*(-1 + RE)*(-1 + S) - 2*S))) + ((-1 + \
        RA)*RA*(-1 + RAPD)**(2)*RB*(-1 + RE)*(2*RB*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))*(-1 + S)**(2) + (RB)**(2)*(-2 + RA*(-1 + RAPD)*(-1 + \
        RE))**(2)*(-1 + S)**(2) - 4*S)**(2)*(alpha)**(2))/(RA*(-1 + \
        RAPD)*RB*(-1 + RE)*(-1 + S) - 2*S)**(2)))/((E)**((2*((-1 - RAPD)*S + \
        RA*(-1 + RAPD)*(RB*(-1 + RE)*(-1 + S) + S))*(alpha)**(2))/(RA*(-1 + \
        RAPD)*RB*(-1 + RE)*(-1 + S) - 2*S))*(-2 + RB*(-2 + RA*(-1 + RAPD)*(-1 \
        + RE))*(-1 + S))**(2)*(RB*(-2 + RA*(-1 + RAPD)*(-1 + RE))*(-1 + S) + \
        2*S)**(2)*sqrt(-(RA*(-1 + RAPD)*RB*(-2 + RA*(-1 + RAPD)*RB*(-1 + \
        RE))*(-1 + RE)*(-1 + S)**(2)) + 4*S))
        
def PY(RAPD, RA, RB, RE, alpha, S):
    return 0
    
def qPsuccess(theta, phi, RAPD, RA, RB, RE, alpha, S):
    return AbsSqB(alpha,theta,phi)*PC(RAPD,RA,RB,RE,-alpha,S) + \
        AbsSqA(alpha,theta,phi)*PC(RAPD,RA,RB,RE,alpha,S) + \
        ImAcB(alpha,theta,phi)*PY(RAPD,RA,RB,RE,alpha,S) + \
        PX(RAPD,RA,RB,RE,alpha,S)*ReAcB(alpha,theta,phi)
        
def qWOut(theta, phi, RHD, RAPD, RA, RB, RE, alpha, S, x, p):
    return .5*qWOutUN(theta, phi, RHD, RAPD, RA, RB, RE, alpha, S, 
                      x/sqrt(2), p/sqrt(2)) /  \
           qPsuccess(theta, phi, RAPD, RA, RB, RE, alpha, S)