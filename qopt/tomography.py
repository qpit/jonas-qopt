# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 21:18:10 2010

@author: Jonas

ToDo:
* consider deleting .data, .trigtimes, .mfarray after integrating (for memory)    

"""


import scipy as sp
import scipy.special as sps
import scipy.optimize
from scipy.stats import binom
from scipy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import inspect
import time

from . import lecroy

pi2 = 2*sp.pi
starttime = 0

def tic():
    global starttime
    starttime = time.perf_counter()
    return starttime
    
def toc(printtime=True):
    endtime = time.perf_counter()
    if printtime:
        print('%.2f' % (endtime - starttime), end='')
    return endtime, (endtime - starttime)
        

class HomodyneTrace:
    def __init__(self, fn='', offsetcorr='tail'):
        self.fn = fn
        self.offsetcorr = offsetcorr
        self.phase = None
        
        if 'deg' in self.fn:
            pos = self.fn.find('deg')
            if self.fn[pos-4] == '-':
                self.phase = int(self.fn[pos-4:pos])
            else:
                self.phase = int(self.fn[pos-3:pos])        
    
    
    def read_lecroy(self, report=False):
        if not self.fn:
            print('File name not set')
            return
            
        self.metadata, self.trigtimes, self.data = lecroy.read(self.fn, scale=False)
        self.segments, self.points = self.data.shape
        self.t = sp.arange(self.metadata['horiz_offset'], 
                           self.metadata['horiz_offset'] + 
                               self.points*self.metadata['horiz_interval'], 
                           step = self.metadata['horiz_interval'])
        self.segvar = self.data.var(axis=0)

            
        if report:
            print(self.fn)
            print('-' * len(self.fn))
            lecroy.pretty_metadata(self.metadata, 'main')
            print('')
        
        
    def set_modefunc(self, type, *parameters):
        mfshapes = {'doubleexp': ( lambda gamma, t0, t: 
                        sp.exp(-gamma*abs(t-t0)) ),
                    'dblexpfilt': ( lambda gamma, kappa, t0, t:
                        kappa*sp.exp(-gamma*abs(t-t0)) - 
                            gamma*sp.exp(-kappa*abs(t-t0)) ),
                    'deltafilt': ( lambda kappa, t0, t:
                        sp.exp(-kappa*abs(t-t0)) * (sp.sign(t0-t)+1)/2 )
                    }
        mfshape = mfshapes.get(type)
        
        if not mfshape:
            print(type, 'is not a valid mode function. Choose from:')
            for mf in mfshapes:
                print(mf)
            return
        
        mfargs = inspect.getargspec(mfshape)[0]
        if len(parameters) != len(mfargs)-1:
            print('The', type, 'function requires', str(len(mfargs)-1), 
                  'parameters:')
            print(mfargs[:-1])
            return
        
        self.mf = lambda t: mfshape(*(parameters + (t,)))
    
    
    def temporal_filter(self):
        if not self.metadata:
            print('Data not loaded yet')
            return
        
        if not self.mf:
            print('No mode function defined')
            return
        
        self.mfarray = self.mf(self.t)
        
        # ### NOTICE ######################
        # We subtract from all raw traces an offset given by the average
        # of all samples from the last 30% of each segment.
        # If we based the offset on the full segments, we would overcompensate
        # the offset becuase it would include the unknown quantum state.
        # This method may not always work properly, so always keep it in mind!
        if self.offsetcorr == 'tail':       
            offset = self.data[:, int(self.points*0.7):].mean()
        else:
            offset = 0
            
        # Integrate the data values with the mode function:
        # First subtract the offset from the data, then multiply all segments
        # with mfarray, finally sum across each
#        self.values = (self.mfarray * (self.data - offset)).sum(1)
        self.values = sp.dot((self.data - offset), self.mfarray)
    
    
    
    
    
class HomodyneTomogram:
    def __init__(self, fn=[], fnvac=[], offsetcorr='tail'):
        self.fn = fn
        self.fnvac = fnvac
        self.mftype = 'dblexpfilt'
        self.mfparameters = [pi2 * 4.5e06, pi2 * 25e06, 0]
        self.offsetcorr = offsetcorr
        
        
    def set_filenames(self, path, pathvac='', fnfilter='cat', fnfiltervac='vac',
                      channel=1):
        pathvac = pathvac or path

        self.fn = sp.array([os.path.join(path, f) for f in os.listdir(path)
                   if fnfilter in f and 'C'+str(channel) in f])
        self.fnvac = sp.array([os.path.join(pathvac, f) for f in 
                      os.listdir(pathvac) if fnfiltervac in f and
                      'C'+str(channel) in f])
    
    
    def load_data(self, keep_raw=False, phases=None):
        # Read all files and extract quadrature data
        traces = [HomodyneTrace(f, offsetcorr=self.offsetcorr) for f in self.fn] #map(HomodyneTrace, self.fn)
        tracesvac = [HomodyneTrace(f, offsetcorr=self.offsetcorr) for f in self.fnvac]        
        
        if len(phases) == len(traces):
            for p, tr in zip(phases, traces):
                tr.phase = p
        
        traces.sort(key=lambda tr: tr.phase)
        if sp.all([tr.phase is not None for tr in tracesvac]):
            tracesvac.sort(key=lambda tr: tr.phase)
        self.phases = [tr.phase for tr in traces]
        self.phases_unique = sorted(list(set(self.phases)))
        self.phasesvac = [tr.phase for tr in tracesvac]
        
        self.metadata = []
        tic()
        for trace in traces + tracesvac:
            trace.read_lecroy(report=False)
            trace.set_modefunc(self.mftype, *self.mfparameters)
            trace.temporal_filter()
            self.metadata.append(trace.metadata)
            print('.', end='')
        print(' read ' + str(len(traces)+len(tracesvac)) + ' files in ', end='')
        toc()
        print(' seconds.')
        self.xval = sp.array([tr.values for tr in traces])
        self.xvalvac = sp.array([tr.values for tr in tracesvac])
        self.trigtimes = sp.array([tr.trigtimes[0] for tr in traces])
        
        # Normalize traces to 0.5 vacuum variance
        vaclevel = self.xvalvac.var()
        self.xval = self.xval / sp.sqrt(vaclevel * 2.)
        self.xvalvac = self.xvalvac / sp.sqrt(vaclevel * 2.)
        
        if self.offsetcorr == 'totalmean':
            self.xval -= self.xval.mean()
            self.xvalvac -= self.xvalvac.mean()
        elif self.offsetcorr == 'vacmean':
            vacmean = self.xvalvac.mean()
            self.xval -= vacmean
            self.xvalvac -= vacmean
        
        # Set phase array of same length as xval
        # - should be extended with other routines for scanned phase
        self.thetaval = sp.outer(self.phases, sp.ones(len(self.xval[0])))
        
        # Extract segment noise variances and mode function array
        self.segvars = sp.array([tr.segvar for tr in traces])
        self.segvarsvac = sp.array([tr.segvar for tr in tracesvac])
        self.mfarray = traces[0].mfarray
        self.mfarray_norm = ( (self.mfarray - self.mfarray.mean()) *
            (self.segvars.mean(axis=0).max() - self.segvars.mean()) /
            (self.mfarray.max() - self.mfarray.mean()) + self.segvars.mean() )

        if keep_raw:
            self.traces = traces
            self.tracesvac = tracesvac
    
    def setData(self, xval, phases_unique):
        """
        Set x and theta data from artificially generated samples.
        xval: Nt x Nx sized array
        phases_unique: Nt long standard Python list of the phases
        """
        self.xval = xval
        self.phases_unique = list(phases_unique)
        self.thetaval = sp.outer(self.phases_unique, sp.ones(len(self.xval[0])))
        
    def segmentnoise(self):
        plt.figure(figsize=(7,10))
        plt.subplot(211)
        for idx, segvar in enumerate(self.segvars):
            plt.plot(segvar, color=cm.winter(idx/(len(self.segvars)-1.*0)))
        for idx, segvar in enumerate(self.segvarsvac):
            plt.plot(segvar, color=cm.autumn(idx/(len(self.segvarsvac)-1.*0)))
        
        plt.subplot(212)
        plt.plot(self.segvars.mean(axis=0), 'b--', linewidth=2)
        plt.plot(self.segvarsvac.mean(axis=0), 'r--', linewidth=2)
        plt.plot(self.mfarray_norm, 'g-')
        
    
    def variancefit(self, p0=[1.,2.,0.,2.]):
        fitfunc = lambda p, x: p[0]*sp.cos(2*sp.pi/360 * (x*2 - p[2])) + p[3]
        errfunc = lambda p, x, y: fitfunc(p,x) - y
        self.variances = self.xval.var(axis=1)
        p1, success = scipy.optimize.leastsq(errfunc, p0[:],
                                    args = (self.phases, self.variances))
        if success:
            theta = sp.arange(0,181)
            plt.figure()
            plt.plot(self.phases, self.variances, 'bo',
                     theta, fitfunc(p1, theta), 'b-')
            ax = plt.axes()
            plt.text(0.5, 0.9, 'Rotation phase: ' + '%.2f' % p1[2],
                     horizontalalignment='center',
                     transform=ax.transAxes, 
                     bbox=dict(facecolor='blue', alpha=.2))
            return p1
        
    
    def reconstruct(self, n_bins=200, n_iter=100, rho0=15, eta=1, report=True):
        """
        for eta=1 sometimes randomly ends up with NaNs in R!
        use eta=.999999.. instead
        """
        self.monitor = []
        if type(rho0) == type(0):
            rho0 = sp.identity(rho0+1)
        rho0 /= rho0.trace()
        
        n_ph = len(self.phases_unique)
        photons = len(rho0)-1
        theta = self.thetaval.flatten()
        x = self.xval.flatten()
        x_max = abs(x).max()
#        theta_edges = self.phases_unique + [self.phases_unique[-1]+1.]
        theta_edges = self.phases_unique + [360.]
        x_edges, dx = sp.linspace(-x_max, x_max, n_bins+1, retstep=True)
        hist, theta_edges, x_edges = sp.histogram2d(theta, x, 
                                        bins=[theta_edges, x_edges])
        self.hist = hist
        hist = hist.flatten() * n_bins / float(len(x))
        # self.maxent = sp.nan_to_num(hist/n_ph * sp.log(hist / float(len(x)))).sum()
        self.maxent = 0
        x_centers = (x_edges[1:] + x_edges[:-1])/2.
        
        phases_rad = 2*sp.pi/360. * sp.array(self.phases_unique)
        phase_grid = sp.tile(phases_rad[:,sp.newaxis], [1, n_bins])
        x_grid = sp.tile(x_centers, [n_ph, 1])
        
        wavefuncs = sp.zeros([n_ph, n_bins, photons + 1])+0j
        for n in sp.arange(photons + 1):
            # use xn().conj() to get <n|x> instead of <x|n>
            wavefuncs[:,:,n] = xn(phase_grid, x_grid, n).conj()
        wavefuncs = wavefuncs.reshape((n_ph * n_bins,
                                                 photons + 1))
        # wavefuncs has dimensions (n_phases * n_bins, photons+1)
        if eta < 1:
            global povm
#            povm = sp.einsum('...i,...j', wavefuncs, wavefuncs.conj())
            basepovm = sp.einsum('...i,...j', wavefuncs, wavefuncs.conj())
            povm = sp.zeros(basepovm.shape, dtype='complex')
            for k in range(photons+1):
                binomials = (lambda i,e: sp.sqrt(binom.pmf(i-k, i, e))) \
                            (sp.arange(photons+1), eta)
                shiftpovm = sp.zeros(basepovm.shape, dtype='complex')
                shiftpovm[:,k:,k:] = basepovm[:,:-k,:-k] if k>0 else basepovm
                shiftpovm *= sp.outer(binomials, binomials)
                povm += shiftpovm
                
            
            
        rho = rho0        
        for k in range(n_iter):
            #WRONG, for a long time: self.probabilities = (sp.tensordot(self.wavefuncs, rho, (1,1))
            #                      * self.wavefuncs.conj()).sum(1)
            rho_former = rho
            if eta == 1:
                probabilities = sp.sum(sp.dot(wavefuncs.conj(), rho)*
                        wavefuncs, axis=1).real * dx
                #WRONG, for a long time: R = sp.dot(hist / self.probabilities * self.wavefuncs.T, self.wavefuncs.conj())
                #changed back to original, but also changed probabilities and wavefuncs
                R = sp.dot(hist / probabilities * wavefuncs.T, 
                        wavefuncs.conj())
            else:
#                probabilities = sp.einsum('hi,ij,hj->h', wavefuncs.conj(),
#                                               rho, wavefuncs) 
#                probabilities = sp.einsum('hij,ji', povm, rho)
                probabilities = sp.tensordot(povm, rho, ([1,2], [1,0])) * dx
                R = sp.tensordot(hist/probabilities, povm, 1)
            rho = sp.dot(R, sp.dot(rho, R))
#           TRYING TO MAKE rho HERMITIAN
            
            rho = (rho + rho.T.conj())/2        
            rho /= rho.trace()
            try:
                sp.asarray_chkfinite(rho)
            except ValueError:
                print('After',k)
                
            likelihood = ((probabilities/n_ph)**hist).prod()
            if report:
	            print(k, sp.log(likelihood), 
	                  (probabilities/n_ph * sp.log(probabilities/n_ph)).sum() * len(x),
	                  self.maxent)
            elemdiff = abs(abs(rho)-abs(rho_former)).sum()
            tracedist = 0#la.svdvals(sp.asmatrix(rho) -
                          #         sp.asmatrix(rho_former)).sum()/2
            self.monitor.append([rho_former,elemdiff,tracedist,likelihood,probabilities])
        self.rho = rho
        self.wavefuncs = wavefuncs
        
    
    def report(self):
        print('Useful info: file names, trigger date+times, file number,'+
              'Wigner minimum, variances, mf parameters, ...')
              
    def __str__(self):
        return self.fn
        
        
    
def xn(theta, x, n):
    """
    x-representation of Fock states, <x|n>
    """
    return ( sp.exp(-1j * n * theta) * sps.hermite(n)(x) * sp.exp(-x*x/2.) /
        sp.sqrt(sp.sqrt(sp.pi) * 2**n * sps.gamma(n+1)) )




    
# ####################
# self test section    
    
if __name__ == '__main__':
    path = 'd:/data/091022/tora1/'
    fn_filter = 'cat'

    fnvac_filter = 'vac'
    
#    allfiles = os.listdir(path)
#    fn = []
#    fnvac = []
#    for file in allfiles:
#        if fn_filter in file:
#            fn.append(path + file)
#        elif fnvac_filter in file:
#            fnvac.append(path + file)
    

    ht = HomodyneTomogram()
    ht.set_filenames(path, fnfilter='cat')
    ht.mfparameters[2] = -94e-09
    ht.load_data()   
#    ht.segmentnoise()
#    ht.variancefit()
    
