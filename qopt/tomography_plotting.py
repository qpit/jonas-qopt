    # -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 15:20:00 2011

@author: exp
"""

#from guiqwt.plot import ImageDialog
#from guiqwt.builder import make
#
#def wigplot():
#    import guidata
#    app = guidata.qapplication()
#    
#    win = ImageWidget(app, edit=False, toolbar=True, 
#                      options = dict(show_xsection=True, show_ysection=True))
#    win.resize(600, 600)
#    
#    wig = make.image(W)
#    plot = win.get_plot()
#    plot.add_item(wig)
#    
#    win.exec_()


import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

c1 = [(0,'#000099'), (1/3.,'#FEFEFE'), (1,'#CC0000')]
c2 = [(0.,'#0047b2'), (0.1,'#0c50b7'), (0.2,'#2765c2'), (0.3,'#5889d3'), \
      (0.4,'#a2bee8'), (0.49,'#ffffff'), (0.51,'#ffffff'), (0.6,'#e8a2a2'), \
      (0.7,'#d35858'), (0.8,'#c22727'), (0.9,'#b70c0c'), (1.,'#b20000')]
cmwig1 = mpl.colors.LinearSegmentedColormap.from_list('cmwig1',c2) 




def wignerContour(XPW, type=1, colorbar=True):
    """
    Build a Wigner contour plot.
    XPW can be a QuantumState instance with X,P,W attributes defined, or it can
    be a list of X,P,W 2-D arrays.
    """
    try:
        X,P,W = XPW.X, XPW.P, XPW.W
    except AttributeError:
        X,P,W = XPW
        
    if type==0:
        im = plt.imshow(W.T, vmin=-1/sp.pi, vmax=1/sp.pi, 
                  extent=[X.min(), X.max(), P.min(), P.max()], origin='lower',
                  cmap = cmwig1)
        plt.colorbar()
        cont = plt.contour(X, P, W, levels=sp.arange(-.32, .32, .04), 
                   colors='k', alpha=.3, linewidths=0.5)
        return im, cont
    elif type==1:
        fig = plt.figure(figsize=(9,6), facecolor='w')
        ax = fig.add_axes([.05,.05,.9,.9], aspect='equal')
        dc = .02
        cont = ax.contourf(X, P, W, sp.arange(-.33, .33+dc, dc), cmap=cmwig1);
        if colorbar:
            fig.colorbar(cont, ax=ax, ticks=sp.arange(-.3,.31,.1), 
                         shrink=.7, fraction=.08)
        ax.grid(True)
        return fig
    elif type==2:
        ax = plt.gca()
        dc = .02
        cont = plt.contourf(X, P, W, sp.arange(-.33, .33+dc, dc), cmap=cmwig1);
        if colorbar:
            plt.colorbar(cont, ax=ax, ticks=sp.arange(-.3,.31,.1), 
                         shrink=.7, fraction=.08)
        plt.grid(True)
        plt.xticks(sp.arange(sp.floor(X.min()+1), sp.ceil(X.max())))
        plt.yticks(sp.arange(sp.floor(P.min()+1), sp.ceil(P.max())))
        return cont 
    elif type==3:
        fig = plt.figure(figsize=(9,6), facecolor='w')
        ax = fig.add_axes([.05,.05,.9,.9], aspect='equal')
        dc = .02
        cont = ax.contourf(X, P, W, sp.arange(-.33, .33+dc, dc), cmap=cmwig1);
        if colorbar:
            fig.colorbar(cont, ax=ax, ticks=sp.arange(-.3,.31,.1), 
                         shrink=.7, fraction=.08)
        ax.grid(True)
        return fig
        

def quadratureTrace(values, points=5000):
    #trace = plt.plot(values.flatten()[::values.size/points], 'b.')
    plt.xlim(0, points)
    plt.ylim(-6,6)
    plt.xticks([])
    trace = plt.plot(values.flatten()[::values.size/points], 'k.', 
                     alpha=1000./points)

    return trace
    
    
def numberDist(rho):
    numdist = plt.bar(sp.arange(rho.shape[0])-.4, rho.diagonal().real,
                      color='#CCCCDD');
    plt.gca().set_xlim(-.5, rho.shape[0]-.5)
    return numdist
    
    
def countRate(ht, bins=100, doplot=False):
#    def trigtimes2CountRate(trigtimes, bins=100):
#        bintimes = sp.linspace(0, trigtimes[-1], num=bins+1, endpoint=True)
#        binpoints = trigtimes.searchsorted(bintimes)
#        bincounts = binpoints[1:] - binpoints[:-1]
#        binintervals = trigtimes[binpoints[1:]] - trigtimes[binpoints[:-1]]
#        countrates = bincounts / binintervals
#        return bintimes[:-1], countrates
    countRates = []
    binTimes = []        
    for tt in ht.trigtimes:
        bintimes = sp.linspace(0, tt[-1], num=bins+1, endpoint=True)
        binpoints = tt.searchsorted(bintimes)
        bincounts = binpoints[1:] - binpoints[:-1]
        binintervals = tt[binpoints[1:]] - tt[binpoints[:-1]]
        countrates = bincounts / binintervals
        binTimes.append(bintimes[:-1])
        countRates.append(countrates)
        
    binTimes = sp.array(binTimes)
    countRates = sp.array(countRates)
        
    
    #    countRates = sp.array([trigtimes2CountRate[tt] for tt in ht.trigtimes])
    startTimes = [meta['trigger_time'] for meta in ht.metadata[:len(binTimes)]]
    startTimes = [st[3]*3600 + st[4]*60 + st[5] for st in startTimes]
    startTimes = [st - startTimes[0] for st in startTimes]
    for i,bt in enumerate(binTimes):
        bt += startTimes[i]
    
    if doplot:
        plt.figure()
        plt.plot(binTimes.flatten(), countRates.flatten(), 'k.')
        plt.plot(binTimes[:,bins/2], countRates.mean(axis=1), 'r-')
        plt.title('Average count rate: ' + str(countRates.mean()))
    return binTimes, countRates

if __name__ == "__main__":
    pass
#    wigplot()