#!/usr/bin/env python
from __future__ import print_function, division
import sys
import psrchive
import numpy as np
import configparser
from numba import njit
from numba.typed import List
from rvm import rvm
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import pypolychord
#import time
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, LogUniformPrior, GaussianPrior
import warnings
warnings.filterwarnings("ignore")
import gc

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0
    pass

Period = 5.54

def dumper(live, dead, logweights, logZ, logZerr):
    print("LogZ: ", logZ)

@njit(fastmath=False)
def func(delta, eta, tau, tau1, epsilon, epsilon1, u1, u2, u3, epst):
    delta_1,delta_2,delta_3=delta
    eqs=[-(1/tau+eta/tau)*delta_1+(1+epsilon)*u3*delta_2-u2*delta_3-u2*u3,
         -(1+epsilon)*u3*delta_1-(1/tau+eta/tau)*delta_2+u1*delta_3+u3*u1,
         u2*delta_1-u1*delta_2-(1/tau/(1+epsilon)+eta/tau)*delta_3 + \
         epst/tau1/epsilon/(1+epsilon)*u3]
    return eqs
    
@njit(fastmath=False)
def f2(epsilon, tau, tau1, delta_1, delta_2, delta_3, u3, epst):
    dudt=np.zeros(4)
    dudt[0]=-epsilon/tau*delta_1
    dudt[1]=-epsilon/tau*delta_2
    dudt[2]=-epsilon/tau/(1+epsilon)*delta_3 + epst*1/tau1/(1+epsilon)*u3
    dudt[3]=-epsilon*u3
    return dudt

@njit(fastmath=False)
def getLbetas(cube, u1, u2, u3, psi, nQ, nU, xm, Qm, Um, mjds, mjd_idx, have_EFAC, nEFAC, rcvr_lut):
    nmpar = 10
    nfiles = len(xm)
    zeta = np.deg2rad(cube[0])
    ##theta0 = np.deg2rad(cube[1]) # the initial wobble angle
    chi = np.deg2rad(cube[2]) # the angle between the magnetic dipole and symmetric axis
    Phi0 =  np.deg2rad(cube[3]) # the initial phase of the precession
    epsilon0 = cube[4]
    epsilon1 = cube[5] # the initial ellipticity
    ##tau = cube[6] # the timescale for frictional coupling
    tau1 = cube[7]*86400 # the damping timescale of the ellipticity
    #eta = cube[8] # eta: the ratio between MoI of the crust and the core
    t0 = cube[9]*86400 # the beginning time of the precession
    phi0s = np.deg2rad(cube[nmpar:nmpar+nfiles])
    psi0s = np.deg2rad(cube[nmpar+nfiles:nmpar+2*nfiles])
    #mjds = mjds * 86400

    chi2 = 0
    logdet = 0
    betas = np.zeros(nfiles, dtype=np.float64)

    for ii in range(nfiles):
        jj = int(mjd_idx[ii])
        t = mjds[jj] - t0
        epsilon=epsilon0+epsilon1 * np.e**(-t/tau1)
        #print(t, epsilon)
        w1 = u1[ii+1]*np.cos(psi[ii+1])+u2[ii+1]*np.sin(psi[ii+1]) ## The first element is for MJD=T0
        w2 = u2[ii+1]*np.cos(psi[ii+1])-u1[ii+1]*np.sin(psi[ii+1])
        w3 = u3[ii+1]

        Psi=np.arctan2(w1,w2)
        theta=np.arccos((1+epsilon)*w3/ (w1**2 + w2**2 + w3**2 * (1+epsilon)**2 )**0.5)
        #print(epsilon, tau, eta, theta)
        # the evolution of magnetic inclination angle
        alpha=np.arccos(np.sin(theta)*np.sin(Psi)*np.sin(chi)+np.cos(theta)*np.cos(chi))
        # the evolution of impact parameter beta
        beta = zeta - alpha
        #print(ii, jj,mjds[jj]/86400.,  u1[ii], u2[ii], u3[ii],  psi[ii], np.rad2deg(beta))
        betas[jj:jj+1] = beta
        phi0 = phi0s[jj]
        psi0 = psi0s[jj]
        
        if have_EFAC:
            EFAC = cube[nmpar+2*nfiles+rcvr_lut[jj]]
        else:
            EFAC = 1.
            
        nQ2 = nQ[jj]*nQ[jj] * EFAC*EFAC
        nU2 = nU[jj]*nU[jj] * EFAC*EFAC
            
        # Compute the modelled PA
        alpha = zeta-beta
        sin_al = np.sin(alpha)
        xp = xm[jj]-phi0
        
        argx = np.cos(alpha)*np.sin(zeta) - sin_al*np.cos(zeta)*np.cos(xp)
        argy =  sin_al * np.sin(xp)
        
        PA2 = 2*(-np.arctan(argy/argx) + psi0)
        cos2PA = np.cos(PA2)
        sin2PA = np.sin(PA2)
        
        L = (Qm[jj] * cos2PA/nQ2 + Um[jj] * sin2PA/nU2) / (cos2PA*cos2PA/nQ2 + sin2PA*sin2PA/nU2) * np.exp(1j*PA2)
        
        chi2 += np.sum((Qm[jj]-np.real(L))**2 / nQ2 + (Um[jj]-np.imag(L))**2 / nU2)
        logdet += len(Qm[jj]) * (np.log(nQ2) + np.log(nU2))
    return -0.5*(chi2+ logdet), 0
        
def dudt(t,u,epsilon0,epsilon1,tau,tau1,eta):

    u1, u2, u3, gamma =u
    epsilon=epsilon0+epsilon1 * np.e**(-t/tau1)
    epst = epsilon1 * np.e**(-t/tau1)
    
    delta_1,delta_2,delta_3=fsolve(func,[0,0,0],xtol=1e-11, args=(eta, tau, tau1, epsilon, epsilon1, u1, u2, u3, epst))
    return f2(epsilon, tau, tau1, delta_1, delta_2, delta_3, u3, epst)

@njit(fastmath=False)
def inipar(cube, mjds):
    zeta = np.deg2rad(cube[0])
    theta0 = np.deg2rad(cube[1]) # the initial wobble angle
    chi = np.deg2rad(cube[2])
    Phi0 =  np.deg2rad(cube[3]) # the initial phase of the precession
    epsilon0 = cube[4]
    epsilon1 = cube[5] # the initial ellipticity
    tau = cube[6] # the frictional time scale
    tau1 = cube[7]*86400 # the damping timescale of the ellipticity
    eta = cube[8] # the ratio between MoI of the crust and the core
    t0 = cube[9]*86400 # the beginning time of the precession
    mjds = mjds * 86400

    omega0=2*np.pi/Period
    a=omega0*np.sin(theta0)
    b=omega0*np.cos(theta0)
    omega10=a*np.cos(Phi0)
    omega20=a*np.sin(Phi0)
    omega30=b
    return zeta, theta0, chi, Phi0, epsilon0, epsilon1, tau, tau1, eta, t0, mjds, omega0, omega10, omega20, omega30


def get_L(cube, nQ, nU, xm, Qm, Um, mjds, mjd_idx, have_EFAC, nEFAC, rcvr_lut):

    zeta, theta0, chi, Phi0, epsilon0,epsilon1, tau, tau1, eta, t0, mjds, omega0, omega10, omega20, omega30 = inipar(cube, mjds)
    if t0>58460*86400:
        return -2e10,0
    
    u0=[omega10,omega20,omega30,0]
    #epsilon=epsilon1 * np.e**(-t/tau1)

    dt = mjds[mjd_idx] - t0

    dt = np.append(0, dt)
    
    try:
        sol=solve_ivp(dudt, [0,dt[-1]], u0, t_eval=dt, args=(epsilon0,epsilon1, tau, tau1, eta), rtol=1e-5, atol=1e-8, method='DOP853')
    except:
        print("boundaries 0 - %f"%dt[-1])
        print("dt = ", dt)
        raise
    u1=sol.y[0]
    u2=sol.y[1]
    u3=sol.y[2]
    psi=sol.y[3]

    chilogdet, betas = getLbetas(cube, u1, u2, u3, psi, nQ, nU, xm, Qm, Um, mjds, mjd_idx, have_EFAC, nEFAC, rcvr_lut)

    return chilogdet, betas
        
class Precessnest():
    def __init__(self, filenames, sig=5, have_EFAC=False, config = None):
    
        self.nI = np.array([])
        self.nQ = np.array([])
        self.nU = np.array([])
        self.nbin = np.array([])
        self.MJDs = np.array([])
        self.xm = List()
        self.Qm = List()
        self.Um = List()
        self.rcvrs = list()

        self.nfiles = len(filenames)
        self.labels = []
        self.have_EFAC = have_EFAC
        
        for filename in filenames:
            self.get_data(filename, sig=sig)
        
        #set_rcvrs = set(self.rcvrs)
        #print(set_rcvrs)
        self.MJD_idx = np.argsort(self.MJDs)
        set_rcvrs = list(dict.fromkeys(self.rcvrs))
        self.nEFAC = len(set_rcvrs)
        rcvr = np.array(set_rcvrs)
        self.rcvrs = np.array(self.rcvrs)

        index = np.argsort(rcvr)
        sorted_x = rcvr[index]
        sorted_index = np.searchsorted(sorted_x, self.rcvrs)

        self.rcvr_lut = np.take(index, sorted_index, mode="clip")

        #print(self.rcvr_lut)
        
        # Check if we have to exclude phase range from the data
        if config.has_section('exc_phases'):
            self.exc_phs(config['exc_phases'])

        self.set_pZeta(config['zeta'])
        self.set_pBeta(config['beta'])
        self.set_pPhi0(config['phi'])
        self.set_pPsi0(config['psi'])         
        
        for ii in range(self.nfiles):
            self.xm[ii] = self.xm[ii].compressed()
            self.Qm[ii] = self.Qm[ii].compressed()
            self.Um[ii] = self.Um[ii].compressed()

            if rank==0:
                pfo = open("Profile_%d-PA.log"%ii, 'w')
                for x,PA,PAe in zip(np.rad2deg(self.xm[ii]), np.rad2deg(0.5*np.arctan2(self.Um[ii],self.Qm[ii])), 28.65*self.nI[ii]/(self.Um[ii]**2 + self.Qm[ii]**2)**.5 ):
                    pfo.write("%f %f %f\n"%(x, PA, PAe))
                pfo.close()
                del pfo
       
        self.set_labels()

        gc.collect()

    def get_nEFAC(self):
        return self.nEFAC

    def set_pZeta(self, pZe):
        for item in pZe.items():
            key = item[0]; val=item[1]

        xval = np.array(val.rstrip().split(';'))            
        val = xval.astype(float)
        self.pZe=(val[0],val[1])
        
    def __set_range(self, c):
        tmp = np.zeros((2, self.nfiles))
        for iprof,key in enumerate(c.keys()):            
            xval = np.array(c[key].rstrip().split(';'))            
            val = xval.astype(float)
            tmp[0,iprof] = val[0]; tmp[1,iprof] = val[1]
            
            if iprof+1 == self.nfiles:
                break
        return tmp       
        
    def set_pBeta(self, pBe):
        
        # Check if we have the right number of inputs vs number of files
        if len(pBe) < self.nfiles:
            raise ValueError("Number of Beta priors in config file (%d) does not match the number of profiles (%d)"%(len(pBe), self.nfiles))  
            
        self.pBe = self.__set_range(pBe)
        
    def set_pPhi0(self, pPh):
        # Check if we have the right number of inputs vs number of files
        if len(pPh) < self.nfiles:
            raise ValueError("Number of Phi0 priors in config file (%d) does not match the number of profiles (%d)"%(len(pBe), self.nfiles))  
            
        # For each entry in config file for phase range exclusion
        self.pPh = self.__set_range(pPh)
                
    def set_pPsi0(self, pPs):
        # Check if we have the right number of inputs vs number of files
        if len(pPs) < self.nfiles:
            raise ValueError("Number of Psi0 priors in config file (%d) does not match the number of profiles (%d)"%(len(pBe), self.nfiles))  
            
        # For each entry in config file for phase range exclusion
        self.pPs = self.__set_range(pPs)
        
    def set_labels(self):
        self.labels.extend(["zeta"])
        self.labels.extend(["theta_0"])
        self.labels.extend(["chi"])
        self.labels.extend(["Phi_0"])
        self.labels.extend(["epsilon_0"])
        self.labels.extend(["epsilon_1"])
        self.labels.extend(["tau_0"])
        self.labels.extend(["tau_1"])
        self.labels.extend(["eta"])
        self.labels.extend(["t_0"])
        self.labels.extend(['phi0_%d'%i for i in range(self.nfiles)])
        self.labels.extend(['psi0_%d'%i for i in range(self.nfiles)])      
        if self.have_EFAC:
            self.labels.extend(["EFAC_%d"%i for i in range(self.nEFAC)])
   
    def get_labels(self):
        return self.labels

    def Prior(self, cube):

        pcube = np.zeros(cube.shape)
        #print (cube.shape)
        ipar = 0

        # Zeta
        pcube[ipar] = GaussianPrior(158.9,.8) (cube[ipar]); ipar += 1 # Zeta
        pcube[ipar] = UniformPrior(1, 90) (cube[ipar]); ipar += 1 # Theta
        pcube[ipar] = UniformPrior(0, 180) (cube[ipar]); ipar += 1 # Chi
        pcube[ipar] = UniformPrior(0, 360) (cube[ipar]); ipar += 1 # Phi
        pcube[ipar] = LogUniformPrior(1e-9, 1e-6) (cube[ipar]); ipar += 1 # ellipticity
        pcube[ipar] = LogUniformPrior(1e-8, 1e-5) (cube[ipar]); ipar += 1 # ellipticity
        pcube[ipar] = LogUniformPrior(.1, 10) (cube[ipar]); ipar += 1 # Frictional time scale (s)
        pcube[ipar] = LogUniformPrior(3, 100) (cube[ipar]); ipar += 1 # tau1
        pcube[ipar] = LogUniformPrior(1e-6, 0.1) (cube[ipar]); ipar += 1 # eta
        pcube[ipar] = GaussianPrior(58445,3) (cube[ipar]); ipar += 1 # T0
        
        for ii in range(self.nfiles):
            pcube[ipar+ii] = GaussianPrior((self.pPh[1][ii]+self.pPh[0][ii])/2., 5) (cube[ipar+ii]);
        ipar += self.nfiles
        for ii in range(self.nfiles):
            pcube[ipar+ii] = GaussianPrior((self.pPs[1][ii]+self.pPs[0][ii])/2., 5) (cube[ipar+ii])
        ipar += self.nfiles
        
        # EFAC
        if self.have_EFAC:
            #pcube[ipar:ipar+self.nEFAC] = cube[ipar:ipar+self.nEFAC]*1.3+1
            for ii in range(self.nEFAC):
                pcube[ipar+ii] =  LogUniformPrior(0.2, 5) (cube[ipar+ii])
        return pcube
        
    def get_data(self, filename,sig=5):
        #print(filename)
        ar = psrchive.Archive_load(filename)
        ar.tscrunch()
        ar.fscrunch()
        ar.convert_state('Stokes')
        ar.remove_baseline()
        rcvr = ar.get_receiver_name()
        self.rcvrs.append(rcvr)

        # Convert to infinite frequency
        try:
                F = psrchive.FaradayRotation()
                F.set_rotation_measure(ar.get_rotation_measure())
                F.execute(ar)
        except:
                print("Could not defaraday to infinite frequency. This option is only possible with a custom/recent version of psrchive")
                pass
        
        self.nbin = np.append(self.nbin, ar.get_nbin())

        data = ar.get_data()
        x = np.arange(0, ar.get_nbin()) / ar.get_nbin()*2*np.pi
        I = data[:,0,:,:][0][0]
        Q = data[:,1,:,:][0][0]
        U = data[:,2,:,:][0][0]
        V = data[:,3,:,:][0][0]
        L = np.sqrt(Q*Q+U*U)
        PA = 0.5*np.arctan2(U,Q)

        integ = ar.get_first_Integration()
        # Get baseline RMS (1) for total intensity (0)
        nI = np.sqrt((integ.baseline_stats()[1][0]))
        nQ = np.sqrt((integ.baseline_stats()[1][1]))
        nU = np.sqrt((integ.baseline_stats()[1][2]))
        
        xm = np.ma.masked_where(L<sig*nI,x)
        Qm = np.ma.masked_where(L<sig*nI,Q)
        Um = np.ma.masked_where(L<sig*nI,U)
        
        self.nI = np.append(self.nI, nI)
        self.nQ = np.append(self.nQ, nQ)
        self.nU = np.append(self.nU, nU)
        self.xm.append(xm)
        self.Qm.append(Qm)
        self.Um.append(Um)

        self.MJDs = np.append(self.MJDs, float(ar.get_first_Integration().get_epoch().strtempo()))
        
        del ar

    def exc_phs(self, exc):
        # Check if we have the right number of inputs vs number of files
        if len(exc) < self.nfiles:
            raise ValueError("Number of input in config file (%d) does not match the number of profiles (%d)"%(len(exc), self.nfiles))
            
        # For each entry in config file for phase range exclusion
        for iprof,key in enumerate(exc.keys()):            
            xval = np.array(exc[key].rstrip().split(';'))            
            val = xval.astype(float)
            # Mask data by range and compress later
            pairs = zip(val[::2], val[1::2])
            for p in pairs:
                #print(p)
                self.xm[iprof][int(p[0]*self.nbin[iprof]):int(p[1]*self.nbin[iprof])] = np.ma.masked
                self.Qm[iprof][int(p[0]*self.nbin[iprof]):int(p[1]*self.nbin[iprof])] = np.ma.masked
                self.Um[iprof][int(p[0]*self.nbin[iprof]):int(p[1]*self.nbin[iprof])] = np.ma.masked
                
            if iprof+1== self.nfiles:
                break
             
    def LogLikelihood(self, cube):
        #print(self.MJDs)
        L =  get_L(cube, self.nQ, self.nU, self.xm, self.Qm, self.Um, self.MJDs, self.MJD_idx, self.have_EFAC, self.nEFAC, self.rcvr_lut)
        return L


             

# Input filenames
filenames = sys.argv[1:]
cfgfilename = "config.ini"
sig = 3 # Threshold for L (in sigma)
have_EFAC = True
nlive = 1000 # Power of 2s for GPU

#frac_remain = 0.1
cfg = configparser.ConfigParser(allow_no_value=True)
cfg.read(cfgfilename)

model = Precessnest(filenames, sig=sig, have_EFAC=have_EFAC, config=cfg)
paramnames = model.get_labels()
ndims = len(paramnames)
nDerived = 0
#nsteps = 2*len(paramnames)
nr = 5

# RUN THE ANALYSIS
settings = PolyChordSettings(ndims, nDerived)
settings.file_root = 'friction_elldampT2_smallPrior2b'
settings.nlive = 10*ndims
settings.cluster_posteriors = False
settings.do_clustering = False
settings.write_dead = False
settings.write_resume = False
settings.read_resume = False
settings.num_repeats = ndims * nr
settings.synchronous = False

if rank==0:
    print("Friction + ellipticity damping analysis using CPUs fp64")
    print("Ndim = %d\n"%ndims)
    print("nEFAC = %d\n"%model.get_nEFAC())
    print("Nrepeats = %d\n"%settings.num_repeats)
    #print("Frac remain = %f\n"%frac_remain)
    print("Nlive = %d\n"%nlive)
    print("Using PolyChord\n")
#"""
output = pypolychord.run_polychord(model.LogLikelihood, ndims, nDerived, settings, model.Prior, dumper)

if rank==0:
    par = [('%s'%i, r'\%s'%i) for i in paramnames]
    par += [('beta_%d*'%i, r'\beta_%d'%i) for i in range(nDerived)]
    output.make_paramnames_files(par)

"""
cube = np.ones(ndims)
#print (model.Prior(cube))
cube[0] = 159
cube[1] = 38
cube[2] = 163
cube[3] = 185
cube[4] = 6e-7
cube[5] = 3
cube[6] = 68
cube[7] = 0.015
cube[8] = 58445
#print (model.LogLikelihood(cube))

for i in range(1):
    #t = time.process_time()
    L, betas = model.LogLikelihood(cube)
    print(betas)
    #print("elapsed : ", time.process_time() - t)

mjds = model.MJDs

#for m,b in zip(mjds, betas):
#    print (m,b)
"""
