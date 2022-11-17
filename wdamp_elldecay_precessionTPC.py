#!/usr/bin/env python
from __future__ import print_function, division
import sys
import psrchive
import numpy as np
import configparser
import numba
from NumbaQuadpack import quadpack_sig, dqags
from numba import njit
import numba as nb
from numba.typed import List
from rvm import rvm
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, LogUniformPrior, GaussianPrior

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0
    pass

# Period of the magnetar in s
Period = 5.54

def dumper(live, dead, logweights, logZ, logZerr):
    print("LogZ: ", logZ)

@nb.cfunc(quadpack_sig)
def func(t, pars_):
    data = nb.carray(pars_, (7,))
    eps0=data[0]
    eps1=data[1]
    tau1=data[2]
    omega0=data[3] # Tweak to check for effect of changing pulse period # - 2*np.pi*86400*(3.215e-13 * (t-data[6]))
    theta0=data[4]
    tau=data[5]
    t0=data[6]
    return (eps0+eps1*np.e**(-(t-t0)/tau1)) * omega0*np.cos(theta0*np.e**(-(t-t0 )/tau))

funcptr = func.address

@njit(fastmath=False)
def get_L(cube, nQ, nU, xm, Qm, Um, mjds, have_EFAC, nEFAC, rcvr_lut):
    ipar = 0
    nfiles = len(xm)
    zeta = np.deg2rad(cube[ipar]); ipar += 1
    theta0 = np.deg2rad(cube[ipar]); ipar += 1  # the initial wobble angle
    chi = np.deg2rad(cube[ipar]); ipar += 1 # the angle between the magnetic dipole and symmetric axis
    Phi0 = np.deg2rad(cube[ipar]); ipar += 1 # initial phase of the precession)
    #Phi0 =  np.arctan2(cube[ipar+1],cube[ipar]); ipar += 2 # the initial phase of the precession
    epsilon0 = cube[ipar]; ipar += 1 # the constant ellipticity
    epsilon1 = cube[ipar]; ipar += 1 # inital damped ellipticity
    tau = cube[ipar]; ipar += 1 # the damping timescale of the wobble angle
    tau1 = cube[ipar]; ipar += 1 # the damping timescale of the ellipticity
    t0 = cube[ipar]; ipar += 1 # the beginning time of the precession
    phi0s = np.deg2rad(cube[ipar:ipar+nfiles])
    psi0s = np.deg2rad(cube[ipar+nfiles:ipar+2*nfiles])

    # Avoid start of precession after we started observing it
    if t0>58460:
        return -2e10, 0
    
    omega0=2*np.pi*86400./Period
    
    chi2 = 0   
    logdet = 0

    betas = np.zeros(nfiles, dtype=np.float64)
    pars = np.zeros(7, dtype=np.float64)
    pars[0] = epsilon0
    pars[1] = epsilon1
    pars[2] = tau1
    pars[3] = omega0
    pars[4] = theta0
    pars[5] = tau
    pars[6] = t0

    for ii in range(nfiles):
        # the exponential damping form of wobble angle
        theta=theta0* np.e**(-(mjds[ii]-t0)/tau)
        # the exponential damping form ellipticity
        #epsilon=epsilon0 + epsilon1 * np.e**(-(mjds[ii]-t0)/tau1)
        # the precessing angular frequency
        #omegap=-epsilon * omega0 * np.cos(theta)

        # Integrate the precession
        omegap, abserr, success = dqags(funcptr, t0, mjds[ii], data=pars)
        # the precessing phase
        psi=np.pi/2-omegap-Phi0
        # the evolution of magnetic inclination angle
        alpha=np.arccos(np.sin(theta)*np.sin(psi)*np.sin(chi)+np.cos(theta)*np.cos(chi))
        # the evolution of impact parameter beta
        beta = zeta - alpha
        
        betas[ii:ii+1] = beta
        phi0 = phi0s[ii]
        psi0 = psi0s[ii]
        
        if have_EFAC:
            EFAC = cube[ipar+2*nfiles+rcvr_lut[ii]]
        else:
            EFAC = 1.
            
        nQ2 = nQ[ii]*nQ[ii] * EFAC*EFAC
        nU2 = nU[ii]*nU[ii] * EFAC*EFAC
        
        # Compute the modelled PA            
        alpha = zeta-beta
        sin_al = np.sin(alpha)
        xp = xm[ii]-phi0
        
        argx = np.cos(alpha)*np.sin(zeta) - sin_al*np.cos(zeta)*np.cos(xp)
        argy =  sin_al * np.sin(xp)
        
        PA2 = 2*(-np.arctan(argy/argx) + psi0)
        cos2PA = np.cos(PA2)
        sin2PA = np.sin(PA2)
        
        L = (Qm[ii] * cos2PA/nQ2 + Um[ii] * sin2PA/nU2) / (cos2PA*cos2PA/nQ2 + sin2PA*sin2PA/nU2) * np.exp(1j*PA2)
        
        chi2 += np.sum((Qm[ii]-np.real(L))**2 / nQ2 + (Um[ii]-np.imag(L))**2 / nU2)
        logdet += len(Qm[ii]) * (np.log(nQ2) + np.log(nU2))
    return -0.5 * chi2 -0.5*logdet, 0

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

        self.npts = 0
        for ii in range(self.nfiles):
            self.xm[ii] = self.xm[ii].compressed()
            self.Qm[ii] = self.Qm[ii].compressed()
            self.Um[ii] = self.Um[ii].compressed()
            self.npts += len(self.xm[ii])
            
            pfo = open("Profile_%d-PA.log"%ii, 'w')
            for x,PA,PAe in zip(np.rad2deg(self.xm[ii]), np.rad2deg(0.5*np.arctan2(self.Um[ii],self.Qm[ii])), 28.65*self.nI[ii]/(self.Um[ii]**2 + self.Qm[ii]**2)**.5 ):
                pfo.write("%f %f %f\n"%(x, PA, PAe))
            pfo.close()
        
        if rank==0:
            print("Number of datapoints: ", self.npts)
        self.set_labels()

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
        self.labels.extend(["t_0"]) # GD
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
        #pcube[ipar] = cube[ipar]*(self.pZe[1]-self.pZe[0])+self.pZe[0]; ipar += 1
        pcube[ipar] = GaussianPrior(158.9,.8) (cube[ipar]); ipar += 1 # GD
        pcube[ipar] = UniformPrior(1, 90) (cube[ipar]); ipar += 1
        pcube[ipar] = UniformPrior(0, 180) (cube[ipar]); ipar += 1
        pcube[ipar] = UniformPrior(0, 360) (cube[ipar]); ipar += 1
        pcube[ipar] = LogUniformPrior(1e-10, 1e-6) (cube[ipar]); ipar += 1
        pcube[ipar] = LogUniformPrior(1e-9, 1e-5) (cube[ipar]); ipar += 1
        pcube[ipar] = LogUniformPrior(1, 200) (cube[ipar]); ipar += 1
        pcube[ipar] = LogUniformPrior(1, 200) (cube[ipar]); ipar += 1
        pcube[ipar] = GaussianPrior(58445,1) (cube[ipar]); ipar += 1
        
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
        L =  get_L(cube, self.nQ, self.nU, self.xm, self.Qm, self.Um, self.MJDs, self.have_EFAC, self.nEFAC, self.rcvr_lut)
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
# GD
nr = 8 

# RUN THE ANALYSIS
settings = PolyChordSettings(ndims, nDerived)
settings.file_root = 'wdamp_elldampT2_smallPrior2b'
settings.nlive = 10*ndims
settings.nlive = 15*ndims
settings.cluster_posteriors = False
settings.do_clustering = False
settings.write_dead = False
settings.write_resume = False
settings.read_resume = False
settings.num_repeats = ndims * nr
settings.synchronous = False

if rank==0:
    print("Wobble + ellipticity damping analysis using CPUs fp64")
    print("Ndim = %d\n"%ndims)
    print("nEFAC = %d\n"%model.get_nEFAC())
    print("Nrepeats = %d\n"%settings.num_repeats)
    #print("Frac remain = %f\n"%frac_remain)
    print("Nlive = %d\n"%nlive)
    print("Using PolyChord\n")

output = pypolychord.run_polychord(model.LogLikelihood, ndims, nDerived, settings, model.Prior, dumper)

if rank==0:
    par = [('%s'%i, r'\%s'%i) for i in paramnames]
    par += [('beta_%d*'%i, r'\beta_%d'%i) for i in range(nDerived)]
    output.make_paramnames_files(par)

