#!/usr/bin/env python
from __future__ import print_function, division
import sys
import psrchive
import numpy as np
import configparser
from numba import njit
from numba.typed import List
from rvm import rvm
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, LogUniformPrior

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

@njit(fastmath=True)
def get_L(cube, nQ, nU, xm, Qm, Um, mjds, have_EFAC, nEFAC, rcvr_lut):
    
        nfiles = len(xm)
        zeta = np.deg2rad(cube[0])
        theta0 = np.deg2rad(cube[1])
        chi = np.deg2rad(cube[2])
        Phi0 =  np.deg2rad(cube[3])
        Prec = cube[4]
        tau = cube[5]
        t0 = cube[6]
        phi0s = np.deg2rad(cube[7:7+nfiles])
        psi0s = np.deg2rad(cube[7+nfiles:7+2*nfiles])

        omega0=2*np.pi/Period

        chi2 = 0   
        logdet = 0
        betas = np.zeros(nfiles, dtype=np.float64)
        for ii in range(nfiles):

            epsilon=np.abs(Period/(Prec)/np.cos(theta0))

            # give the initial value of angular frequency 
            a=omega0*np.sin(theta0)
            b=omega0*np.cos(theta0)
            
            # give the time evolution of theta
            L=(a**2 + (1+epsilon)**2 * b**2)**0.5
            omega3=b * np.e**((mjds[ii]-t0)/tau)*(1+epsilon)/(np.e**((mjds[ii]-t0)/tau)+epsilon)
            I3=(1+epsilon*np.e**(-(mjds[ii]-t0)/tau))
            theta=np.arccos(I3*omega3/L)
            
            # give the time evolution of psi, alpha, and beta
            psi=np.pi/2-(b*(1 + epsilon)*((mjds[ii]-t0)  - tau*np.log(np.e**((mjds[ii]-t0)/tau) + epsilon) + tau * np.log(1+epsilon))) - Phi0
            alpha=np.arccos(np.sin(theta)*np.sin(psi)*np.sin(chi)+np.cos(theta)*np.cos(chi))
            beta=zeta-alpha
            #print(mjds[ii], np.degrees(beta))
        
            betas[ii:ii+1] = beta
            phi0 = phi0s[ii]
            psi0 = psi0s[ii]

            if have_EFAC:
                EFAC = cube[7+2*nfiles+rcvr_lut[ii]]
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
        return -0.5 * chi2 -0.5*logdet, np.degrees(betas)

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
        
        for ii in range(self.nfiles):
            self.xm[ii] = self.xm[ii].compressed()
            self.Qm[ii] = self.Qm[ii].compressed()
            self.Um[ii] = self.Um[ii].compressed()

            pfo = open("Profile_%d-PA.log"%ii, 'w')
            for x,PA,PAe in zip(np.rad2deg(self.xm[ii]), np.rad2deg(0.5*np.arctan2(self.Um[ii],self.Qm[ii])), 28.65*self.nI[ii]/(self.Um[ii]**2 + self.Qm[ii]**2)**.5 ):
                pfo.write("%f %f %f\n"%(x, PA, PAe))
            pfo.close()
       
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
        self.labels.extend(["Pp"])
        self.labels.extend(["tau"])
        self.labels.extend(["t0"])
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
        pcube[ipar] = cube[ipar]*(self.pZe[1]-self.pZe[0])+self.pZe[0]; ipar += 1
        pcube[ipar] = UniformPrior(2,42) (cube[ipar]); ipar += 1
        pcube[ipar] = UniformPrior(160,180) (cube[ipar]); ipar += 1
        pcube[ipar] = UniformPrior(30,310) (cube[ipar]); ipar += 1

        pcube[ipar] = LogUniformPrior(5, 180) (cube[ipar]); ipar += 1
        pcube[ipar] = LogUniformPrior(5, 180) (cube[ipar]); ipar += 1
        pcube[ipar] = cube[ipar]*(60) + 58420; ipar += 1

        pcube[ipar:ipar+self.nfiles] = cube[ipar:ipar+self.nfiles]*(self.pPh[1]-self.pPh[0])+self.pPh[0]; ipar += self.nfiles
        pcube[ipar:ipar+self.nfiles] = cube[ipar:ipar+self.nfiles]*(self.pPs[1]-self.pPs[0])+self.pPs[0]; ipar += self.nfiles
        
        # EFAC
        if self.have_EFAC:
            pcube[ipar:ipar+self.nEFAC] = UniformPrior(1, 6)  (cube[ipar:ipar+self.nEFAC])
        return pcube
        
    def get_data(self, filename,sig=5):
        print(filename)
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
sig = 4 # Threshold for L (in sigma)
have_EFAC = True
nlive = 1500 # Power of 2s for GPU
#frac_remain = 0.1
cfg = configparser.ConfigParser(allow_no_value=True)
cfg.read(cfgfilename)

model = Precessnest(filenames, sig=sig, have_EFAC=have_EFAC, config=cfg)
paramnames = model.get_labels()
ndims = len(paramnames)
nDerived = len(filenames)
#nsteps = 2*len(paramnames)

# RUN THE ANALYSIS
settings = PolyChordSettings(ndims, nDerived)
settings.file_root = 'elldamp4'
settings.nlive = nlive
settings.cluster_posteriors = False
settings.do_clustering = False
settings.write_resume = False
settings.read_resume = False
settings.num_repeats = ndims * 10
settings.synchronous = False

if rank==0:
    print("Decreasing ellipticity precession analysis using CPUs fp64")
    print("Ndim = %d\n"%ndims)
    print("nEFAC = %d\n"%model.get_nEFAC())
    print("Nrepeats = %d\n"%settings.num_repeats)
    print("Nlive = %d\n"%nlive)
    print("Using PolyChord\n")

output = pypolychord.run_polychord(model.LogLikelihood, ndims, nDerived, settings, model.Prior, dumper)

if rank==0:
    par = [('%s'%i, r'\%s'%i) for i in paramnames]
    par += [('beta_%d*'%i, r'\beta_%d'%i) for i in range(nDerived)]
    output.make_paramnames_files(par)

