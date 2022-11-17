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
from pypolychord.priors import UniformPrior, GaussianPrior, LogUniformPrior

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0
    pass


def dumper(live, dead, logweights, logZ, logZerr):
    print("LogZ: ", logZ)

@njit(fastmath=True)
def get_L(cube, nQ, nU, xm, Qm, Um, have_EFAC, nEFAC, rcvr_lut):
    
        nfiles = len(xm)
        zeta = np.deg2rad(cube[0])
        betas = np.deg2rad(cube[1:1+nfiles])
        phi0s = np.deg2rad(cube[1+nfiles:1+2*nfiles])
        psi0s = np.deg2rad(cube[1+2*nfiles:1+3*nfiles])
        
        chi = 0   
        logdet = 0
        for ii in range(nfiles):
            
            beta = betas[ii]
            phi0 = phi0s[ii]
            psi0 = psi0s[ii]

            if have_EFAC:
                EFAC = cube[1+3*nfiles+rcvr_lut[ii]]
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
            
            chi += np.sum((Qm[ii]-np.real(L))**2 / nQ2 + (Um[ii]-np.imag(L))**2 / nU2)
            logdet += len(Qm[ii]) * (np.log(nQ2) + np.log(nU2))
        return -0.5 * chi -0.5*logdet, 0

class Precessnest():
    def __init__(self, filenames, sig=5, have_EFAC=False, config = None):
    
        self.nI = np.array([])
        self.nQ = np.array([])
        self.nU = np.array([])
        self.nbin = np.array([])
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

        npts = 0
        for ii in range(self.nfiles):
            if rank==0:
                pfo = open("Profile_%d-PA.log"%ii, 'w')
                xu = np.ma.getdata(self.xm[ii])
                Uu = np.ma.getdata(self.Um[ii])
                Qu = np.ma.getdata(self.Qm[ii])
                PAs = np.rad2deg(0.5*np.arctan2(Uu, Qu))
                PAes = 28.65 * self.nI[ii] / (Uu**2 + Qu**2)**.5
                for m, x, PA, PAe in zip(self.xm[ii], np.rad2deg(xu), PAs, PAes):
                    if m is np.ma.masked:
                        pfo.write("%f %f %f 0\n"%(x, PA, PAe))
                    else:
                        pfo.write("%f %f %f 1\n"%(x, PA, PAe))
                pfo.close()

            self.xm[ii] = self.xm[ii].compressed()
            self.Qm[ii] = self.Qm[ii].compressed()
            self.Um[ii] = self.Um[ii].compressed()
            npts += len(self.xm[ii])
       
        self.set_labels()
        if rank==0:
            print("Total Npts = %d"%npts)

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
        self.labels.extend(['beta_%d'%i for i in range(self.nfiles)])
        self.labels.extend(['phi0_%d'%i for i in range(self.nfiles)])
        self.labels.extend(['psi0_%d'%i for i in range(self.nfiles)])      
        if self.have_EFAC:
            self.labels.extend(["EFAC_%d"%i for i in range(self.nEFAC)])
   
    def get_labels(self):
        return self.labels

    def Prior(self, cube):

        pcube = np.zeros(cube.shape)
        ipar = 0

        # Zeta
        pcube[ipar] = GaussianPrior(160,5) (cube[ipar]); ipar += 1
        for ii in range(self.nfiles):
            pcube[ipar+ii] = GaussianPrior((self.pBe[1][ii]+self.pBe[0][ii])/2., 5) (cube[ipar+ii]);
        ipar += self.nfiles
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

        # Write Stokes Profile in ascii format
        if rank==0:
            pfo = open("Profile_%d-prof.log"%(len(self.rcvrs)-1), 'w')
            for xi, Ii, Qi, Ui, Vi in zip(x,I,Q,U,V):
                pfo.write("%f %f %f %f %f\n"%(xi, Ii, Qi, Ui, Vi))
            pfo.close()
            
        integ = ar.get_first_Integration()
        # Get baseline RMS (1) for total intensity (0)
        nI = np.sqrt((integ.baseline_stats()[1][0]))
        nQ = np.sqrt((integ.baseline_stats()[1][1]))
        nU = np.sqrt((integ.baseline_stats()[1][2]))
        
        #xm = np.ma.masked_where(L<sig*nI,x)
        #Qm = np.ma.masked_where(L<sig*nI,Q)
        #Um = np.ma.masked_where(L<sig*nI,U)
        xm = np.ma.masked_where(L<sig*nI,x).compressed()
        Qm = np.ma.masked_where(L<sig*nI,Q).compressed()
        Um = np.ma.masked_where(L<sig*nI,U).compressed()
        
        self.nI = np.append(self.nI, nI)
        self.nQ = np.append(self.nQ, nQ)
        self.nU = np.append(self.nU, nU)
        self.xm.append(np.ma.array(xm))
        self.Qm.append(np.ma.array(Qm))
        self.Um.append(np.ma.array(Um))

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

                for ip,phas in enumerate(self.xm[iprof]):
                    if p[0]*2*np.pi <= phas and phas <= p[1]*2*np.pi:
                
                        self.xm[iprof][ip] = np.ma.masked
                        self.Qm[iprof][ip] = np.ma.masked
                        self.Um[iprof][ip] = np.ma.masked
                
            if iprof+1== self.nfiles:
                break
             
    def LogLikelihood(self, cube):
        return get_L(cube, self.nQ, self.nU, self.xm, self.Qm, self.Um, self.have_EFAC, self.nEFAC, self.rcvr_lut)


             

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

# RUN THE ANALYSIS
settings = PolyChordSettings(ndims, nDerived)
settings.file_root = 'freeLog'
settings.nlive = ndims * 10
if max(1000,settings.nlive)==1000:
    settings.nlive = 1000
settings.cluster_posteriors = False
settings.do_clustering = False
settings.write_dead = False
settings.write_resume = False
settings.read_resume = False
settings.num_repeats = ndims * 10
settings.synchronous = False
if rank==0:
    print("Free precession analysis using CPUs fp64")
    print("Ndim = %d\n"%ndims)
    print("nEFAC = %d\n"%model.get_nEFAC())
    print("Nrepeats = %d\n"%settings.num_repeats)
    #print("Frac remain = %f\n"%frac_remain)
    print("Nlive = %d\n"%nlive)
    print("Using PolyChord\n")

output = pypolychord.run_polychord(model.LogLikelihood, ndims, nDerived, settings, model.Prior, dumper)

if rank==0:
    par = [('%s'%i, r'\%s'%i) for i in paramnames]
    output.make_paramnames_files(par)
