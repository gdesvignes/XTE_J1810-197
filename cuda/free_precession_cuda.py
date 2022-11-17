#!/usr/bin/env python
from __future__ import print_function, division
import sys
import psrchive
import numpy as np
from ultranest import ReactiveNestedSampler, stepsampler
from ultranest.mlfriends import RobustEllipsoidRegion
import configparser
from rvm import rvm
from numpy.random import default_rng

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""

__device__ void warpReduce(volatile float *sdata, unsigned int tid, unsigned int blockSize) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void get_L_gpu(float *cube, float *nQ, float *nU,float *xm, float *Qm, float *Um, int *lut, float *odata, int nparam)
{
  const int ifile = blockIdx.y;
  const int ipt = threadIdx.x + blockDim.x*ifile;
  __shared__ float sdata[256];
float2 L;
float Qi = Qm[ipt];
float Ui = Um[ipt];
float nQ2, nU2, PA2;

if (Qi ==0 && Ui==0) sdata[threadIdx.x] = 0.0;
else {
  float zeta = cube[blockIdx.x*nparam] * 0.017453292519943295;
  float beta = cube[blockIdx.x*nparam + ifile + 1]* 0.017453292519943295;
  float phi0 = cube[blockIdx.x*nparam + ifile + gridDim.y + 1]* 0.017453292519943295;
  float psi0 = cube[blockIdx.x*nparam + ifile + gridDim.y*2 + 1]* 0.017453292519943295;
  float EFAC = cube[blockIdx.x*nparam + gridDim.y*3 + 1 + lut[ifile]];

  nQ2 = nQ[ifile] * EFAC*EFAC;
  nU2 = nU[ifile] * EFAC*EFAC;

  // Compute the modelled PA
  float zb = zeta-beta;
  float xp = xm[ipt]-phi0;
  float argx = cosf(zb)*sinf(zeta) - sinf(zb)*cosf(zeta)*cosf(xp);
  float argy =  sinf(zb) * sinf(xp);
  float cos2PA, sin2PA;
  PA2 = -2*atanf(argy/argx) + 2*psi0;
  sincosf(PA2, &sin2PA, &cos2PA);

  argx = (Qi * cos2PA/nQ2 + Ui * sin2PA/nU2) / (cos2PA*cos2PA/nQ2 + sin2PA*sin2PA/nU2);
  L.x = argx * cos2PA;
  L.y = argx * sin2PA;
  sdata[threadIdx.x] = (Qi-L.x)*(Qi-L.x) / nQ2 + (Ui-L.y)*(Ui-L.y) / nU2 + logf(nQ2) + logf(nU2);
}

__syncthreads();

if (blockDim.x >= 256) { if (threadIdx.x < 128) { sdata[threadIdx.x] += sdata[threadIdx.x + 128]; } __syncthreads(); }
if (blockDim.x >= 128) { if (threadIdx.x < 64) { sdata[threadIdx.x] += sdata[threadIdx.x + 64]; } __syncthreads(); }
if (threadIdx.x < 32)  warpReduce(sdata, threadIdx.x, blockDim.x);
if (threadIdx.x == 0) odata[blockIdx.y * gridDim.x + blockIdx.x] = -0.5 * sdata[0];
}


__global__ void get_L_gpu2(float *idata, float *odata, int nfiles, int nlive) 
{
__shared__ float sdata[128];

unsigned int tid = threadIdx.x;
unsigned int bs = blockDim.x; 
sdata[tid] = 0.0;

if (blockIdx.x*bs + tid < nlive) {
  for(int i=0; i<nfiles; i++) {
    sdata[tid] += idata[tid + bs*blockIdx.x + i*nlive];
  }
  odata[tid + bs*blockIdx.x] = sdata[tid];
}
}

""")


get_L_gpu = mod.get_function("get_L_gpu")
get_L_gpu2 = mod.get_function("get_L_gpu2")

class Precessnest():
    def __init__(self, filenames, sig=5, have_EFAC=False, config = None, nlive=512):
    
        self.nI = np.array([])
        self.nQ = np.array([])
        self.nU = np.array([])
        self.xm = []
        self.Qm = []
        self.Um = []
        self.rcvrs = list()

        self.filenames = filenames
        self.nfiles = len(self.filenames)
        self.labels = []
        self.have_EFAC = have_EFAC
        self.nlive = nlive
        self.nparams = None
        
        for filename in self.filenames:
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

        print(self.rcvr_lut)
        
        # Check if we have to exclude phase range from the data
        if config.has_section('exc_phases'):
            self.exc_phs(config['exc_phases'])
            
        self.set_pZeta(config['zeta'])
        self.set_pBeta(config['beta'])
        self.set_pPhi0(config['phi'])
        self.set_pPsi0(config['psi'])         
        self.set_labels()
        self.nparams = len(self.get_labels())
        
        self.manage_arrays()
        self.write_data()
        
    def write_data(self):
        for ii in range(self.nfiles):
            pfo = open("Profile_%d-PA.log"%ii, 'w')
            for x,PA in zip(np.rad2deg(self.xm[ii]), np.rad2deg(0.5*np.arctan2(self.Um[ii],self.Qm[ii]))):
                pfo.write("%f %f\n"%(x, PA))
            pfo.close()
            
    def manage_arrays(self):
        array_len = np.array([], dtype=np.int8)    
        for ii in range(self.nfiles):
            self.xm[ii] = self.xm[ii].compressed()
            self.Qm[ii] = self.Qm[ii].compressed()
            self.Um[ii] = self.Um[ii].compressed()

            array_len = np.append(array_len, int(len(self.xm[ii])))
            print(self.filenames[ii], len(self.xm[ii]))
        max_array_len = np.max(array_len)
        #print(max_array_len)
        
        # Find next multiple of 32 for multiple of warp size
        self.max_array_len =  (max_array_len|255)+1
        #self.max_array_len = 384
        print("Using block size of ", self.max_array_len)
        x = np.zeros((self.nfiles, self.max_array_len))
        Q = np.zeros((self.nfiles, self.max_array_len))
        U = np.zeros((self.nfiles, self.max_array_len))

        for ii in range(self.nfiles):
            np.put(x[ii], np.indices(self.xm[ii].shape), self.xm[ii])
            np.put(Q[ii], np.indices(self.Qm[ii].shape), self.Qm[ii])
            np.put(U[ii], np.indices(self.Um[ii].shape), self.Um[ii])
        self.xm = x.astype(np.float32)
        self.Qm = Q.astype(np.float32)
        self.Um = U.astype(np.float32)

        # Create GPU memory for the sampling cube
        self.cube = np.zeros((self.nlive, self.nparams), dtype=np.float32)
        self.cube_gpu = cuda.mem_alloc(self.cube.nbytes)

        # Copy input Stokes (Q,U) and phasedata to GPU
        self.xm_gpu = cuda.mem_alloc(self.xm.nbytes)
        cuda.memcpy_htod(self.xm_gpu, self.xm)

        self.Qm_gpu = cuda.mem_alloc(self.Qm.nbytes)
        cuda.memcpy_htod(self.Qm_gpu, self.Qm)

        self.Um_gpu = cuda.mem_alloc(self.Um.nbytes)
        cuda.memcpy_htod(self.Um_gpu, self.Um)

        # Recast noise of Q and U as 32 bit float and send to GPU
        self.nQ = self.nQ.astype(np.float32)
        self.nU	= self.nU.astype(np.float32)
        self.nQ_gpu = cuda.mem_alloc(self.nQ.nbytes)
        self.nU_gpu = cuda.mem_alloc(self.nU.nbytes)
        cuda.memcpy_htod(self.nQ_gpu, self.nQ)
        cuda.memcpy_htod(self.nU_gpu, self.nU)

        # Array to store the likelihood values
        self.LogLike_tmp = np.zeros((self.nlive, self.nfiles), dtype=np.float32)
        self.LogLike_tmp_gpu = cuda.mem_alloc(self.LogLike_tmp.nbytes)
        #print("%d bytes"% self.LogLike_tmp.nbytes)
        
        self.LogLike = np.zeros(self.nlive, dtype=np.float32)
        self.LogLike_gpu = cuda.mem_alloc(self.LogLike.nbytes)

        # Copy LUT to GPU
        self.rcvr_lut = self.rcvr_lut.astype(np.int32)
        self.rcvr_lut_gpu = cuda.mem_alloc(self.rcvr_lut.nbytes)
        cuda.memcpy_htod(self.rcvr_lut_gpu, self.rcvr_lut)
        
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

        pcube[:,ipar] = cube[:,ipar]*(self.pZe[1]-self.pZe[0])+self.pZe[0]; ipar +=1
        pcube[:,ipar:ipar+self.nfiles] = cube[:,ipar:ipar+self.nfiles]*(self.pBe[1]-self.pBe[0])+self.pBe[0]; ipar += self.nfiles
        pcube[:,ipar:ipar+self.nfiles] = cube[:,ipar:ipar+self.nfiles]*(self.pPh[1]-self.pPh[0])+self.pPh[0]; ipar += self.nfiles
        pcube[:,ipar:ipar+self.nfiles] = cube[:,ipar:ipar+self.nfiles]*(self.pPs[1]-self.pPs[0])+self.pPs[0]; ipar += self.nfiles
        
        # EFAC
        if self.have_EFAC:
            pcube[:,ipar:ipar+self.nEFAC] = cube[:,ipar:ipar+self.nEFAC]*2+.3
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
        
        self.nbin = ar.get_nbin()
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
        
        self.nI = np.append(self.nI, nI*nI)
        self.nQ = np.append(self.nQ, nQ*nQ)
        self.nU = np.append(self.nU, nU*nU)

        self.xm.append(xm)
        self.Qm.append(Qm)
        self.Um.append(Um)

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
            print ("\nFile %s Exc phases:"%self.filenames[iprof])
            for p in pairs:
                print(p, )
                self.xm[iprof][int(p[0]*self.nbin):int(p[1]*self.nbin)] = np.ma.masked
                self.Qm[iprof][int(p[0]*self.nbin):int(p[1]*self.nbin)] = np.ma.masked
                self.Um[iprof][int(p[0]*self.nbin):int(p[1]*self.nbin)] = np.ma.masked
                
            if iprof+1== self.nfiles:
                break
             
    def LogLikelihood(self, cube):
        return get_L(cube, self.nQ, self.nU, self.xm, self.Qm, self.Um, self.have_EFAC, self.nEFAC, self.rcvr_lut)

    def LogLikelihood_gpu(self, cube):
        nlive = cube.shape[0]
        #print("\n",nlive,"\n")
        cuda.memcpy_htod(self.cube_gpu, cube.astype(np.float32))
        get_L_gpu(self.cube_gpu, self.nQ_gpu, self.nU_gpu, self.xm_gpu, self.Qm_gpu, self.Um_gpu, self.rcvr_lut_gpu, self.LogLike_tmp_gpu, np.int32(self.nparams), block=(int(self.max_array_len), 1,1), grid=(nlive, self.nfiles))

        get_L_gpu2(self.LogLike_tmp_gpu, self.LogLike_gpu, np.int32(self.nfiles), np.int32(nlive), block=(128, 1,1), grid=(int(np.ceil(nlive/128)), 1))

        cuda.memcpy_dtoh(self.LogLike, self.LogLike_gpu)
        return self.LogLike[0:nlive]

        
# Input filenames
filenames = sys.argv[1:]
cfgfilename = "config.ini"
sig = 4 # Threshold for L (in sigma)
have_EFAC = True
nlive = 512 # Power of 2s for GPU
frac_remain = 0.4
cfg = configparser.ConfigParser(allow_no_value=True)
cfg.read(cfgfilename)

model = Precessnest(filenames, sig=sig, have_EFAC=have_EFAC, config=cfg, nlive=nlive)
paramnames = model.get_labels()
nsteps = 2*len(paramnames)

# RUN THE ANALYSIS
print("Free precession analysis using CUDA fp32")
print("Ndim = %d\n"%len(paramnames))
print("nEFAC = %d\n"%model.get_nEFAC())
print("Nsteps = %d\n"%nsteps)
print("Nsteps = %f\n"%frac_remain)
print("Nlive = %d\n"%nlive)
print("Using SpeedVariableRegionSliceSampler\n")
sampler = ReactiveNestedSampler(paramnames, model.LogLikelihood_gpu, transform=model.Prior, vectorized=True, log_dir=".", num_test_samples=0, ndraw_min=2048)
# create step sampler:
matrix = np.ones((nsteps, len(paramnames)), dtype=bool)
matrix[int(nsteps/3):-1,-1] = False
matrix[int(nsteps/3):-1,-2] = False
matrix[int(nsteps/3):-1,-3] = False
#matrix[2,-1] = False
#sampler.stepsampler = stepsampler.SpeedVariableRegionSliceSampler(matrix, adaptive_nsteps='move-distance')
#sampler.stepsampler = stepsampler.RegionSliceSampler(nsteps=200, adaptive_nsteps='move-distance')
sampler.stepsampler = stepsampler.RegionBallSliceSampler(nsteps=len(paramnames),  adaptive_nsteps='move-distance')
sampler.run(min_num_live_points=nlive, viz_callback=False, frac_remain=frac_remain)
sampler.print_results()

"""

cube = default_rng(42).random((nlive,len(paramnames)))
import time
for i in range(6):
        t = time.process_time()
#        c = model.Prior(np.ones((nlive,len(paramnames)))); print("elapsed : ", time.process_time() - t)
        l = model.LogLikelihood_gpu(cube)
        #l = model.LogLikelihood_gpu(np.ones((nlive,len(paramnames))))
        print("elapsed : ", time.process_time() - t, "LogL = ", l[0], l[1], l[2], l[nlive-1])
"""
