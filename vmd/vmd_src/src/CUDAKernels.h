/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDAKernels.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.56 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Wrapper for CUDA kernels and utility functions
 *   used by the CUDAAccel C++ class 
 ***************************************************************************/
#ifndef CUDAKERNELS_H
#define CUDAKERNELS_H

#include "WKFThreads.h"

/* avoid parameter name collisions with AIX5 "hz" macro */
#undef hz

#if defined(__cplusplus)
extern "C" {
#endif

/* 
 * number of CUDA devices available 
 */
#define VMDCUDA_ERR_NONE          0
#define VMDCUDA_ERR_GENERAL      -1
#define VMDCUDA_ERR_NODEVICES    -2
#define VMDCUDA_ERR_SOMEDEVICES  -3
#define VMDCUDA_ERR_DRVMISMATCH  -4
#define VMDCUDA_ERR_EMUDEVICE    -5
int vmd_cuda_num_devices(int *numdev);

/* replicate CUDA compute mode enumerations */
#define VMDCUDA_COMPUTEMODE_DEFAULT     0
#define VMDCUDA_COMPUTEMODE_EXCLUSIVE   1
#define VMDCUDA_COMPUTEMODE_PROHIBITED  2

/* 
 * retrieve device properties 
 */
int vmd_cuda_device_props(int dev, char *name, int namelen,
                          int *revmajor, int *revminor, 
                          unsigned long *memb, int *clockratekhz,
                          int *smcount, int *integratedgpu,
                          int *asyncenginecount, int *kerneltimeout,
                          int *canmaphostmem, int *computemode);


/*
 * All available CUDA kernels
 */
void * vmd_cuda_devpool_setdevice(void * voidparms);

void * vmd_cuda_devpool_clear_device_mem(void *);

int vmd_cuda_madd_gflops(int numdevs, int *devlist, double *gflops, 
                         int testloops);

int vmd_cuda_bus_bw(int numdevs, int *devlist, 
                    double *hdmbsec, double *hdlatusec,
                    double *phdmbsec, double *phdlatusec,
                    double *dhmbsec, double *dhlatusec,
                    double *pdhmbsec, double *pdhlatusec);

int vmd_cuda_globmem_bw(int numdevs, int *devlist, 
                        double *memsetgbsec, double *memcpygbsec);

int vmd_cuda_devpool_latency(wkf_threadpool_t *devpool, int tilesize,
                             double *kernlaunchlatency,
                             double *barlatency,
                             double *cyclelatency, 
                             double *tilelatency,
                             double *kernellatency);

int vmd_cuda_measure_latencies(wkf_threadpool_t *devpool);

int vmd_cuda_vol_cpotential(long int natoms, float* atoms, float* grideners, 
                            long int numplane, long int numcol, long int numpt, 
                            float gridspacing);

int vmd_cuda_evaluate_occupancy_map(
    int mx, int my, int mz,             // map dimensions
    float *map,                         // buffer space for occupancy map
                                        // (length mx*my*mz floats)

    float max_energy,                   // max energy threshold
    float cutoff,                       // vdw cutoff distance
    float hx, float hy, float hz,       // map lattice spacing
    float x0, float y0, float z0,       // map origin
    float bx_1, float by_1, float bz_1, // inverse of atom bin lengths

    int nbx, int nby, int nbz,          // bin dimensions
    const float *bin,                   // atom bins XXX typecast to flint
                                        // (length BIN_SIZE*nbx*nby*nbz)
    const float *bin_zero,              // bin pointer shifted to origin

    int num_binoffsets,                 // number of offsets
    const char *binoffsets,             // bin neighborhood index offsets
                                        // (length 3*num_bin_offsets)

    int num_extras,                     // number of extra atoms
    const float *extra,                 // extra atoms from overfilled bins
                                        // XXX typecast to flint
                                        // (length BIN_SLOTSIZE*num_extras)

    int num_vdwparms,                   // number of vdw parameter types
    const float *vdwparms,              // vdw parameters
                                        // (length 2*num_vdw_params)

    int num_probes,                     // number of probe atoms
    const float *probevdwparms,         // vdw parameters of probe atoms
                                        // (length 2*num_probes)

    int num_conformers,                 // number of conformers
    const float *conformers             // probe atom offsets for conformers
                                        // (length 3*num_probes*num_conformers)
    );


int rdf_gpu(wkf_threadpool_t *devpool, // GPU worker thread pool
            int usepbc,                // periodic or non-periodic calc.
            int natoms1,               // array of the number of atoms in
                                       // selection 1 in each frame.
            float* xyz,                // coordinates of first selection.
                                       // [natoms1][3]
            int natoms2,               // array of the number of atoms in
                                       // selection 2 in each frame.
            float* xyz2,               // coordinates of selection 2.
                                       // [natoms2][3]
            float* cell,               // the cell x y and z dimensions [3]
            float* hist,               // the histograms, 1 per block
                                       // [ncudablocks][maxbin]
            int maxbin,                // the number of bins in the histogram
            float rmin,                // the minimum value of the first bin
            float delr);               // the width of each bin


#if defined(__cplusplus)
}
#endif

#endif

