/***************************************************************************
 *cr
 *cr            (C) Copyright 2007-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: MDFF.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.12 $      $Date: 2015/05/21 03:35:35 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Multi-core CPU versions of MDFF functions
 *
 * "GPU-Accelerated Analysis and Visualization of Large Structures
 *  Solved by Molecular Dynamics Flexible Fitting"
 *  John E. Stone, Ryan McGreevy, Barry Isralewitz, and Klaus Schulten.
 *  Faraday Discussion 169, 2014. (In press)
 *  Online full text available at http://dx.doi.org/10.1039/C4FD00005F
 * 
 ***************************************************************************/


#include <tcl.h>
#include "TclCommands.h"
#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "Molecule.h"
#include "VolumetricData.h"
#include "VolMapCreate.h"
#include "QuickSurf.h"
#include <math.h>
#include "MDFF.h"

typedef struct{
  double mapA_sum;
  double mapB_sum;
  double mapA_ss;
  double mapB_ss;
  double cc;
  int size;
  const VolumetricData *targetVol;
  float *volmap;
  const int *numvoxels;
  VolumetricData *qsVol;
  double threshold;
  wkf_mutex_t mtx;
} ccparms;

static void * correlationthread(void *voidparms) {
  wkf_tasktile_t tile;
  ccparms *parms = NULL;
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);

  int gx,gy,gz;
  float ix,iy,iz;

  double origin[3] = {0.0, 0.0, 0.0};
  double delta[3] = {0.0, 0.0, 0.0};
  origin[0] = parms->qsVol->origin[0];
  origin[1] = parms->qsVol->origin[1];
  origin[2] = parms->qsVol->origin[2];
  delta[0] = parms->qsVol->xaxis[0] / (parms->qsVol->xsize - 1);
  delta[1] = parms->qsVol->yaxis[1] / (parms->qsVol->ysize - 1);
  delta[2] = parms->qsVol->zaxis[2] / (parms->qsVol->zsize - 1);

  double lmapA_sum = 0.0f;
  double lmapB_sum = 0.0f;
  double lmapA_ss = 0.0f;
  double lmapB_ss = 0.0f;
  double lcc = 0.0f;
  int lsize = 0;
  while (wkf_threadlaunch_next_tile(voidparms, 16384, &tile) != WKF_SCHED_DONE) { 
    int x;
    for (x=tile.start; x<tile.end; x++) {
      int xsize = parms->numvoxels[0];
      int ysize = parms->numvoxels[1];
       
      gz = x / (ysize*xsize);
      gy = (x % (ysize*xsize)) / xsize;
      gx = x % xsize;
#if 0
      parms->qsVol->voxel_coord(gx,gy,gz,ix,iy,iz);
#else
      ix = origin[0] + (gx * delta[0]);
      iy = origin[1] + (gy * delta[1]);
      iz = origin[2] + (gz * delta[2]);
#endif

      float voxelA = parms->volmap[x];
      float voxelB = parms->targetVol->voxel_value_interpolate_from_coord(ix,iy,iz);
//      float voxelB = parms->targetVol->voxel_value_from_coord(ix,iy,iz);

      // checks for nans (nans always false when self compared)
      if (voxelB == voxelB) {
// XXX what's up with this test vs. NULL?:
//          if(parms->threshold == NULL){ 
//          if (parms->threshold == 0.0){ 
//            lmapA_sum += voxelA;
//            lmapB_sum += voxelB;
//            lmapA_ss += voxelA*voxelA;
//            lmapB_ss += voxelB*voxelB;
//            lcc += voxelA*voxelB;
//            lsize++;
            // Should be voxel B? thresholding simulated aren't we?
            // Thats why the old style NaN check is there.
//      } else if(voxelB >= parms->threshold) { 
        if (voxelA >= parms->threshold) {
          lmapA_sum += voxelA;
          lmapB_sum += voxelB;
          lmapA_ss += voxelA*voxelA;
          lmapB_ss += voxelB*voxelB;
          lcc += voxelA*voxelB;
          lsize++;
        }
      }
    }
  }

  wkf_mutex_lock(&parms->mtx); 
  parms->mapA_sum += lmapA_sum;
  parms->mapB_sum += lmapB_sum;    
  parms->mapA_ss += lmapA_ss;
  parms->mapB_ss += lmapB_ss;
  parms->cc += lcc;
  parms->size += lsize;
  wkf_mutex_unlock(&parms->mtx);

  return NULL;
}


int cc_threaded(VolumetricData *qsVol, const VolumetricData *targetVol, double *cc, double threshold) {
  ccparms parms;
  memset(&parms, 0, sizeof(parms));
  parms.mapA_sum = 0;
  parms.mapB_sum = 0;
  parms.mapA_ss = 0;
  parms.mapB_ss = 0;
  parms.cc = 0;
  parms.size = 0;
  parms.volmap = qsVol->data;
  int numvoxels [] = {qsVol->xsize, qsVol->ysize, qsVol->zsize};
  parms.numvoxels = numvoxels;
  parms.targetVol = targetVol;
  parms.qsVol = qsVol;
  parms.threshold = threshold;

  int physprocs = wkf_thread_numprocessors();
  int maxprocs = physprocs;
  float *voltexmap = NULL;

  // We can productively use only a few cores per socket due to the
  // limited memory bandwidth per socket. Also, hyperthreading
  // actually hurts performance.  These two considerations combined
  // with the linear increase in memory use prevent us from using large
  // numbers of cores with this simple approach, so if we've got more 
  // than 8 CPU cores, we'll iteratively cutting the core count in 
  // half until we're under 8 cores.
  while (maxprocs > 8) 
    maxprocs /= 2;

  // Limit the number of CPU cores used so we don't run the 
  // machine out of memory during surface computation.
  // Use either a dynamic or hard-coded heuristic to limit the
  // number of CPU threads we will spawn so that we don't run
  // the machine out of memory.  
  long volsz = numvoxels[0] * numvoxels[1] * numvoxels[2];
  long volmemsz = sizeof(float) * volsz;
  long volmemszkb = volmemsz / 1024;
  long volmemtexszkb = volmemszkb + ((voltexmap != NULL) ? 3*volmemszkb : 0);

  // Platforms that don't have a means of determining available
  // physical memory will return -1, in which case we fall back to the
  // simple hard-coded 2GB-max-per-core heuristic.
  long vmdcorefree = -1;

#if defined(ARCH_BLUEWATERS) || defined(ARCH_CRAY_XC) || defined(ARCH_CRAY_XK) || defined(ARCH_LINUXAMD64) || defined(ARCH_SOLARIS2_64) || defined(ARCH_SOLARISX86_64) || defined(ARCH_AIX6_64) || defined(ARCH_MACOSXX86_64) 
  // XXX The core-free query scheme has one weakness in that we might have a 
  // 32-bit version of VMD running on a 64-bit machine, where the available
  // physical memory may be much larger than is possible for a 
  // 32-bit VMD process to address.  To do this properly we must therefore
  // use conditional compilation safety checks here until we  have a better
  // way of determining this with a standardized helper routine.
  vmdcorefree = vmd_get_avail_physmem_mb();
#endif

  if (vmdcorefree >= 0) {
    // Make sure QuickSurf uses no more than a fraction of the free memory
    // as an upper bound alternative to the hard-coded heuristic.
    // This should be highly preferable to the fixed-size heuristic
    // we had used in all cases previously.
    while ((volmemtexszkb * maxprocs) > (1024*vmdcorefree/4)) {
      maxprocs /= 2;
    }
  } else {
    // Set a practical per-core maximum memory use limit to 2GB, for all cores
    while ((volmemtexszkb * maxprocs) > (2 * 1024 * 1024))
      maxprocs /= 2;
  }

  if (maxprocs < 1) 
    maxprocs = 1;

  int numprocs = maxprocs; // ever the optimist
  wkf_mutex_init(&parms.mtx);
  wkf_tasktile_t tile;
  tile.start = 0;
  tile.end = volsz;
  wkf_threadlaunch(numprocs, &parms, correlationthread, &tile);
  wkf_mutex_destroy(&parms.mtx);

  int size = parms.size;
  double aMean = parms.mapA_sum/size;
  double bMean = parms.mapB_sum/size;
  double stdA = sqrt(parms.mapA_ss/size - aMean*aMean);
  double stdB = sqrt(parms.mapB_ss/size - bMean*bMean);

  *cc = ((parms.cc) - size*aMean*bMean)/(size * stdA * stdB);
  return 0;
}



