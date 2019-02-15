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
 *      $RCSfile: MDFF.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.15 $      $Date: 2019/01/17 21:38:55 $
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

#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "Molecule.h"
#include "VolumetricData.h"
#include "VolMapCreate.h"
#include "QuickSurf.h"
#include <math.h>
#include "MDFF.h"
#include "Voltool.h"
#include <stdio.h>
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
  VolumetricData *newvol;
  double threshold;
  wkf_mutex_t mtx;
} ccparms;

static void * correlationthread(void *voidparms) {
  wkf_tasktile_t tile;
  ccparms *parms = NULL;
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);
  double lmapA_sum = 0.0f;
  double lmapB_sum = 0.0f;
  double lmapA_ss = 0.0f;
  double lmapB_ss = 0.0f;
  double lcc = 0.0f;
  int lsize = 0;
  while (wkf_threadlaunch_next_tile(voidparms, 16384, &tile) != WKF_SCHED_DONE) { 
    int x;
    for (x=tile.start; x<tile.end; x++) {
      float ix,iy,iz;
      voxel_coord(x, ix, iy, iz, parms->newvol);     
      float voxelA = parms->qsVol->voxel_value_interpolate_from_coord(ix,iy,iz);
      float voxelB = parms->targetVol->voxel_value_interpolate_from_coord(ix,iy,iz);
      // checks for nans (nans always false when self compared)
      if (!myisnan(voxelB) && !myisnan(voxelA)) {
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
 

  //get intersection map so we look at all the overlapping voxels
  double origin[3] = {0., 0., 0.};
  double xaxis[3] = {0., 0., 0.};
  double yaxis[3] = {0., 0., 0.};
  double zaxis[3] = {0., 0., 0.};
  int numvoxelstmp [3] = {0, 0, 0};
  float *data = NULL;
  VolumetricData *newvol  = new VolumetricData("density map", origin, xaxis, yaxis, zaxis,
                                 numvoxelstmp[0], numvoxelstmp[1], numvoxelstmp[2],
                                 data);
  init_from_intersection(qsVol, targetVol, newvol);
  parms.newvol = newvol;
  long volsz = newvol->xsize*newvol->ysize*newvol->zsize;
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



