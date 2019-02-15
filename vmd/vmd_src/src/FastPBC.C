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
 *      $RCSfile: FastPBC.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Fast PBC wrapping code
 ***************************************************************************/
#include <stdio.h>
#include <math.h>
#include "Measure.h"
#include "FastPBC.h"

#ifdef VMDOPENACC
#include <accelmath.h>
#endif

///
/// XXX the current implementation can't handle on-the-fly fall-back to CPU,
///     and error handling is quite limited.
/// 

// If there is no CUDA defined, fall back to the CPU version.
#ifndef VMDCUDA
void fpbc_exec_unwrap(Molecule* mol, int first, int last, int sellen, int* indexlist) {
  fpbc_exec_unwrap_cpu(mol, first, last, sellen, indexlist);
}

void fpbc_exec_wrapatomic(Molecule* mol, int first, int last, int sellen, int* indexlist, 
  float* weights, AtomSel* csel, float* center) {
  fpbc_exec_wrapatomic_cpu(mol, first, last, sellen, indexlist, weights, csel, center);
}

void fpbc_exec_wrapcompound(Molecule* mol, int first, int last, int fnum, int *compoundmap, int sellen, int* indexlist,
  float* weights, AtomSel* csel, float* center, float* massarr) {
  fpbc_exec_wrapcompound_cpu(mol, first, last, fnum, compoundmap, sellen, indexlist, weights, csel, center, massarr);
}

void fpbc_exec_join(Molecule* mol, int first, int last, int fnum, int *compoundmap, int sellen, int* indexlist) {
  fpbc_exec_join_cpu(mol, first, last, fnum, compoundmap, sellen, indexlist);
}

void fpbc_exec_recenter(Molecule* mol, int first, int last, int csellen, int* cindexlist, int fnum, int *compoundmap, int sellen, int* indexlist, float* weights, AtomSel* csel, float* massarr) {
  fpbc_exec_recenter_cpu(mol, first, last, csellen, cindexlist, fnum, compoundmap, sellen, indexlist,  weights, csel, massarr);
}
#endif

void fpbc_exec_wrapatomic_cpu(Molecule* mol, int first, int last, int sellen, int* indexlist, 
  float* weights, AtomSel* csel, float* center) {
  int f, i, j, k;
  float boxsize[3];
  float invboxsize[3];
  for (f=first; f<=last; f++) {
    Timestep *ts = mol->get_frame(f);
    boxsize[0] = ts->a_length;
    boxsize[1] = ts->b_length;
    boxsize[2] = ts->c_length;
    for (j=0; j < 3; j++) {
      invboxsize[j] = 1.0/boxsize[j];
    }
    //If the center of mass needs to be found, find it! There is a function for that. :D
    if (csel != NULL) {
      measure_center(csel, ts->pos, weights, center);
    }
    for (k=0; k<sellen; k++) {
      i = indexlist[k];
      for (j=0; j < 3; j++) {
        //Compute the shift in terms of the number of box-lengths, scale by it, and reposition.
        ts->pos[i*3+j] = ts->pos[i*3+j] - (roundf((ts->pos[i*3+j] - center[j]) * invboxsize[j]) * boxsize[j]);
      }
    }
  }
}


void fpbc_exec_wrapcompound_cpu(Molecule* mol, int first, int last, int fnum, int *compoundmap, int sellen, int* indexlist,
  float* weights, AtomSel* csel, float* center, float* massarr) {
  int f, i, j, k, l;
  float boxsize[3];
  float invboxsize[3];
  for (f=first; f<=last; f++) {
    Timestep *ts = mol->get_frame(f);
    boxsize[0] = ts->a_length;
    boxsize[1] = ts->b_length;
    boxsize[2] = ts->c_length;
    for (j=0; j < 3; j++) {
      invboxsize[j] = 1.0/boxsize[j];
    }
    //If the center of mass needs to be found, find it! There is a function for that. :D
    if (csel != NULL) {
      measure_center(csel, ts->pos, weights, center);
    }
    for (l = 0; l < fnum; l++) {
      int lowbound = compoundmap[l];
      int highbound = compoundmap[l+1];
      //calculate the mass weighted center of the compound
      float cmass = 0;            // initialize mass accumulator
      float ccenter[3] = {0,0,0}; // initialize position accumulator
      // cycle through atoms
      for (k = lowbound; k < highbound; k++ ) {
        i = indexlist[k];
        for (j=0; j < 3; j++) {
          ccenter[j] += massarr[i] * (ts->pos[i*3+j]);
        }
        cmass += massarr[i];
      }
      cmass = 1.0 / cmass;
      // divide pos by mass to get final mass weighted com
      for (j=0; j < 3; j++) {
        ccenter[j] *= cmass;
      }
      //move the compound
      for (k = lowbound; k < highbound; k++ ) {
        i = indexlist[k];
        for (j=0; j < 3; j++) {
          ts->pos[i*3+j] = ts->pos[i*3+j] - (roundf((ccenter[j] - center[j]) * invboxsize[j]) * boxsize[j]);
        }
      }
    }
  }
}


void fpbc_exec_join_cpu(Molecule* mol, int first, int last, int fnum, int *compoundmap, int sellen, int* indexlist) {
  int f, i, j, k, l;
  float boxsize[3];
  float invboxsize[3];
  float *pos;
  Timestep *ts;
  #pragma acc data copyin (indexlist[sellen], compoundmap[fnum+1])
  {
    //Loop over all the frames
    for (f=first; f<=last; f++) {
      //Grab the current coordinates and boxlength.
      ts = mol->get_frame(f);
      boxsize[0] = ts->a_length;
      boxsize[1] = ts->b_length;
      boxsize[2] = ts->c_length;
      pos = ts->pos;
      #pragma acc data copy(pos[3*mol->nAtoms]) copyin(boxsize[3]) create(invboxsize[3])  async(f%2)
      {
        #pragma acc kernels private(i, j, k, l) async(f%2)
        {
          //Minimize divisions by computing an inverse box-length.
          for (int q=0; q < 3; q++) {
            invboxsize[q] = 1.0/boxsize[q];
          }
          #pragma acc loop independent private(i, j, k, l)
          for (l = 0; l < fnum; l++) {
            float ccenter[3];
            int lowbound = compoundmap[l];
            int highbound = compoundmap[l+1];
            i = indexlist[lowbound];
            //Use the first element within the compound as the center.
            for (j=0; j < 3; j++) {
              ccenter[j] = pos[(i+((highbound-lowbound)/2))*3+j];
            }
            //move the compound, wrapping it to be within half a box dimension from the center
            for (k = lowbound; k < highbound; k++ ) {
              i = indexlist[k];
              for (j=0; j < 3; j++) {
                pos[i*3+j] = pos[i*3+j] - (rintf((pos[i*3+j] - ccenter[j]) * invboxsize[j]) * boxsize[j]);
              }
            }
          }
        }
      }
    }
    #pragma acc wait
  }
}


void fpbc_exec_unwrap_cpu(Molecule* mol, int first, int last, int sellen, int* indexlist) {
  Timestep *ts, *prev;
  int f, i, j, idx;
  float boxsize[3];
  float invboxsize[3];
  for (f=first+1; f<=last; f++) {
    ts = mol->get_frame(f);
    prev = mol->get_frame(f-1);
    boxsize[0] = ts->a_length;
    boxsize[1] = ts->b_length;
    boxsize[2] = ts->c_length;
    for (j=0; j < 3; j++) {
      invboxsize[j] = 1.0/boxsize[j];
    }
    for (i=0; i<sellen; i++) {
      idx = indexlist[i];
      for (j=0; j < 3; j++) {
        ts->pos[idx*3+j] = ts->pos[idx*3+j] - (rintf((ts->pos[idx*3+j] - prev->pos[idx*3+j]) * invboxsize[j]) * boxsize[j]);
      }
    }
  }
}


void fpbc_exec_recenter_cpu(Molecule* mol, int first, int last, int csellen, int* cindexlist, int fnum, int *compoundmap, int sellen, int* indexlist, float* weights, AtomSel* csel, float* massarr) {
  float center[3];//This needs to be passed, but since csel is guaranteed to point somewhere,
  //it does not need to be set.
  fpbc_exec_unwrap_cpu(mol, first, last, csellen, cindexlist);
  if (fnum)
    fpbc_exec_wrapcompound_cpu(mol, first, last, fnum, compoundmap, sellen, indexlist, weights, csel, &center[0], massarr);
  else
    fpbc_exec_wrapatomic_cpu(mol, first, last, sellen, indexlist, weights, csel, &center[0]);
}




