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
 *      $RCSfile: MeasureRDF.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Code to compute radial distribution functions for MD trajectories.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Measure.h"
#include "AtomSel.h"
#include "utilities.h"
#include "ResizeArray.h"
#include "MoleculeList.h"
#include "Inform.h"
#include "Timestep.h"
#include "VMDApp.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAAccel.h"
#include "CUDAKernels.h"

#define MIN(X,Y) (((X)<(Y))? (X) : (Y))
#define MAX(X,Y) (((X)>(Y))? (X) : (Y))

/*! the volume of a spherical cap is defined as:
 * <pre> pi / 9 * h^2 * (3 * r  - h) </pre>
 * with h = height of cap = radius - boxby2.
 * \brief the volume of a sperical cap. */
static inline double spherical_cap(const double &radius, const double &boxby2) {
  return (VMD_PI / 3.0 * (radius - boxby2) * (radius - boxby2)
          * ( 2.0 * radius + boxby2));
}

void rdf_cpu(int natoms1,     // array of the number of atoms in
                              // selection 1 in each frame. 
             float* xyz,      // coordinates of first selection.
                              // [natoms1][3]
             int natoms2,     // array of the number of atoms in
                              // selection 2 in each frame. 
             float* xyz2,     // coordinates of selection 2.
                              // [natoms2][3]
             float* cell,     // the cell x y and z dimensions [3]
             float* hist,     // the histograms, 1 per block
                              // [ncudablocks][maxbin]
             int maxbin,      // the number of bins in the histogram
             float rmin,      // the minimum value of the first bin
             float delr)      // the width of each bin
{
  int iatom, jatom, ibin;
  float rij, rxij, rxij2, x1, y1, z1, x2, y2, z2;
  float cellx, celly, cellz;
  int *ihist = new int[maxbin];

  for (ibin=0; ibin<maxbin; ibin++) {
    ihist[ibin]=0;
  }

  cellx = cell[0];
  celly = cell[1];
  cellz = cell[2];

  for (iatom=0; iatom<natoms1; iatom++) {
    long addr = 3L * iatom;
    xyz[addr    ] = fmodf(xyz[addr    ], cellx);
    xyz[addr + 1] = fmodf(xyz[addr + 1], celly);
    xyz[addr + 2] = fmodf(xyz[addr + 2], cellz);
  }

  for (iatom=0; iatom<natoms2; iatom++) {
    long addr = 3L * iatom;
    xyz2[addr    ] = fmodf(xyz2[addr    ], cellx);
    xyz2[addr + 1] = fmodf(xyz2[addr + 1], celly);
    xyz2[addr + 2] = fmodf(xyz2[addr + 2], cellz);
  }

  for (iatom=0; iatom<natoms1; iatom++) {
    x1 = xyz[3L * iatom    ];
    y1 = xyz[3L * iatom + 1];
    z1 = xyz[3L * iatom + 2];
    for (jatom=0;jatom<natoms2;jatom++) {
      x2 = xyz2[3L * jatom    ];
      y2 = xyz2[3L * jatom + 1];
      z2 = xyz2[3L * jatom + 2];

      rxij = fabsf(x1 - x2);
      rxij2 = cellx - rxij;
      rxij = MIN(rxij, rxij2);
      rij = rxij * rxij;

      rxij = fabsf(y1 - y2);
      rxij2 = celly - rxij;
      rxij = MIN(rxij, rxij2);
      rij += rxij * rxij;

      rxij = fabsf(z1 - z2);
      rxij2 = cellz - rxij;
      rxij = MIN(rxij, rxij2);
      rij += rxij * rxij;

      rij = sqrtf(rij);

      ibin = (int)floorf((rij-rmin)/delr);
      if (ibin<maxbin && ibin>=0) {
        ++ihist[ibin];
      }
    }
  }

  delete [] ihist;
}




int measure_rdf(VMDApp *app,
                AtomSel *sel1, AtomSel *sel2, MoleculeList *mlist,
                const int count_h, double *gofr, 
                double *numint, double *histog,
                const float delta, int first, int last, int step, 
                int *framecntr, int usepbc, int selupdate) {
  int i, j, frame;

  float a, b, c, alpha, beta, gamma;
  float pbccell[3];
  int isortho=0;     // orthogonal unit cell not assumed by default.
  int duplicates=0;  // counter for duplicate atoms in both selections.
  float rmin = 0.0f; // min distance to histogram

  // initialize a/b/c/alpha/beta/gamma to arbitrary defaults to please the compiler.
  a=b=c=9999999.0;
  alpha=beta=gamma=90.0;

  // reset counter for total, skipped, and _orth processed frames.
  framecntr[0]=framecntr[1]=framecntr[2]=0;

  // First round of sanity checks.
  // neither list can be undefined
  if (!sel1 || !sel2 ) {
    return MEASURE_ERR_NOSEL;
  }

  // make sure that both selections are from the same molecule
  // so that we know that PBC unit cell info is the same for both
  if (sel2->molid() != sel1->molid()) {
    return MEASURE_ERR_MISMATCHEDMOLS;
  }

  Molecule *mymol = mlist->mol_from_id(sel1->molid());
  int maxframe = mymol->numframes() - 1;
  int nframes = 0;

  if (last == -1)
    last = maxframe;

  if ((last < first) || (last < 0) || (step <=0) || (first < -1)
      || (maxframe < 0) || (last > maxframe)) {
      msgErr << "measure rdf: bad frame range given."
             << " max. allowed frame#: " << maxframe << sendmsg;
    return MEASURE_ERR_BADFRAMERANGE;
  }

  // test for non-orthogonal PBC cells, zero volume cells, etc.
  if (usepbc) {
    for (isortho=1, nframes=0, frame=first; frame <=last; ++nframes, frame += step) {
      const Timestep *ts;

      if (first == -1) {
        // use current frame only. don't loop.
        ts = sel1->timestep(mlist);
        frame=last;
      } else {
        ts = mymol->get_frame(frame);
      }
      // get periodic cell information for current frame
      a = ts->a_length;
      b = ts->b_length;
      c = ts->c_length;
      alpha = ts->alpha;
      beta = ts->beta;
      gamma = ts->gamma;

      // check validity of PBC cell side lengths
      if (fabsf(a*b*c) < 0.0001) {
        msgErr << "measure rdf: unit cell volume is zero." << sendmsg;
        return MEASURE_ERR_GENERAL;
      }

      // check PBC unit cell shape to select proper low level algorithm.
      if ((alpha != 90.0) || (beta != 90.0) || (gamma != 90.0)) {
        isortho=0;
      }
    }
  } else {
    // initialize a/b/c/alpha/beta/gamma to arbitrary defaults
    isortho=1;
    a=b=c=9999999.0;
    alpha=beta=gamma=90.0;
  }

  // until we can handle non-orthogonal periodic cells, this is fatal
  if (!isortho) {
    msgErr << "measure rdf: only orthorhombic cells are supported (for now)." << sendmsg;
    return MEASURE_ERR_GENERAL;
  }

  // clear the result arrays
  for (i=0; i<count_h; ++i) {
    gofr[i] = numint[i] = histog[i] = 0.0;
  }

  // pre-allocate coordinate buffers of the max size we'll
  // ever need, so we don't have to reallocate if/when atom
  // selections are updated on-the-fly
  float *sel1coords = new float[3L*sel1->num_atoms];
  float *sel2coords = new float[3L*sel2->num_atoms];
  float *lhist = new float[count_h];

  for (nframes=0,frame=first; frame <=last; ++nframes, frame += step) {
    const Timestep *ts1, *ts2;

    if (frame  == -1) {
      // use current frame only. don't loop.
      ts1 = sel1->timestep(mlist);
      ts2 = sel2->timestep(mlist);
      frame=last;
    } else {
      sel1->which_frame = frame;
      sel2->which_frame = frame;
      ts1 = ts2 = mymol->get_frame(frame); // requires sels from same mol
    }

    if (usepbc) {
      // get periodic cell information for current frame
      a     = ts1->a_length;
      b     = ts1->b_length;
      c     = ts1->c_length;
      alpha = ts1->alpha;
      beta  = ts1->beta;
      gamma = ts1->gamma;
    }

    // compute half periodic cell size
    float boxby2[3];
    boxby2[0] = 0.5f * a;
    boxby2[1] = 0.5f * b;
    boxby2[2] = 0.5f * c;

    // update the selections if the user desires it
    if (selupdate) {
      if (sel1->change(NULL, mymol) != AtomSel::PARSE_SUCCESS)
        msgErr << "measure rdf: failed to evaluate atom selection update";
      if (sel2->change(NULL, mymol) != AtomSel::PARSE_SUCCESS)
        msgErr << "measure rdf: failed to evaluate atom selection update";
    }

    // check for duplicate atoms in the two lists, as these will have
    // to be subtracted back out of the first histogram slot
    if (sel2->molid() == sel1->molid()) {
      int i;
      for (i=0, duplicates=0; i<sel1->num_atoms; ++i) {
        if (sel1->on[i] && sel2->on[i])
          ++duplicates;
      }
    }

    // copy selected atoms to the two coordinate lists
    // requires that selections come from the same molecule
    const float *framepos = ts1->pos;
    for (i=0, j=0; i<sel1->num_atoms; ++i) {
      if (sel1->on[i]) {
        long a = i*3L;
        sel1coords[j    ] = framepos[a    ];
        sel1coords[j + 1] = framepos[a + 1];
        sel1coords[j + 2] = framepos[a + 2];
        j+=3;
      }
    }
    framepos = ts2->pos;
    for (i=0, j=0; i<sel2->num_atoms; ++i) {
      if (sel2->on[i]) {
        long a = i*3L;
        sel2coords[j    ] = framepos[a    ];
        sel2coords[j + 1] = framepos[a + 1];
        sel2coords[j + 2] = framepos[a + 2];
        j+=3;
      }
    }

    // copy unit cell information
    pbccell[0]=a;
    pbccell[1]=b;
    pbccell[2]=c;

    // clear the histogram for this frame
    memset(lhist, 0, count_h * sizeof(float));

    if (isortho && sel1->selected && sel2->selected) {
      // do the rdf calculation for orthogonal boxes.
      // XXX. non-orthogonal box not supported yet. detected and handled above.
      int rc=-1;
#if defined(VMDCUDA)
      if (!getenv("VMDNOCUDA") && (app->cuda != NULL)) {
//        msgInfo << "Running multi-GPU RDF calc..." << sendmsg;
        rc=rdf_gpu(app->cuda->get_cuda_devpool(),
                   usepbc,
                   sel1->selected, sel1coords,
                   sel2->selected, sel2coords, 
                   pbccell,
                   lhist,
                   count_h,
                   rmin,
                   delta);
      } 
#endif
      if (rc != 0) {
//        msgInfo << "Running single-core CPU RDF calc..." << sendmsg;
        rdf_cpu(sel1->selected, sel1coords,
                sel2->selected, sel2coords, 
                pbccell,
                lhist,
                count_h,
                rmin,
                delta);
      }

      ++framecntr[2]; // frame processed with rdf algorithm
    } else {
      ++framecntr[1]; // frame skipped
    }
    ++framecntr[0];   // total frames.

#if 0
    // XXX elimination of duplicates is now handled within the 
    //     GPU kernels themselves, so we do not need to subtract them
    //     off during the histogram normalization calculations.
    // Correct the first histogram slot for the number of atoms that are
    // present in both lists. they'll end up in the first histogram bin.
    // we subtract only from the first thread histogram which is always defined.
    lhist[0] -= duplicates;
#endif

    // in case of going 'into the edges', we should cut
    // off the part that is not properly normalized to
    // not confuse people that don't know about this.
    int h_max=count_h;
    float smallside=a;
    if (isortho && usepbc) {
      if(b < smallside) {
        smallside=b;
      }
      if(c < smallside) {
        smallside=c;
      }
      h_max=(int) (sqrtf(0.5f)*smallside/delta) +1;
      if (h_max > count_h) {
        h_max=count_h;
      }
    }

    // compute normalization function.
    double all=0.0;
    double pair_dens = 0.0;
    
    if (sel1->selected && sel2->selected) {
      if (usepbc) {
        pair_dens = a * b * c / ((double)sel1->selected * (double)sel2->selected - (double)duplicates);
      } else { // assume a particle volume of 30 \AA^3 (~ 1 water).
        pair_dens = 30.0 * (double)sel1->selected /
          ((double)sel1->selected * (double)sel2->selected - (double)duplicates);
      }
    }

    // XXX for orthogonal boxes, we can reduce this to rmax < sqrt(0.5)*smallest side
    for (i=0; i<h_max; ++i) {
      // radius of inner and outer sphere that form the spherical slice
      double r_in  = delta * (double)i;
      double r_out = delta * (double)(i+1);
      double slice_vol = 4.0 / 3.0 * VMD_PI
        * ((r_out * r_out * r_out) - (r_in * r_in * r_in));

      if (isortho && usepbc) {
        // add correction for 0.5*box < r <= sqrt(0.5)*box
        if (r_out > boxby2[0]) {
          slice_vol -= 2.0 * spherical_cap(r_out, boxby2[0]);
        }
        if (r_out > boxby2[1]) {
          slice_vol -= 2.0 * spherical_cap(r_out, boxby2[1]);
        }
        if (r_out > boxby2[2]) {
          slice_vol -= 2.0 * spherical_cap(r_out, boxby2[2]);
        }
        if (r_in > boxby2[0]) {
          slice_vol += 2.0 * spherical_cap(r_in, boxby2[0]);
        }
        if (r_in > boxby2[1]) {
          slice_vol += 2.0 * spherical_cap(r_in, boxby2[1]);
        }
        if (r_in > boxby2[2]) {
          slice_vol += 2.0 * spherical_cap(r_in, boxby2[2]);
        }
      }

      double normf = pair_dens / slice_vol;
      double histv = (double) lhist[i];
      gofr[i] += normf * histv;
      all     += histv;
      if (sel1->selected) {
        numint[i] += all / (double)(sel1->selected);
      }
      histog[i] += histv;
    }
  }
  delete [] sel1coords;
  delete [] sel2coords;
  delete [] lhist;

  double norm = 1.0 / (double) nframes;
  for (i=0; i<count_h; ++i) {
    gofr[i]   *= norm;
    numint[i] *= norm;
    histog[i] *= norm;
  }

  return MEASURE_NOERR;
}

