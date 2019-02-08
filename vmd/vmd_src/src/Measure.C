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
 *      $RCSfile: Measure.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.149 $       $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Code to measure atom distances, angles, dihedrals, etc.
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Measure.h"
#include "AtomSel.h"
#include "Matrix4.h"
#include "utilities.h"
#include "fitrms.h"
#include "ResizeArray.h"
#include "SpatialSearch.h"  // for find_within()
#include "MoleculeList.h"
#include "Inform.h"
#include "Timestep.h"
#include "VMDApp.h"
#include "WKFThreads.h"
#include "WKFUtils.h"

// Standard functions available to everyone
static const char *measure_error_messages[] = {
  "no atom selection",                                  // -1
  "no atoms in selection",                              // -2
  "incorrect number of weights for selection",          // -3
  "internal error: NULL pointer given",                 // -4
  "bad weight sum, would cause divide by zero",         // -5
  "molecule was deleted(?)",                            // -6
  "cannot understand weight parameter",                 // -7
  "non-number given as parameter",                      // -8
  "two selections don't have the same number of atoms", // -9
  "internal error: out of range",                       // -10
  "no coordinates in selection",                        // -11
  "couldn't compute eigenvalue/vectors",                // -12
  "unknown Tcl problem",                                // -13
  "no atom radii",                                      // -14
  "order parameter contains out-of-range atom index",   // -15
  "order parameter not supported in old RMS fit mode",  // -16
  "specified frames are out of range",                  // -17
  "both selections must reference the same molecule",   // -18
  "one atom appears twice in list",                     // -19
  "molecule contains no frames",                        // -20
  "invalid atom id",                                    // -21
  "cutoff must be smaller than cell dimension",         // -22
  "Zero volmap gridsize"                                // -23
};
  
const char *measure_error(int errnum) {
  if (errnum >= 0 || errnum < -23) 
    return "bad error number";
  return measure_error_messages[-errnum - 1];
}

int measure_move(const AtomSel *sel, float *framepos, const Matrix4 &mat) {
  if (!sel)       return MEASURE_ERR_NOSEL;
  if (!framepos)  return MEASURE_ERR_NOFRAMEPOS;

  // and apply it to the coordinates
  int i;
  float *pos = framepos;
  if (mat.mat[3]==0 && mat.mat[7]==0 && mat.mat[11]==0 && mat.mat[15]==1) {
    // then this is just a rotation followed by a translation.  Do it
    // the fast way.
    const float dx = mat.mat[12];
    const float dy = mat.mat[13];
    const float dz = mat.mat[14];
    const float *m = mat.mat;
    if (sel->selected == sel->num_atoms) {
      for (i=0; i<sel->num_atoms; i++) {
        float x = pos[0]*m[0] + pos[1]*m[4] + pos[2]*m[8] + dx;
        float y = pos[0]*m[1] + pos[1]*m[5] + pos[2]*m[9] + dy;
        float z = pos[0]*m[2] + pos[1]*m[6] + pos[2]*m[10] + dz;
        pos[0] = x;
        pos[1] = y;
        pos[2] = z;
        pos += 3;
      }
    } else {
      pos += sel->firstsel*3L;
      for (i=sel->firstsel; i<=sel->lastsel; i++) {
        if (sel->on[i]) {
          float x = pos[0]*m[0] + pos[1]*m[4] + pos[2]*m[8] + dx;
          float y = pos[0]*m[1] + pos[1]*m[5] + pos[2]*m[9] + dy;
          float z = pos[0]*m[2] + pos[1]*m[6] + pos[2]*m[10] + dz;
          pos[0] = x;
          pos[1] = y;
          pos[2] = z;
        }
        pos += 3;
      }
    }
  } else {
    pos += sel->firstsel*3L;
    for (i=sel->firstsel; i<=sel->lastsel; i++) {
      if (sel->on[i]) {
        mat.multpoint3d(pos, pos);
      }
      pos += 3;
    }
  }
  return MEASURE_NOERR;
}

// compute the sum of a set of weights which correspond either to
// all atoms, or to selected atoms
int measure_sumweights(const AtomSel *sel, int numweights, 
                       const float *weights, float *weightsum) {
  if (!sel)       return MEASURE_ERR_NOSEL;
  if (!weights)   return MEASURE_ERR_NOWEIGHT;
  if (!weightsum) return MEASURE_ERR_GENERAL;

  double sum = 0;

  if (numweights == sel->num_atoms) {
    int i;
    for (i=sel->firstsel; i<=sel->lastsel; i++) {
      if (sel->on[i]) {
        sum += weights[i];
      }
    }
  } else if (numweights == sel->selected) {
    int i, j;
    for (j=0,i=sel->firstsel; i<=sel->lastsel; i++) {
      if (sel->on[i]) {
        sum += weights[j];
        j++;
      }
    }
  } else {
    return MEASURE_ERR_BADWEIGHTPARM;  
  }

  *weightsum = (float) sum;
  return MEASURE_NOERR;
}

extern int measure_center_perresidue(MoleculeList *mlist, const AtomSel *sel, const float *framepos,
                   const float *weight, float *com) {
  if (!sel)      return MEASURE_ERR_NOSEL;
  if (!framepos) return MEASURE_ERR_NOFRAMEPOS;
  if (!weight)   return MEASURE_ERR_NOWEIGHT;
  if (!com)      return MEASURE_ERR_NOCOM;

  Molecule *mol = mlist->mol_from_id(sel->molid());
  int residue = mol->atom(sel->firstsel)->uniq_resid;
  int rescount = 0;
  int i, j = 0;
  float x=0, y=0, z=0, w=0;
  for (i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      if (residue != mol->atom(i)->uniq_resid) {
        com[3*rescount    ] = x / w;
        com[3*rescount + 1] = y / w;
        com[3*rescount + 2] = z / w;
        residue = mol->atom(i)->uniq_resid;
        rescount++;
        x = 0;
        y = 0;
        z = 0;
        w = 0;
      }
      float tw = weight[j];
      w += tw;
      x += tw * framepos[3*i  ];
      y += tw * framepos[3*i+1];
      z += tw * framepos[3*i+2];
      j++;
    }
  }
  com[3*rescount    ] = x / w;
  com[3*rescount + 1] = y / w;
  com[3*rescount + 2] = z / w;
  rescount++;
  return rescount;
}

// compute the center of mass
// return -5 if total weight == 0, otherwise 0 for success.
int measure_center(const AtomSel *sel, const float *framepos,
                   const float *weight, float *com) {
  if (!sel)      return MEASURE_ERR_NOSEL;
  if (!framepos) return MEASURE_ERR_NOFRAMEPOS;
  if (!weight)   return MEASURE_ERR_NOWEIGHT;
  if (!com)      return MEASURE_ERR_NOCOM;   

  // use double precision floating point in case we have a large selection
  int i, j;
  double x, y, z, w;
  j=0;
  w=x=y=z=0;
  for (i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      float tw = weight[j];
      w += double(tw);
      x += double(tw * framepos[3L*i  ]);
      y += double(tw * framepos[3L*i+1]);
      z += double(tw * framepos[3L*i+2]);
      j++;
    }
  }

  if (w == 0) {
    return MEASURE_ERR_BADWEIGHTSUM;
  }

  x/=w;
  y/=w;
  z/=w;

  com[0] = float(x);
  com[1] = float(y);
  com[2] = float(z);

  return MEASURE_NOERR;
}


// compute the axis aligned aligned bounding box for the selected atoms
int measure_minmax(int num, const int *on, const float *framepos, 
                   const float *radii, float *min_coord, float *max_coord) {
  int i;
  float minx, miny, minz;
  float maxx, maxy, maxz;

  if (!on)                      return MEASURE_ERR_NOSEL;
  if (num == 0)                 return MEASURE_ERR_NOATOMS;
  if (!min_coord || !max_coord) return MEASURE_ERR_NOMINMAXCOORDS;

  vec_zero(min_coord);
  vec_zero(max_coord);

  // find first selected atom
  int firstsel = 0;
  int lastsel = -1;
  if (find_first_selection_aligned(num, on, &firstsel))
    return MEASURE_NOERR; // no atoms selected

  if (find_last_selection_aligned(num, on, &lastsel))
    return MEASURE_NOERR; // no atoms selected or internal inconsistency

  // initialize minmax coords to the first selected atom
  i=firstsel;
  if (radii == NULL) {
    // calculate bounding box of atom centers
    minx = maxx = framepos[i*3L  ];
    miny = maxy = framepos[i*3L+1];
    minz = maxz = framepos[i*3L+2];
  } else {
    // calculate bounding box for atoms /w given radii 
    minx = framepos[i*3L  ] - radii[i];
    maxx = framepos[i*3L  ] + radii[i];
    miny = framepos[i*3L+1] - radii[i];
    maxy = framepos[i*3L+1] + radii[i];
    minz = framepos[i*3L+2] - radii[i];
    maxz = framepos[i*3L+2] + radii[i];
  }

  // continue looping from there until we finish
  if (radii == NULL) {
    // calculate bounding box of atom centers
#if 0
    minmax_selected_3fv_aligned(framepos, on, atomSel->num_atoms,
                                firstsel, lastsel, fmin, fmax);
#else
    for (i++; i<=lastsel; i++) {
      if (on[i]) {
        long ind = i * 3L;
        float tmpx = framepos[ind  ];
        if (tmpx < minx) minx = tmpx; 
        if (tmpx > maxx) maxx = tmpx;
  
        float tmpy = framepos[ind+1];
        if (tmpy < miny) miny = tmpy; 
        if (tmpy > maxy) maxy = tmpy;
  
        float tmpz = framepos[ind+2];
        if (tmpz < minz) minz = tmpz; 
        if (tmpz > maxz) maxz = tmpz;
      }
    }
#endif
  } else {
    // calculate bounding box for atoms /w given radii 
    for (i++; i<=lastsel; i++) {
      if (on[i]) {
        long ind = i * 3L;
        float mintmpx = framepos[ind  ] - radii[i];
        float maxtmpx = framepos[ind  ] + radii[i];
        if (mintmpx < minx) minx = mintmpx;
        if (maxtmpx > maxx) maxx = maxtmpx;
  
        float mintmpy = framepos[ind+1] - radii[i];
        float maxtmpy = framepos[ind+1] + radii[i];
        if (mintmpy < miny) miny = mintmpy; 
        if (maxtmpy > maxy) maxy = maxtmpy;
  
        float mintmpz = framepos[ind+2] - radii[i];
        float maxtmpz = framepos[ind+2] + radii[i];
        if (mintmpz < minz) minz = mintmpz; 
        if (maxtmpz > maxz) maxz = maxtmpz;
      }
    }
  }

  // set the final min/max output values
  min_coord[0]=minx;
  min_coord[1]=miny;
  min_coord[2]=minz;
  max_coord[0]=maxx;
  max_coord[1]=maxy;
  max_coord[2]=maxz;

  return MEASURE_NOERR;
}

/*I'm going to assume that *avpos points to an array the length of 3*sel. The return
value will indicate the ACTUAL number of residues in the selection,
which tells the caller how many values should be in the returned list, or a number
less than zero on error.
*/
int measure_avpos_perresidue(const AtomSel *sel, MoleculeList *mlist, 
                         int start, int end, int step, float *avpos)  {
  //Get the per-atom average position. We'll be accumulating on this array.
  int retval = measure_avpos(sel, mlist, start, end, step, avpos);
  if (retval != MEASURE_NOERR) {
    return retval;
  }
  Molecule *mol = mlist->mol_from_id(sel->molid());
  int residue = mol->atom(sel->firstsel)->uniq_resid;
  int rescount = 0;
  int ressize = 0;
  float accumulate[3] = {0.0,0.0,0.0};
  int j = 0, k, i;
  for (i = sel->firstsel; i <= sel->lastsel; i++) {
    if (sel->on[i]) {
      if (residue != mol->atom(i)->uniq_resid) {
        for (k = 0; k < 3; k++) {
          avpos[3*rescount + k] = accumulate[k] / ressize;
          accumulate[k] = 0.0;
        }
        residue = mol->atom(i)->uniq_resid;
        ressize = 0;
        rescount++;
      }
      for (k = 0; k < 3; k++) {
        accumulate[k] += avpos[3*j + k];
      }
      j++;
      ressize++;
    }
  }
  for (k = 0; k < 3; k++) {
    avpos[3*rescount + k] = accumulate[k] / ressize;
  }
  rescount++;
  return rescount;
}

// Calculate average position of selected atoms over selected frames
extern int measure_avpos(const AtomSel *sel, MoleculeList *mlist, 
                         int start, int end, int step, float *avpos) {
  if (!sel)                     return MEASURE_ERR_NOSEL;
  if (sel->num_atoms == 0)      return MEASURE_ERR_NOATOMS;

  Molecule *mymol = mlist->mol_from_id(sel->molid());
  int maxframes = mymol->numframes();
  
  // accept value of -1 meaning "all" frames
  if (end == -1)
    end = maxframes-1;

  if (maxframes == 0 || start < 0 || start > end || 
      end >= maxframes || step <= 0)
    return MEASURE_ERR_BADFRAMERANGE;

  long i;
  for (i=0; i<(3L*sel->selected); i++)
    avpos[i] = 0.0f;

  long frame, avcount, j;
  for (avcount=0,frame=start; frame<=end; avcount++,frame+=step) {
    const float *framepos = (mymol->get_frame(frame))->pos;
    for (j=0,i=sel->firstsel; i<=sel->lastsel; i++) {
      if (sel->on[i]) {
        avpos[j*3L    ] += framepos[i*3L    ];
        avpos[j*3L + 1] += framepos[i*3L + 1];
        avpos[j*3L + 2] += framepos[i*3L + 2];
        j++;
      }
    } 
  }

  float avinv = 1.0f / (float) avcount;
  for (j=0; j<(3L*sel->selected); j++) {
    avpos[j] *= avinv;
  } 

  return MEASURE_NOERR;
}


// Calculate dipole moment for selected atoms
extern int measure_dipole(const AtomSel *sel, MoleculeList *mlist, 
                          float *dipole, int unitsdebye, int usecenter) {
  if (!sel)                     return MEASURE_ERR_NOSEL;
  if (sel->num_atoms == 0)      return MEASURE_ERR_NOATOMS;

  Molecule *mymol = mlist->mol_from_id(sel->molid());
  double  rvec[3] = {0, 0, 0};
  double qrvec[3] = {0, 0, 0};
  double mrvec[3] = {0, 0, 0};
  double totalq = 0.0;
  double totalm = 0.0;
  int i;

  // get atom coordinates
  const float *framepos = sel->coordinates(mlist);

  // get atom charges
  const float *q = mymol->charge();
  const float *m = mymol->mass();

  for (i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      int ind = i * 3;
      rvec[0] += framepos[ind    ];
      rvec[1] += framepos[ind + 1];
      rvec[2] += framepos[ind + 2];

      qrvec[0] += framepos[ind    ] * q[i];
      qrvec[1] += framepos[ind + 1] * q[i];
      qrvec[2] += framepos[ind + 2] * q[i];

      mrvec[0] += framepos[ind    ] * m[i];
      mrvec[1] += framepos[ind + 1] * m[i];
      mrvec[2] += framepos[ind + 2] * m[i];

      totalq += q[i];
      totalm += m[i];
    }
  }

  // fall back to geometrical center when bad or no masses
  if (totalm < 0.0001)
    usecenter=1;
  
  switch (usecenter) {
    case 1:
    {
        double rscale = totalq / sel->selected; 
        dipole[0] = (float) (qrvec[0] - (rvec[0] * rscale)); 
        dipole[1] = (float) (qrvec[1] - (rvec[1] * rscale)); 
        dipole[2] = (float) (qrvec[2] - (rvec[2] * rscale)); 
        break;
    }

    case -1: 
    {
        double mscale = totalq / totalm; 
        dipole[0] = (float) (qrvec[0] - (mrvec[0] * mscale)); 
        dipole[1] = (float) (qrvec[1] - (mrvec[1] * mscale)); 
        dipole[2] = (float) (qrvec[2] - (mrvec[2] * mscale)); 
        break;
    }
    
    case 0: // fallthrough
    default: 
    {
        dipole[0] = (float) qrvec[0];
        dipole[1] = (float) qrvec[1];
        dipole[2] = (float) qrvec[2];
        break;
    }
  }

  // According to the values in
  // http://www.physics.nist.gov/cuu/Constants/index.html
  // 1 e*A = 4.80320440079 D
  // 1 D = 1E-18 Fr*cm = 0.208194346224 e*A
  // 1 e*A = 1.60217653 E-29 C*m
  // 1 C*m = 6.24150947961 E+28 e*A
  // 1 e*A = 1.88972613458 e*a0
  // 1 e*a0 = 0.529177208115 e*A

  if (unitsdebye) {
    // 1 e*A = 4.80320440079 D 
    // latest CODATA (2006) gives:
    //         4.80320425132073913031
    dipole[0] *= 4.80320425132f;
    dipole[1] *= 4.80320425132f;
    dipole[2] *= 4.80320425132f;
  }
 
  return MEASURE_NOERR;
}


extern int measure_hbonds(Molecule *mol, AtomSel *sel1, AtomSel *sel2, double cutoff, double maxangle, int *donlist, int *hydlist, int *acclist, int maxsize) {
  int hbondcount = 0;
  const float *pos = sel1->coordinates(mol->app->moleculeList);
  const int *A = sel1->on;
  const int *B = sel2 ? sel2->on : sel1->on;
  GridSearchPair *pairlist = vmd_gridsearch2(pos, sel1->num_atoms, A, B, (float) cutoff, sel1->num_atoms * 27);
  GridSearchPair *p, *tmp;
  float donortoH[3], Htoacceptor[3];
  
  for (p=pairlist; p != NULL; p=tmp) {
    MolAtom *a1 = mol->atom(p->ind1); 
    MolAtom *a2 = mol->atom(p->ind2); 
    
    // neither the donor nor acceptor may be hydrogens
    if (mol->atom(p->ind1)->atomType == ATOMHYDROGEN ||
        mol->atom(p->ind2)->atomType == ATOMHYDROGEN) {
      tmp = p->next;
      free(p);
      continue;
    } 
    if (!a1->bonded(p->ind2)) {
      int b1 = a1->bonds;
      int b2 = a2->bonds;
      const float *coor1 = pos + 3*p->ind1;
      const float *coor2 = pos + 3*p->ind2;
      int k;
      // first treat sel1 as donor
      for (k=0; k<b1; k++) {
        const int hindex = a1->bondTo[k];
        if (mol->atom(hindex)->atomType == ATOMHYDROGEN) {         
          const float *hydrogen = pos + 3*hindex;
          vec_sub(donortoH,hydrogen,coor1);
          vec_sub(Htoacceptor,coor2,hydrogen);
          if (angle(donortoH, Htoacceptor)  < maxangle ) {
            if (hbondcount < maxsize) {
              donlist[hbondcount] = p->ind1;
              acclist[hbondcount] = p->ind2;
              hydlist[hbondcount] = hindex;
            }
            hbondcount++;
          }
        }
      }
      // if only one atom selection was given, treat sel2 as a donor as well
      if (!sel2) {
        for (k=0; k<b2; k++) {
          const int hindex = a2->bondTo[k];
          if (mol->atom(hindex)->atomType == ATOMHYDROGEN) {
            const float *hydrogen = pos + 3*hindex;
            vec_sub(donortoH,hydrogen,coor2);
            vec_sub(Htoacceptor,coor1,hydrogen);
            if (angle(donortoH, Htoacceptor)  < maxangle ) {
              if (hbondcount < maxsize) {
                donlist[hbondcount] = p->ind2;
                acclist[hbondcount] = p->ind1;
                hydlist[hbondcount] = hindex;
              }
              hbondcount++;
            }
          }
        }
      } 
    }
    tmp = p->next;
    free(p);
  }
  return hbondcount;
}

int measure_rmsf_perresidue(const AtomSel *sel, MoleculeList *mlist, 
                        int start, int end, int step, float *rmsf) {
  if (!sel)                     return MEASURE_ERR_NOSEL;
  if (sel->num_atoms == 0)      return MEASURE_ERR_NOATOMS;

  Molecule *mymol = mlist->mol_from_id(sel->molid());
  int maxframes = mymol->numframes();

  // accept value of -1 meaning "all" frames
  if (end == -1)
    end = maxframes-1;

  if (maxframes == 0 || start < 0 || start > end ||
      end >= maxframes || step <= 0)
    return MEASURE_ERR_BADFRAMERANGE;

  int i;
  for (i=0; i<sel->selected; i++)
    rmsf[i] = 0.0f;

  int rc; 
  float *avpos = new float[3*sel->selected];
  //rc will be the number of residues.
  rc = measure_avpos_perresidue(sel, mlist, start, end, step, avpos);
  if (rc <= 0) {
    delete [] avpos;
    return rc;
  }
  
  int frame, avcount;
  float *center = new float[3*rc];
  float *weights = new float[sel->selected];
  for (i = 0; i < sel->selected; i++) {
    weights[i] = 1.0;
  }
  // calculate per-residue variance here. Its a simple calculation, since we have measure_center_perresidue.
  for (avcount=0,frame=start; frame<=end; avcount++,frame+=step) {
    const float *framepos = (mymol->get_frame(frame))->pos;
    measure_center_perresidue(mlist, sel, framepos, weights, center);
    for (i = 0; i < rc; i++) {
      rmsf[i] += distance2(&avpos[3*i], &center[3*i]);
    }
  }
  delete [] center;
  delete [] weights;

  float avinv = 1.0f / (float) avcount;
  for (i=0; i<rc; i++) {
    rmsf[i] = sqrtf(rmsf[i] * avinv);
  }

  delete [] avpos;
  return rc;
}

// Calculate RMS fluctuation of selected atoms over selected frames
extern int measure_rmsf(const AtomSel *sel, MoleculeList *mlist, 
                        int start, int end, int step, float *rmsf) {
  if (!sel)                     return MEASURE_ERR_NOSEL;
  if (sel->num_atoms == 0)      return MEASURE_ERR_NOATOMS;

  Molecule *mymol = mlist->mol_from_id(sel->molid());
  int maxframes = mymol->numframes();

  // accept value of -1 meaning "all" frames
  if (end == -1)
    end = maxframes-1;

  if (maxframes == 0 || start < 0 || start > end ||
      end >= maxframes || step <= 0)
    return MEASURE_ERR_BADFRAMERANGE;

  int i;
  for (i=0; i<sel->selected; i++)
    rmsf[i] = 0.0f;

  int rc; 
  float *avpos = new float[3L*sel->selected];
  rc = measure_avpos(sel, mlist, start, end, step, avpos);

  if (rc != MEASURE_NOERR) {
    delete [] avpos;
    return rc;
  }

  // calculate per-atom variance here
  int frame, avcount, j;
  for (avcount=0,frame=start; frame<=end; avcount++,frame+=step) {
    const float *framepos = (mymol->get_frame(frame))->pos;
    for (j=0,i=sel->firstsel; i<=sel->lastsel; i++) {
      if (sel->on[i]) {
        rmsf[j] += distance2(&avpos[3L*j], &framepos[3L*i]);
        j++;
      }
    }
  }

  float avinv = 1.0f / (float) avcount;
  for (j=0; j<sel->selected; j++) {
    rmsf[j] = sqrtf(rmsf[j] * avinv);
  }

  delete [] avpos;
  return MEASURE_NOERR;
}


/// measure the radius of gyration, including the given weights
//  rgyr := sqrt(sum (mass(n) ( r(n) - r(com) )^2)/sum(mass(n)))
//  The return value, a float, is put in 'float *rgyr'
//  The function return value is 0 if ok, <0 if not
int measure_rgyr(const AtomSel *sel, MoleculeList *mlist, const float *weight, 
                 float *rgyr) {
  if (!rgyr) return MEASURE_ERR_NORGYR;
  if (!sel) return MEASURE_ERR_NOSEL;
  if (!mlist) return MEASURE_ERR_GENERAL;

  const float *framepos = sel->coordinates(mlist);

  // compute the center of mass with the current weights
  float com[3];
  int ret_val = measure_center(sel, framepos, weight, com);
  if (ret_val < 0) 
    return ret_val;
  
  // measure center of gyration
  int i, j;
  float total_w, sum;

  total_w=sum=0;
  for (j=0,i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      float w = weight[j];
      total_w += w;
      sum += w * distance2(framepos + 3L*i, com);
      j++;
    } 
  }

  if (total_w == 0) {
    return MEASURE_ERR_BADWEIGHTSUM;
  }

  // and finalize the computation
  *rgyr = sqrtf(sum/total_w);

  return MEASURE_NOERR;
}

/*I'm going to assume that *rmsd points to an array the length of sel1. The return
value will indicate the ACTUAL number of residues in the selection (specifically sel1),
which tells the caller how many values should be in the returned list, or a number
less than zero on error.
*/
int measure_rmsd_perresidue(const AtomSel *sel1, const AtomSel *sel2, MoleculeList *mlist,
                 int num,
                 float *weight, float *rmsd) {
  if (!sel1 || !sel2)   return MEASURE_ERR_NOSEL;
  if (sel1->selected < 1 || sel2->selected < 1) return MEASURE_ERR_NOSEL;
  if (!weight || !rmsd) return MEASURE_ERR_NOWEIGHT;

  // the number of selected atoms must be the same
  if (sel1->selected != sel2->selected) return MEASURE_ERR_MISMATCHEDCNT;

  // need to know how to traverse the list of weights
  // there could be 1 weight per atom (sel_flg == 1) or 
  // 1 weight per selected atom (sel_flg == 0)
  int sel_flg;
  if (num == sel1->num_atoms) {
    sel_flg = 1; // using all elements
  } else {
    sel_flg = 0; // using elements from selection
  }

  // temporary variables
  float tmp_w;
  int w_index = 0;              // the term in the weight field to use
  int sel1ind = sel1->firstsel; // start from the first selected atom
  int sel2ind = sel2->firstsel; // start from the first selected atom
  float wsum = 0;
  float twsum = 0;
  float rmsdsum = 0;
  const float *framepos1 = sel1->coordinates(mlist);
  const float *framepos2 = sel2->coordinates(mlist);
  Molecule *mol = mlist->mol_from_id(sel1->molid());


  int count = sel1->selected;
  int residue = mol->atom(sel1ind)->uniq_resid;
  int rescount = 0;
  while (count-- > 0) {
    // find next 'on' atom in sel1 and sel2
    // loop is safe since we already stop the on count > 0 above
    while (!sel1->on[sel1ind])
      sel1ind++;
    while (!sel2->on[sel2ind])
      sel2ind++;
    if (residue != mol->atom(sel1ind)->uniq_resid) {
      rmsd[rescount] = sqrtf(rmsdsum / wsum);
      rmsdsum = 0;
      twsum += wsum;
      wsum = 0;
      residue = mol->atom(sel1ind)->uniq_resid;
      rescount++;
    }
    // the weight offset to use depends on how many terms there are
    if (sel_flg == 0) {
      tmp_w = weight[w_index++];
    } else {
      tmp_w = weight[sel1ind]; // use the first selection for the weights
    }

    // sum the calculated rmsd and weight values
    rmsdsum += tmp_w * distance2(framepos1 + 3*sel1ind, framepos2 + 3*sel2ind);
    wsum += tmp_w;

    // and advance to the next atom pair
    sel1ind++;
    sel2ind++;
  }
  twsum += wsum;
  // check weight sum
  if (twsum == 0) {
    return MEASURE_ERR_BADWEIGHTSUM;
  }

  // finish the rmsd calcs
  rmsd[rescount++] = sqrtf(rmsdsum / wsum);

  return rescount; // and say rmsd is OK
}

/// measure the rmsd given a selection and weight term
//  1) if num == sel.selected ; assumes there is one weight per 
//           selected atom
//  2) if num == sel.num_atoms; assumes weight[i] is for atom[i]
//  returns 0 and value in rmsd if good
//   return < 0 if invalid
//  Function is::=  rmsd = 
//    sqrt(sum(weight(n) * sqr(r1(i(n))-r2(i(n))))/sum(weight(n)) / N
int measure_rmsd(const AtomSel *sel1, const AtomSel *sel2,
                 int num, const float *framepos1, const float *framepos2,
                 float *weight, float *rmsd) {
  if (!sel1 || !sel2)   return MEASURE_ERR_NOSEL;
  if (sel1->selected < 1 || sel2->selected < 1) return MEASURE_ERR_NOSEL;
  if (!weight || !rmsd) return MEASURE_ERR_NOWEIGHT;

  // the number of selected atoms must be the same
  if (sel1->selected != sel2->selected) return MEASURE_ERR_MISMATCHEDCNT;

  // need to know how to traverse the list of weights
  // there could be 1 weight per atom (sel_flg == 1) or 
  // 1 weight per selected atom (sel_flg == 0)
  int sel_flg;
  if (num == sel1->num_atoms) {
    sel_flg = 1; // using all elements
  } else {
    sel_flg = 0; // using elements from selection
  }

  // temporary variables
  float tmp_w;
  int w_index = 0;              // the term in the weight field to use
  int sel1ind = sel1->firstsel; // start from the first selected atom
  int sel2ind = sel2->firstsel; // start from the first selected atom
  float wsum = 0;
  float rmsdsum = 0;

  *rmsd = 10000000; // if we bail out, return a huge value

  // compute the rmsd
  int count = sel1->selected;
  while (count-- > 0) {
    // find next 'on' atom in sel1 and sel2
    // loop is safe since we already stop the on count > 0 above
    while (!sel1->on[sel1ind])
      sel1ind++;
    while (!sel2->on[sel2ind])
      sel2ind++;

    // the weight offset to use depends on how many terms there are
    if (sel_flg == 0) {
      tmp_w = weight[w_index++];
    } else {
      tmp_w = weight[sel1ind]; // use the first selection for the weights
    }

    // sum the calculated rmsd and weight values
    rmsdsum += tmp_w * distance2(framepos1 + 3L*sel1ind, framepos2 + 3L*sel2ind);
    wsum += tmp_w;

    // and advance to the next atom pair
    sel1ind++;
    sel2ind++;
  }

  // check weight sum
  if (wsum == 0) {
    return MEASURE_ERR_BADWEIGHTSUM;
  }

  // finish the rmsd calcs
  *rmsd = sqrtf(rmsdsum / wsum);

  return MEASURE_NOERR; // and say rmsd is OK
}

/* jacobi.C, taken from Numerical Recipes and specialized to 3x3 case */

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);

static int jacobi(float a[4][4], float d[3], float v[3][3])
{
  int n=3;
  int j,iq,ip,i;
  float tresh,theta,tau,t,sm,s,h,g,c,*b,*z;
  
  b=new float[n];
  z=new float[n];
  for (ip=0;ip<n;ip++) {
    for (iq=0;iq<n;iq++) v[ip][iq]=0.0;
    v[ip][ip]=1.0;
  }
  for (ip=0;ip<n;ip++) {
    b[ip]=d[ip]=a[ip][ip];
    z[ip]=0.0;
  }
  for (i=1;i<=50;i++) {
    sm=0.0;
    for (ip=0;ip<n-1;ip++) {
      for (iq=ip+1;iq<n;iq++)
	sm += (float) fabs(a[ip][iq]);
    }
    if (sm == 0.0) {
      delete [] z;
      delete [] b;
      return 0; // Exit normally
    }
    if (i < 4)
      tresh=0.2f*sm/(n*n);
    else
      tresh=0.0f;
    for (ip=0;ip<n-1;ip++) {
      for (iq=ip+1;iq<n;iq++) {
	g=100.0f * fabsf(a[ip][iq]);
	if (i > 4 && (float)(fabs(d[ip])+g) == (float)fabs(d[ip])
	    && (float)(fabs(d[iq])+g) == (float)fabs(d[iq]))
	  a[ip][iq]=0.0;
	else if (fabs(a[ip][iq]) > tresh) {
	  h=d[iq]-d[ip];
	  if ((float)(fabs(h)+g) == (float)fabs(h))
	    t=(a[ip][iq])/h;
	  else {
	    theta=0.5f*h/(a[ip][iq]);
	    t=1.0f/(fabsf(theta)+sqrtf(1.0f+theta*theta));
	    if (theta < 0.0f) t = -t;
	  }
	  c=1.0f/sqrtf(1+t*t);
	  s=t*c;
	  tau=s/(1.0f+c);
	  h=t*a[ip][iq];
	  z[ip] -= h;
	  z[iq] += h;
	  d[ip] -= h;
	  d[iq] += h;
	  a[ip][iq]=0.0;
	  for (j=0;j<=ip-1;j++) {
	    ROTATE(a,j,ip,j,iq)
	      }
	  for (j=ip+1;j<=iq-1;j++) {
	    ROTATE(a,ip,j,j,iq)
	      }
	  for (j=iq+1;j<n;j++) {
	    ROTATE(a,ip,j,iq,j)
	      }
	  for (j=0;j<n;j++) {
	    ROTATE(v,j,ip,j,iq)
	      }
	}
      }
    }
    for (ip=0;ip<n;ip++) {
      b[ip] += z[ip];
      d[ip]=b[ip];
      z[ip]=0.0;
    }
  }
  delete [] b;
  delete [] z;
  return 1; // Failed to converge
}
#undef ROTATE

static int transvecinv(const double *v, Matrix4 &res) {
  double x, y, z;
  x=v[0];
  y=v[1];
  z=v[2];
  if (x == 0.0 && y == 0.0) {
    if (z == 0.0) {
      return -1;
    }
    if (z > 0)
      res.rot(90, 'y');
    else
      res.rot(-90, 'y');
    return 0;
  }
  double theta = atan2(y,x);
  double length = sqrt(x*x + y*y);
  double phi = atan2(z,length);
  res.rot((float) RADTODEG(phi),  'y');
  res.rot((float) RADTODEG(-theta), 'z');
  return 0;
}
 
static int transvec(const double *v, Matrix4 &res) {
  double x, y, z;
  x=v[0];
  y=v[1];
  z=v[2];
  if (x == 0.0 && y == 0.0) {
    if (z == 0.0) {
      return -1;
    }
    if (z > 0)
      res.rot(-90, 'y');
    else
      res.rot(90, 'y');
    return 0;
  }
  double theta = atan2(y,x);
  double length = sqrt(x*x + y*y);
  double phi = atan2(z,length);
  res.rot((float) RADTODEG(theta), 'z');
  res.rot((float) RADTODEG(-phi),  'y');
  return 0;
}

static Matrix4 myfit2(const float *x, const float *y, 
               const float *comx, const float *comy) {

  Matrix4 res;
  double dx[3], dy[3];
  dx[0] = x[0] - comx[0];
  dx[1] = x[1] - comx[1];
  dx[2] = x[2] - comx[2];
  dy[0] = y[0] - comy[0];
  dy[1] = y[1] - comy[1];
  dy[2] = y[2] - comy[2];
  
  res.translate(comy[0], comy[1], comy[2]);
  transvec(dy, res);
  transvecinv(dx, res);
  res.translate(-comx[0], -comx[1], -comx[2]);
  return res;
}

static Matrix4 myfit3(const float *x1, const float *x2, 
                      const float *y1, const float *y2,
                      const float *comx, const float *comy) {
   
  Matrix4 mx, my, rx1, ry1;
  double dx1[3], dy1[3], angle;
  float dx2[3], dy2[3], x2t[3], y2t[3];

  for (int i=0; i<3; i++) {
    dx1[i] = x1[i] - comx[i];
    dx2[i] = x2[i] - comx[i];
    dy1[i] = y1[i] - comy[i];
    dy2[i] = y2[i] - comy[i];
  }

  // At some point, multmatrix went from pre-multiplying, as the code of 
  // Matrix.C itself suggests, to post multiplying, which is what the 
  // users must have expected. Thus my.multmatrix(mx) is the same as 
  // my = my * mx, not mx * my.  This means that you use the matrix 
  // conventions of openGL (first matrix in the code is the last 
  // matrix applied)
  transvecinv(dx1, rx1);
  rx1.multpoint3d(dx2, x2t);
  angle = atan2(x2t[2], x2t[1]);
  mx.rot((float) RADTODEG(angle), 'x'); 
  mx.multmatrix(rx1);
  mx.translate(-comx[0], -comx[1], -comx[2]);
  
  transvecinv(dy1, ry1);
  ry1.multpoint3d(dy2, y2t);
  angle = atan2(y2t[2], y2t[1]);
  my.rot((float) RADTODEG(angle), 'x');
  my.multmatrix(ry1);
  my.translate(-comy[0], -comy[1], -comy[2]);
  my.inverse();
  my.multmatrix(mx);
  return my;
}

// find the best fit alignment to take the first structure into the second
// Put the result in the matrix 'mat'
//  This algorithm comes from Kabsch, Acta Cryst. (1978) A34, 827-828.
// Need the 2nd weight for the com calculation
int measure_fit(const AtomSel *sel1, const AtomSel *sel2, const float *x, 
                const float *y, const float *weight, 
                const int *order, Matrix4 *mat) {
  float comx[3];
  float comy[3];
  int num = sel1->selected;
  int ret_val;
  ret_val = measure_center(sel1, x, weight, comx);
  if (ret_val < 0) {
    return ret_val;
  }
  ret_val = measure_center(sel2, y, weight, comy);
  if (ret_val < 0) {
    return ret_val;
  }

  // the Kabsch method won't work of the number of atoms is less than 4
  // (and won't work in some cases of n > 4; I think it works so long as
  // three or more planes are needed to intersect all the data points
  switch (sel1->selected) {
  case 1: { // simple center of mass alignment
    Matrix4 tmp;
    tmp.translate(-comx[0], -comx[1], -comx[2]);
    tmp.translate(comy[0], comy[1], comy[2]);
    memcpy(mat->mat, tmp.mat, 16L*sizeof(float));
    return MEASURE_NOERR;
  }
  case 3:
  case 2: { 
    // find the first (n-1) points (from each molecule)
    int pts[6], count = 0;
    int n;
    for (n=sel1->firstsel; n<=sel1->lastsel; n++) {
      if (sel1->on[n]) {
	pts[count++] = n;
	if (sel1->selected == 2) {
	  count++;                   // will put y data in pts[3]
	  break;
	}
	if (count == 2) break;
      }
    }
    for (n=sel2->firstsel; n<=sel2->lastsel; n++) {
      if (sel2->on[n]) {
        pts[count++] = n;
        if (sel1->selected == 2) {
          count++;
          break;
        }
        if (count == 4) break;
      }
    }
    if (count != 4) {
      return MEASURE_ERR_MISMATCHEDCNT;
    }
    
    // reorder the sel2 atoms according to the order parameter
    if (order != NULL) {
      int i; 
      int tmp[6];
      memcpy(tmp, pts, sizeof(pts));
      for (i=0; i<num; i++) {
        pts[i + num] = tmp[num + order[i]]; // order indices are 0-based
      } 
    }

    if (sel1->selected == 2) {
      *mat = myfit2(x+3L*pts[0], y+3L*pts[2], comx, comy);
      ret_val = 0;
    } else {
      *mat = myfit3(x+3L*pts[0], x+3L*pts[1], y+3L*pts[2], y+3L*pts[3], comx, comy);
      ret_val = 0;
    }  
    if (ret_val != 0) {
      return MEASURE_ERR_GENERAL;
    }

    return 0;
  }
  default:
    break;
  }
  // at this point I know all the data values are good
 

  // use the new RMS fit implementation by default unless told otherwise 
  char *opt = getenv("VMDRMSFITMETHOD");
  if (!opt || strcmp(opt, "oldvmd")) {
    int i, k;
    float *v1, *v2;
    v1 = new float[3L*num];
    v2 = new float[3L*num];
    for (k=0, i=sel1->firstsel; i<=sel1->lastsel; i++) {
      if (sel1->on[i]) {
        long ind = 3L * i;
        v1[k++] = x[ind    ];
        v1[k++] = x[ind + 1];
        v1[k++] = x[ind + 2];
      }
    }
    for (k=0, i=sel2->firstsel; i<=sel2->lastsel; i++) {
      if (sel2->on[i]) {
        long ind = 3L * i;
        v2[k++] = y[ind    ];
        v2[k++] = y[ind + 1];
        v2[k++] = y[ind + 2];
      }
    }

    // reorder the sel2 atoms according to the order parameter
    if (order != NULL) {
      int i; 
      float *tmp = new float[3L*num];
      memcpy(tmp, v2, 3L*num*sizeof(float));
      for (i=0; i<num; i++) {
        long ind = 3L * i;
        long idx = 3L * order[i]; // order indices are 0-based
        v2[ind    ] = tmp[idx    ];
        v2[ind + 1] = tmp[idx + 1];
        v2[ind + 2] = tmp[idx + 2];
      } 
      delete[] tmp;
    }

    float tmp[16];
    // MatrixFitRMS returns RMS distance of fitted molecule.  Would be nice
    // to return this information to the user since it would make computing
    // the fitted RMSD much faster (no need to get the matrix, apply the
    // transformation, recompute RMS).
    MatrixFitRMS(num, v1, v2, weight, tmp);

    delete [] v1;
    delete [] v2;
    //fprintf(stderr, "got err %f\n", err);
    // output from FitRMS is a 3x3 matrix, plus a pre-translation stored in
    // row 3, and a post-translation stored in column 3.
    float pre[3], post[3];
    for (i=0; i<3; i++) {
      post[i] = tmp[4L*i+3];
      tmp[4L*i+3] = 0;
    }
    for (i=0; i<3; i++) {
      pre[i] = tmp[12+i];
      tmp[12+i] = 0;
    }
    Matrix4 result;
    result.translate(pre);
    result.multmatrix(Matrix4(tmp));
    result.translate(post);
    memcpy(mat->mat, result.mat, 16L*sizeof(float));
    return 0;
  }

  // the old RMS fit code doesn't support reordering of sel2 currently
  if (order != NULL) {
    return MEASURE_ERR_NOTSUP;
  }

  // a) compute R = r(i,j) = sum( w(n) * (y(n,i)-comy(i)) * (x(n,j)-comx(j)))
  Matrix4 R;
  int i,j;
  float scale = (float) num * num;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      float tmp = 0;
      int nx = 0, ny = 0, k = 0; 
      while (nx < sel1->num_atoms && ny < sel2->num_atoms) { 
        if (!sel1->on[nx]) {
          nx++;
          continue;
        }
        if (!sel2->on[ny]) {
          ny++;
          continue;
        }

	// found both, so get data
        
	tmp += weight[k] * (y[3L*ny+i] - comy[i]) * (x[3L*nx+j] - comx[j]) /
	  scale;
	nx++;
	ny++;
        k++;
      }
      R.mat[4L*i+j] = tmp;
    }
  }

  // b) 1) form R~R
  Matrix4 Rt;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      Rt.mat[4L*i+j] = R.mat[4L*j+i];
    }
  }
  Matrix4 RtR(R);
  RtR.multmatrix(Rt);

  // b) 2) find the eigenvalues and eigenvectors
  float evalue[3];
  float evector[3][3];
  float tmpmat[4][4];
  for (i=0; i<4; i++)
    for (j=0; j<4; j++)
      tmpmat[i][j]=RtR.mat[4L*i+j];

  if(jacobi(tmpmat,evalue,evector) != 0) return MEASURE_ERR_NONZEROJACOBI;
  // transposition the evector matrix to put the vectors in rows
  float vectmp;
  vectmp=evector[0][1]; evector[0][1]=evector[1][0]; evector[1][0]=vectmp;
  vectmp=evector[0][2]; evector[0][2]=evector[2][0]; evector[2][0]=vectmp;
  vectmp=evector[2][1]; evector[2][1]=evector[1][2]; evector[1][2]=vectmp;


  // b) 4) sort so that the eigenvalues are from largest to smallest
  //      (or rather so a[0] is eigenvector with largest eigenvalue, ...)
  float *a[3];
  a[0] = evector[0];
  a[1] = evector[1];
  a[2] = evector[2];
#define SWAP(qq,ww) {                                           \
    float v; float *v1;                                         \
    v = evalue[qq]; evalue[qq] = evalue[ww]; evalue[ww] = v;    \
    v1 = a[qq]; a[qq] = a[ww]; a[ww] = v1;                      \
}
  if (evalue[0] < evalue[1]) {
    SWAP(0, 1);
  }
  if (evalue[0] < evalue[2]) {
    SWAP(0, 2);
  }
  if (evalue[1] < evalue[2]) {
    SWAP(1, 2);
  }

  
  // c) determine b(i) = R*a(i)
  float b[3][3];

  Rt.multpoint3d(a[0], b[0]);
  vec_normalize(b[0]);

  Rt.multpoint3d(a[1], b[1]);
  vec_normalize(b[1]);

  Rt.multpoint3d(a[2], b[2]);
  vec_normalize(b[2]);

  // d) compute U = u(i,j) = sum(b(k,i) * a(k,j))
  Matrix4 U;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      float *tmp = &(U.mat[4L*j+i]);
      *tmp = 0;
      for (int k=0; k<3; k++) {
	*tmp += b[k][i] * a[k][j];
      }
    }
  }

  // Check the determinant of U.  If it's negative, we need to
  // flip the sign of the last row.
  float *um = U.mat;
  float det = 
    um[0]*(um[4+1]*um[8+2] - um[4+2]*um[8+1]) -
    um[1]*(um[4+0]*um[8+2] - um[4+2]*um[8+0]) +
    um[2]*(um[4+0]*um[8+1] - um[4+1]*um[8+0]);
  if (det < 0) {
    for (int q=0; q<3; q++) um[8+q] = -um[8+q];
  }

  // e) apply the offset for com
  Matrix4 tx;
  tx.translate(-comx[0], -comx[1], -comx[2]);
  Matrix4 ty;
  ty.translate(comy[0], comy[1], comy[2]);
  //  U.multmatrix(com);
  ty.multmatrix(U);
  ty.multmatrix(tx);
  memcpy(mat->mat, ty.mat, 16L*sizeof(float));
  return MEASURE_NOERR;
}

// For different values of the random seed, the computed SASA's of brH.pdb 
// converge to within 1% of each other when the number of points is about
// 500.  We therefore use 500 as the default number.
#define NPTS 500 
extern int measure_sasa(const AtomSel *sel, const float *framepos,
    const float *radius, float srad, float *sasa, 
    ResizeArray<float> *sasapts, const AtomSel *restrictsel,
    const int *nsamples) {

  // check arguments
  if (!sel) return MEASURE_ERR_NOSEL;
  if (!sel->selected) {
    *sasa = 0;
    return MEASURE_NOERR;
  }
  if (!framepos) return MEASURE_ERR_NOFRAMEPOS;
  if (!radius)   return MEASURE_ERR_NORADII;
  if (restrictsel && restrictsel->num_atoms != sel->num_atoms)
    return MEASURE_ERR_MISMATCHEDCNT;

  int i;
  int npts = nsamples ? *nsamples : NPTS;
  float maxrad = -1;

#if 0
  // Query min/max atom radii for the entire molecule
  mol->get_radii_minmax(minrad, maxrad);
#endif

  // find biggest atom radius 
  for (i=0; i<sel->num_atoms; i++) {
    float rad = radius[i];
    if (maxrad < rad) maxrad = rad;
  }

  // Find atoms within maxrad of each other.  
  // build a list of pairs for each atom
  ResizeArray<int> *pairlist = new ResizeArray<int>[sel->num_atoms];
  {
    GridSearchPair *pairs;
    pairs = vmd_gridsearch1(framepos, sel->num_atoms, sel->on, 
                            2.0f * (maxrad + srad), 0, sel->num_atoms * 1000);

    GridSearchPair *p, *tmp; 
    for (p = pairs; p != NULL; p = tmp) {
      int ind1=p->ind1;
      int ind2=p->ind2;
      pairlist[ind1].append(ind2);
      pairlist[ind2].append(ind1);
      tmp = p->next;
      free(p);
    }
  }

  static const float RAND_MAX_INV = 1.0f/VMD_RAND_MAX;
  // Seed the random number generator before each calculation.  This gives
  // reproducible results and still allows a more accurate answer to be
  // obtained by increasing the samples size.  I don't know if this is a
  // "good" seed value or not, I just picked something random-looking.
  vmd_srandom(38572111);

  // All the spheres use the same random points.  
  float *spherepts = new float[3L*npts];
  for (i=0; i<npts; i++) {
    float u1 = (float) vmd_random();
    float u2 = (float) vmd_random();
    float z = 2.0f*u1*RAND_MAX_INV -1.0f;
    float phi = (float) (2.0f*VMD_PI*u2*RAND_MAX_INV);
    float R = sqrtf(1.0f-z*z);
    spherepts[3L*i  ] = R*cosf(phi);
    spherepts[3L*i+1] = R*sinf(phi);
    spherepts[3L*i+2] = z;
  }

  const float prefac = (float) (4 * VMD_PI / npts);
  float totarea = 0.0f;
  // compute area for each atom based on its pairlist
  for (i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      // only atoms in restrictsel contribute
      if (restrictsel && !restrictsel->on[i]) continue;
      const float *loc = framepos+3L*i;
      float rad = radius[i]+srad;
      float surfpos[3];
      int surfpts = npts;
      const ResizeArray<int> &nbrs = pairlist[i];
      for (int j=0; j<npts; j++) {
        surfpos[0] = loc[0] + rad*spherepts[3L*j  ];
        surfpos[1] = loc[1] + rad*spherepts[3L*j+1];
        surfpos[2] = loc[2] + rad*spherepts[3L*j+2];
        int on = 1;
        for (int k=0; k<nbrs.num(); k++) {
          int ind = nbrs[k];
          const float *nbrloc = framepos+3L*ind;
          float radsq = radius[ind]+srad; radsq *= radsq;
          float dx = surfpos[0]-nbrloc[0];
          float dy = surfpos[1]-nbrloc[1];
          float dz = surfpos[2]-nbrloc[2];
          if (dx*dx + dy*dy + dz*dz <= radsq) {
            on = 0;
            break;
          }
        }
        if (on) {
          if (sasapts) {
            sasapts->append3(&surfpos[0]);
          }
        } else {
          surfpts--;
        }
      }
      float atomarea = prefac * rad * rad * surfpts;
      totarea += atomarea;
    }
  }

  delete [] pairlist;
  delete [] spherepts;
  *sasa = totarea;
  return 0;
}



#if 1

// #define DEBUGSASATHR 1

typedef struct {
  MoleculeList *mlist;
  const AtomSel **sellist; 
  int numsels;
  float srad; 
  float *sasalist;
  int nsamples;
  float *spherepts;
} sasathreadparms;

// For different values of the random seed, the computed SASA's of brH.pdb 
// converge to within 1% of each other when the number of points is about
// 500.  We therefore use 500 as the default number.
static void * measure_sasa_thread(void *voidparms) {
  int threadid;
  sasathreadparms *parms = NULL;
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);
  wkf_threadlaunch_getid(voidparms, &threadid, NULL);
#if defined(DEBUGSASATHR)
printf("sasathread[%d] running...\n", threadid);
#endif

  /*
   * copy in per-thread parameters
   */
  MoleculeList *mlist = parms->mlist;
  const AtomSel **sellist = parms->sellist;
//  int numsels = parms->numsels;
  float srad = parms->srad;
  float *sasalist = parms->sasalist;
  const int npts = parms->nsamples;
  float *spherepts = parms->spherepts;

  int i, selidx;
  wkf_tasktile_t tile;
  while (wkf_threadlaunch_next_tile(voidparms, 1, &tile) != WKF_SCHED_DONE) {
#if defined(DEBUGSASATHR)
printf("sasathread[%d] running idx %d to %d\n", threadid, tile.start, tile.end);
#endif
    for (selidx=tile.start; selidx<tile.end; selidx++) {
      const AtomSel *sel = sellist[selidx];
      Molecule *mol = mlist->mol_from_id(sel->molid());

      if (!sel->selected) {
        sasalist[selidx] = 0;
        continue;
      }

      const float *framepos = sel->coordinates(mlist);
      if (!framepos) {
#if defined(DEBUGSASATHR)
printf("measure_sasalist: failed to get coords!!!\n");
#endif
        return NULL; // MEASURE_ERR_NOFRAMEPOS;
      }

      const float *radius = mol->extraflt.data("radius");
      if (!radius) {
#if defined(DEBUGSASATHR)
printf("measure_sasalist: failed to get radii!!!\n");
#endif
        return NULL; // MEASURE_ERR_NORADII;
      }


      float minrad=-1, maxrad=-1;
#if 1
      // Query min/max atom radii for the entire molecule
      mol->get_radii_minmax(minrad, maxrad);
#else
      // find biggest atom radius 
      for (i=0; i<sel->num_atoms; i++) {
        float rad = radius[i];
        if (maxrad < rad) maxrad = rad;
      }
#endif

      // Find atoms within maxrad of each other.  
      // build a list of pairs for each atom
      ResizeArray<int> *pairlist = new ResizeArray<int>[sel->num_atoms];
      {
        GridSearchPair *pairs;
        pairs = vmd_gridsearch1(framepos, sel->num_atoms, sel->on, 
                                2.0f * (maxrad + srad), 0, 
                                sel->num_atoms * 1000);

        GridSearchPair *p, *tmp; 
        for (p = pairs; p != NULL; p = tmp) {
          int ind1=p->ind1;
          int ind2=p->ind2;
          pairlist[ind1].append(ind2);
          pairlist[ind2].append(ind1);
          tmp = p->next;
          free(p);
        }
      }

      const float prefac = (float) (4 * VMD_PI / npts);
      float totarea = 0.0f;
      // compute area for each atom based on its pairlist
      for (i=sel->firstsel; i<=sel->lastsel; i++) {
        if (sel->on[i]) {
          const float *loc = framepos+3L*i;
          float rad = radius[i]+srad;
          float surfpos[3];
          int surfpts = npts;
          const ResizeArray<int> &nbrs = pairlist[i];
          for (int j=0; j<npts; j++) {
            surfpos[0] = loc[0] + rad*spherepts[3L*j  ];
            surfpos[1] = loc[1] + rad*spherepts[3L*j+1];
            surfpos[2] = loc[2] + rad*spherepts[3L*j+2];
            int on = 1;
            for (int k=0; k<nbrs.num(); k++) {
              int ind = nbrs[k];
              const float *nbrloc = framepos+3L*ind;
              float radsq = radius[ind]+srad; radsq *= radsq;
              float dx = surfpos[0]-nbrloc[0];
              float dy = surfpos[1]-nbrloc[1];
              float dz = surfpos[2]-nbrloc[2];
              if (dx*dx + dy*dy + dz*dz <= radsq) {
                on = 0;
                break;
              }
            }
            if (!on) {
              surfpts--;
            }
          }
          float atomarea = prefac * rad * rad * surfpts;
          totarea += atomarea;
        }
      }

      delete [] pairlist;
      sasalist[selidx] = totarea;
    }
  }

  return NULL;
}

#if 1

// For different values of the random seed, the computed SASA's of brH.pdb 
// converge to within 1% of each other when the number of points is about
// 500.  We therefore use 500 as the default number.
extern int measure_sasalist(MoleculeList *mlist,
                            const AtomSel **sellist, int numsels,
                            float srad, float *sasalist, const int *nsamples) {

  // check arguments
  if (!sellist) return MEASURE_ERR_NOSEL;

  int i, rc;
  int npts = nsamples ? *nsamples : NPTS;

#if defined(VMDTHREADS)
  int numprocs = wkf_thread_numprocessors();
#else
  int numprocs = 1;
#endif

#if defined(DEBUGSASATHR)
printf("sasaprocs: %d\n", numprocs);
#endif

  static const float RAND_MAX_INV = 1.0f/VMD_RAND_MAX;

  // Seed the random number generator before each calculation.  This gives
  // reproducible results and still allows a more accurate answer to be
  // obtained by increasing the samples size.  I don't know if this is a
  // "good" seed value or not, I just picked something random-looking.
  vmd_srandom(38572111);

  // All the spheres use the same random points.  
  float *spherepts = new float[3L*npts];
  for (i=0; i<npts; i++) {
    float u1 = (float) vmd_random();
    float u2 = (float) vmd_random();
    float z = 2.0f*u1*RAND_MAX_INV -1.0f;
    float phi = (float) (2.0f*VMD_PI*u2*RAND_MAX_INV);
    float R = sqrtf(1.0f-z*z);
    spherepts[3L*i  ] = R*cosf(phi);
    spherepts[3L*i+1] = R*sinf(phi);
    spherepts[3L*i+2] = z;
  }

  sasathreadparms parms;
  parms.mlist = mlist;
  parms.sellist = sellist;
  parms.numsels = numsels;
  parms.srad = srad;
  parms.sasalist = sasalist;
  parms.nsamples = npts;
  parms.spherepts = spherepts;


  /* spawn child threads to do the work */
  wkf_tasktile_t tile;
  tile.start=0;
  tile.end=numsels;
  rc = wkf_threadlaunch(numprocs, &parms, measure_sasa_thread, &tile);

  delete [] spherepts;

  return rc;
}


#else

// For different values of the random seed, the computed SASA's of brH.pdb 
// converge to within 1% of each other when the number of points is about
// 500.  We therefore use 500 as the default number.
extern int measure_sasalist(MoleculeList *mlist,
                            const AtomSel **sellist, int numsels,
                            float srad, float *sasalist, const int *nsamples) {

  // check arguments
  if (!sellist) return MEASURE_ERR_NOSEL;

  int i;
  int npts = nsamples ? *nsamples : NPTS;

  int selidx;
  for (selidx=0; selidx<numsels; selidx++) {
    const AtomSel *sel = sellist[selidx];
    Molecule *mol = mlist->mol_from_id(sel->molid());

    if (!sel->selected) {
      sasalist[selidx] = 0;
      continue;
    }

    const float *framepos = sel->coordinates(mlist);
    if (!framepos) {
#if defined(DEBUGSASATHR)
printf("measure_sasalist: failed to get coords!!!\n");
#endif
      return MEASURE_ERR_NOFRAMEPOS;
    }

    const float *radius = mol->extraflt.data("radius");
    if (!radius) {
#if defined(DEBUGSASATHR)
printf("measure_sasalist: failed to get radii!!!\n");
#endif
      return MEASURE_ERR_NORADII;
    }

    float minrad=-1, maxrad=-1;
#if 1
    // Query min/max atom radii for the entire molecule
    mol->get_radii_minmax(minrad, maxrad);
#else
    // find biggest atom radius 
    for (i=0; i<sel->num_atoms; i++) {
      float rad = radius[i];
      if (maxrad < rad) maxrad = rad;
    }
#endif

    // Find atoms within maxrad of each other.  
    // build a list of pairs for each atom
    ResizeArray<int> *pairlist = new ResizeArray<int>[sel->num_atoms];
    {
      GridSearchPair *pairs;
      pairs = vmd_gridsearch1(framepos, sel->num_atoms, sel->on, 
                              2.0f * (maxrad + srad), 0, sel->num_atoms * 1000);

      GridSearchPair *p, *tmp; 
      for (p = pairs; p != NULL; p = tmp) {
        int ind1=p->ind1;
        int ind2=p->ind2;
        pairlist[ind1].append(ind2);
        pairlist[ind2].append(ind1);
        tmp = p->next;
        free(p);
      }
    }

    static const float RAND_MAX_INV = 1.0f/VMD_RAND_MAX;
    // Seed the random number generator before each calculation.  This gives
    // reproducible results and still allows a more accurate answer to be
    // obtained by increasing the samples size.  I don't know if this is a
    // "good" seed value or not, I just picked something random-looking.
    vmd_srandom(38572111);

    // All the spheres use the same random points.  
    float *spherepts = new float[3L*npts];
    for (i=0; i<npts; i++) {
      float u1 = (float) vmd_random();
      float u2 = (float) vmd_random();
      float z = 2.0f*u1*RAND_MAX_INV -1.0f;
      float phi = (float) (2.0f*VMD_PI*u2*RAND_MAX_INV);
      float R = sqrtf(1.0f-z*z);
      spherepts[3L*i  ] = R*cosf(phi);
      spherepts[3L*i+1] = R*sinf(phi);
      spherepts[3L*i+2] = z;
    }

    const float prefac = (float) (4 * VMD_PI / npts);
    float totarea = 0.0f;
    // compute area for each atom based on its pairlist
    for (i=sel->firstsel; i<=sel->lastsel; i++) {
      if (sel->on[i]) {
        const float *loc = framepos+3L*i;
        float rad = radius[i]+srad;
        float surfpos[3];
        int surfpts = npts;
        const ResizeArray<int> &nbrs = pairlist[i];
        for (int j=0; j<npts; j++) {
          surfpos[0] = loc[0] + rad*spherepts[3L*j  ];
          surfpos[1] = loc[1] + rad*spherepts[3L*j+1];
          surfpos[2] = loc[2] + rad*spherepts[3L*j+2];
          int on = 1;
          for (int k=0; k<nbrs.num(); k++) {
            int ind = nbrs[k];
            const float *nbrloc = framepos+3L*ind;
            float radsq = radius[ind]+srad; radsq *= radsq;
            float dx = surfpos[0]-nbrloc[0];
            float dy = surfpos[1]-nbrloc[1];
            float dz = surfpos[2]-nbrloc[2];
            if (dx*dx + dy*dy + dz*dz <= radsq) {
              on = 0;
              break;
            }
          }
          if (!on) {
            surfpts--;
          }
        }
        float atomarea = prefac * rad * rad * surfpts;
        totarea += atomarea;
      }
    }

    delete [] pairlist;
    delete [] spherepts;
    sasalist[selidx] = totarea;
  }

  return 0;
}

#endif
#endif



//
// Calculate g(r)
//

// find the minimum image distance for one coordinate component 
// and square the result (orthogonal cells only).
static float fix_pbc_n_sqr(float delta, const float boxby2) {
  while (delta >= boxby2) { delta -= 2.0f * boxby2; }
  while (delta < -boxby2) { delta += 2.0f * boxby2; }
  return delta * delta;
}

// calculate the minimum distance between two positions with respect 
// to the periodic cell (orthogonal cells only).
static float min_dist_with_pbc(const float *a, const float *b, 
                                const float *boxby2) {
  float distsqr;
  distsqr  = fix_pbc_n_sqr(a[0] - b[0], boxby2[0]);
  distsqr += fix_pbc_n_sqr(a[1] - b[1], boxby2[1]);
  distsqr += fix_pbc_n_sqr(a[2] - b[2], boxby2[2]);
  return sqrtf(distsqr);
}

/*! the volume of a spherical cap is defined as:
 * <pre> pi / 9 * h^2 * (3 * r  - h) </pre>
 * with h = height of cap = radius - boxby2.
 * \brief the volume of a sperical cap. */
static inline double spherical_cap(const double &radius, const double &boxby2) {
  return (VMD_PI / 3.0 * (radius - boxby2) * (radius - boxby2)
          * ( 2.0 * radius + boxby2));
}


typedef struct {
  int threadid;
  int threadcount;
  int count_o_start;
  int count_o_end;
  const float *olist;
  int count_i;
  const float *ilist;
  int count_h;
  int *hlist;
  float delta;
  const float *boxby2;
  wkfmsgtimer *msgtp;
  int curframe;
  int maxframe;
} gofrparms_t;
    
// calculate the non-normalized pair-distribution function
// for two lists of atom coordinates and store the resulting
// histogram in the hlist array. orthogonal cell version.
//
// NOTE: this is just the workhorse function. special issues related
// to atoms present in both lists have to be dealt with in the uplevel
// functions, that then also can do various post-processing steps.
extern "C" void * measure_gofr_orth(void *voidparms) {
  // handle per-thread parameters
  gofrparms_t *parms = (gofrparms_t *) voidparms;
  const int count_o_start = parms->count_o_start;
  const int count_o_end   = parms->count_o_end;
  const int count_i       = parms->count_i;
  const int count_h       = parms->count_h;
  const float *olist      = parms->olist;
  const float *ilist      = parms->ilist;
  const float *boxby2     = parms->boxby2;
  wkfmsgtimer *msgtp         = parms->msgtp;
  const int curframe      = parms->curframe;
  const int maxframe      = parms->maxframe;
  int *hlist              = parms->hlist;
  // other local variables
  int i, j, idx;
  float dist;
  const float deltascale = 1.0f / parms->delta;
  int msgcnt=0;

  // loop over the chunk of pairs that was associated to this thread.
  for (i=count_o_start; i<count_o_end; ++i) {

    // print progress messages only for thread(s) that have
    // a timer defined (usually only tid==0).
    if (msgtp && wkf_msg_timer_timeout(msgtp)) {
      char tmpbuf[1024];
      if (msgcnt==0) 
        sprintf(tmpbuf, "timestep %d of %d", curframe, maxframe);
      else
        sprintf(tmpbuf, "timestep %d of %d: (%6.2f %% complete)", curframe, maxframe, 
                (100.0f * i) / (float) (count_o_end - count_o_start + 1));
      msgInfo << "vmd_measure_gofr_orth: " << tmpbuf << sendmsg;
      msgcnt++;
      // XXX we should update the display here...
    }

    for (j=0; j<count_i; ++j) {
      // calculate distance and add to histogram
      dist = min_dist_with_pbc(&olist[i*3L], &ilist[j*3L], boxby2);
      idx = (int) (dist * deltascale);
      if ((idx >= 0) && (idx < count_h)) 
        ++hlist[idx];
    }
  }

  return MEASURE_NOERR;
}

// main entry point for 'measure gofr'.
// tasks:
// - sanity check on arguments.
// - select proper algorithm for PBC treatment.
int measure_gofr(AtomSel *sel1, AtomSel *sel2, MoleculeList *mlist,
                 const int count_h, double *gofr, double *numint, double *histog,
                 const float delta, int first, int last, int step, int *framecntr,
                 int usepbc, int selupdate) {
  int i, j, frame;
  float a, b, c, alpha, beta, gamma;
  int isortho=0;    // orthogonal unit cell not assumed by default.
  int duplicates=0; // counter for duplicate atoms in both selections.

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
      msgErr << "measure gofr: bad frame range given." 
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
        msgErr << "measure gofr: unit cell volume is zero." << sendmsg;
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
    msgErr << "measure gofr: only orthorhombic cells are supported (for now)." << sendmsg;
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

  // setup status message timer
  wkfmsgtimer *msgt = wkf_msg_timer_create(5);

  // threading setup.
  wkf_thread_t *threads;
  gofrparms_t  *parms;
#if defined(VMDTHREADS)
  int numprocs = wkf_thread_numprocessors();
#else
  int numprocs = 1;
#endif

  threads = new wkf_thread_t[numprocs];
  memset(threads, 0, numprocs * sizeof(wkf_thread_t));

  // allocate and (partially) initialize array of per-thread parameters
  parms = new gofrparms_t[numprocs];
  for (i=0; i<numprocs; ++i) {
    parms[i].threadid = i;
    parms[i].threadcount = numprocs;
    parms[i].delta = (float) delta;
    parms[i].msgtp = NULL;
    parms[i].count_h = count_h;
    parms[i].hlist = new int[count_h];
  }

  msgInfo << "measure gofr: using multi-threaded implementation with " 
          << numprocs << ((numprocs > 1) ? " CPUs" : " CPU") << sendmsg;

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
        msgErr << "measure gofr: failed to evaluate atom selection update";
      if (sel2->change(NULL, mymol) != AtomSel::PARSE_SUCCESS)
        msgErr << "measure gofr: failed to evaluate atom selection update";
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

    // clear the histogram for this frame
    // and set up argument structure for threaded execution.
    int maxframe = (int) ((last - first + 1) / ((float) step));
    for (i=0; i<numprocs; ++i) {
      memset(parms[i].hlist, 0, count_h * sizeof(int));
      parms[i].boxby2 = boxby2;
      parms[i].curframe = frame;
      parms[i].maxframe = maxframe;
    }
    parms[0].msgtp = msgt;
    
    if (isortho && sel1->selected && sel2->selected) {
      int count_o = sel1->selected;
      int count_i = sel2->selected;
      const float *olist = sel1coords;
      const float *ilist = sel2coords;
      // make sure the outer loop is the longer one to have 
      // better threading efficiency and cache utilization.
      if (count_o < count_i) {
        count_o = sel2->selected;
        count_i = sel1->selected;
        olist = sel2coords;
        ilist = sel1coords;
      }

      // distribute outer loop across threads in fixed size chunks.
      // this should work very well for small numbers of threads.
      // thrdelta is the chunk size and we need it to be at least 1 
      // _and_ numprocs*thrdelta >= count_o.
      int thrdelta = (count_o + (numprocs-1)) / numprocs;
      int o_min = 0;
      int o_max = thrdelta;
      for (i=0; i<numprocs; ++i, o_min += thrdelta, o_max += thrdelta) {
        if (o_max >  count_o)  o_max = count_o; // curb loop to max
        if (o_min >= count_o)  o_max = - 1;     // no work for this thread. too little data.
        parms[i].count_o_start = o_min;
        parms[i].count_o_end   = o_max;
        parms[i].count_i       = count_i;
        parms[i].olist         = olist;
        parms[i].ilist         = ilist;
      }
      
      // do the gofr calculation for orthogonal boxes.
      // XXX. non-orthogonal box not supported yet. detected and handled above.
#if defined(VMDTHREADS)
      for (i=0; i<numprocs; ++i) {
        wkf_thread_create(&threads[i], measure_gofr_orth, &parms[i]);
      }
      for (i=0; i<numprocs; ++i) {
        wkf_thread_join(threads[i], NULL);
      } 
#else
      measure_gofr_orth((void *) &parms[0]);
#endif
      ++framecntr[2]; // frame processed with _orth algorithm
    } else {
      ++framecntr[1]; // frame skipped
    }
    ++framecntr[0];   // total frames.

    // correct the first histogram slot for the number of atoms that are 
    // present in both lists. they'll end up in the first histogram bin. 
    // we subtract only from the first thread histogram which is always defined.
    parms[0].hlist[0] -= duplicates;

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
      double histv = 0.0;
      for (j=0; j<numprocs; ++j) {
        histv += (double) parms[j].hlist[i];
      }
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

  double norm = 1.0 / (double) nframes;
  for (i=0; i<count_h; ++i) {
    gofr[i]   *= norm;
    numint[i] *= norm;
    histog[i] *= norm;
  }
  wkf_msg_timer_destroy(msgt);

  // release thread-related storage
  for (i=0; i<numprocs; ++i) {
    delete [] parms[i].hlist;
  }
  delete [] threads;
  delete [] parms;

  return MEASURE_NOERR;
}


int measure_geom(MoleculeList *mlist, int *molid, int *atmid, ResizeArray<float> *gValues,
		 int frame, int first, int last, int defmolid, int geomtype) {
  int i, ret_val;
  for(i=0; i < geomtype; i++) {
    // make sure an atom is not repeated in this list
    if(i > 0 && molid[i-1]==molid[i] && atmid[i-1]==atmid[i]) {
      printf("measure_geom: %i/%i %i/%i\n", molid[i-1],atmid[i-1],molid[i],atmid[i]);
      return MEASURE_ERR_REPEATEDATOM;
    }
  }

  float value;
  int max_ts, orig_ts;

  // use the default molecule to determine which frames to cycle through
  Molecule *mol = mlist->mol_from_id(defmolid);
  if( !mol )
    return MEASURE_ERR_NOMOLECULE;
  
  // get current frame number and make sure there are frames
  if((orig_ts = mol->frame()) < 0)
    return MEASURE_ERR_NOFRAMES;
  
  // get the max frame number and determine frame range
  max_ts = mol->numframes()-1;
  if (frame<0) {
    if (first<0 && last<0) first = last = orig_ts; 
    if (last<0 || last>max_ts) last = max_ts;
    if (first<0) first = 0;
  } else {
    if (frame>max_ts) frame = max_ts;
    first = last = frame; 
  }
  
  // go through all the frames, calculating values
  for(i=first; i <= last; i++) {
    mol->override_current_frame(i);
    switch (geomtype) {
    case MEASURE_BOND:
      if ((ret_val=calculate_bond(mlist, molid, atmid, &value))<0)
	return ret_val;
      gValues->append(value);
      break;
    case MEASURE_ANGLE:
      if ((ret_val=calculate_angle(mlist, molid, atmid, &value))<0)
	return ret_val;
      gValues->append(value);
      break;
    case MEASURE_DIHED:
      if ((ret_val=calculate_dihed(mlist, molid, atmid, &value))<0)
	return ret_val;
      gValues->append(value);
      break;
    }
  }
  
  // reset the current frame
  mol->override_current_frame(orig_ts);
  
  return MEASURE_NOERR;
}
  
  
// calculate the value of this geometry, and return it
int calculate_bond(MoleculeList *mlist, int *molid, int *atmid, float *value) {

  // get coords to calculate distance 
  int ret_val;
  float pos1[3], pos2[3];
  if ((ret_val=normal_atom_coord(mlist->mol_from_id(molid[0]), atmid[0], pos1))<0)
    return ret_val;
  if ((ret_val=normal_atom_coord(mlist->mol_from_id(molid[1]), atmid[1], pos2))<0)
    return ret_val;
  
  vec_sub(pos2, pos2, pos1);
  *value = norm(pos2);

  return MEASURE_NOERR;
}

// calculate the value of this geometry, and return it
int calculate_angle(MoleculeList *mlist, int *molid, int *atmid, float *value) {

  // get coords to calculate distance 
  int ret_val;
  float pos1[3], pos2[3], pos3[3], r1[3], r2[3];
  if((ret_val=normal_atom_coord(mlist->mol_from_id(molid[0]), atmid[0], pos1))<0)
    return ret_val;
  if((ret_val=normal_atom_coord(mlist->mol_from_id(molid[1]), atmid[1], pos2))<0)
    return ret_val;
  if((ret_val=normal_atom_coord(mlist->mol_from_id(molid[2]), atmid[2], pos3))<0)
    return ret_val;

  vec_sub(r1, pos1, pos2);
  vec_sub(r2, pos3, pos2);
  *value = angle(r1, r2);

  return MEASURE_NOERR;
}

// calculate the value of this geometry, and return it
int calculate_dihed(MoleculeList *mlist, int *molid, int *atmid, float *value) {

  // get coords to calculate distance 
  int ret_val;
  float pos1[3], pos2[3], pos3[3], pos4[3]; 
  if((ret_val=normal_atom_coord(mlist->mol_from_id(molid[0]), atmid[0], pos1))<0)
    return ret_val;
  if((ret_val=normal_atom_coord(mlist->mol_from_id(molid[1]), atmid[1], pos2))<0)
    return ret_val;
  if((ret_val=normal_atom_coord(mlist->mol_from_id(molid[2]), atmid[2], pos3))<0)
    return ret_val;
  if((ret_val=normal_atom_coord(mlist->mol_from_id(molid[3]), atmid[3], pos4))<0)
    return ret_val;

  *value = dihedral(pos1, pos2, pos3, pos4);

  return MEASURE_NOERR;
}


// for the given Molecule, find the UNTRANSFORMED coords for the given atom
// return Molecule pointer if successful, NULL otherwise.
int normal_atom_coord(Molecule *mol, int a, float *pos) {
  Timestep *now;

  int cell[3];
  memset(cell, 0, 3L*sizeof(int));

  // get the molecule pointer, and get the coords for the current timestep
  int ret_val = check_mol(mol, a);
  if (ret_val<0) 
    return ret_val;

  if ((now = mol->current())) {
    memcpy((void *)pos, (void *)(now->pos + 3L*a), 3L*sizeof(float));
    
    // Apply periodic image transformation before returning
    Matrix4 mat;
    now->get_transform_from_cell(cell, mat);
    mat.multpoint3d(pos, pos);
    
    return MEASURE_NOERR;
  }
  
  // if here, error (i.e. molecule contains no frames)
  return MEASURE_ERR_NOFRAMES;
}


// check whether the given molecule & atom index is OK
// if OK, return Molecule pointer; otherwise, return NULL
int check_mol(Molecule *mol, int a) {

  if (!mol)
    return MEASURE_ERR_NOMOLECULE;
  if (a < 0 || a >= mol->nAtoms)
    return MEASURE_ERR_BADATOMID;
  
  return MEASURE_NOERR;
}


int measure_energy(MoleculeList *mlist, int *molid, int *atmid, int natoms, ResizeArray<float> *gValues,
		 int frame, int first, int last, int defmolid, double *params, int geomtype) {
  int i, ret_val;
  for(i=0; i < natoms; i++) {
    // make sure an atom is not repeated in this list
    if(i > 0 && molid[i-1]==molid[i] && atmid[i-1]==atmid[i]) {
      printf("measure_energy: %i/%i %i/%i\n", molid[i-1],atmid[i-1],molid[i],atmid[i]);
      return MEASURE_ERR_REPEATEDATOM;
    }
  }

  float value;
  int max_ts, orig_ts;

  // use the default molecule to determine which frames to cycle through
  Molecule *mol = mlist->mol_from_id(defmolid);
  if( !mol )
    return MEASURE_ERR_NOMOLECULE;
  
  // get current frame number and make sure there are frames
  if((orig_ts = mol->frame()) < 0)
    return MEASURE_ERR_NOFRAMES;
  
  // get the max frame number and determine frame range
  max_ts = mol->numframes()-1;
  if (frame==-1) {
    if (first<0 && last<0) first = last = orig_ts; 
    if (last<0 || last>max_ts) last = max_ts;
    if (first<0) first = 0;
  } else {
    if (frame>max_ts || frame==-2) frame = max_ts;
    first = last = frame; 
  }
  
  // go through all the frames, calculating values
  for(i=first; i <= last; i++) {
    mol->override_current_frame(i);
    switch (geomtype) {
    case MEASURE_BOND:
      if ((ret_val=compute_bond_energy(mlist, molid, atmid, &value, (float) params[0], (float) params[1]))<0)
	return ret_val;
      gValues->append(value);
      break;
    case MEASURE_ANGLE:
      if ((ret_val=compute_angle_energy(mlist, molid, atmid, &value, (float) params[0], (float) params[1], (float) params[2], (float) params[3]))<0)
	return ret_val;
      gValues->append(value);
      break;
    case MEASURE_DIHED:
      if ((ret_val=compute_dihed_energy(mlist, molid, atmid, &value, (float) params[0], int(params[1]), (float) params[2]))<0)
	return ret_val;
      gValues->append(value);
      break;
    case MEASURE_IMPRP:
      if ((ret_val=compute_imprp_energy(mlist, molid, atmid, &value, (float) params[0], (float) params[1]))<0)
	return ret_val;
      gValues->append(value);
      break;
    case MEASURE_VDW:
      if ((ret_val=compute_vdw_energy(mlist, molid, atmid, &value, (float) params[0], (float) params[1], (float) params[2], (float) params[3], (float) params[4], (float) params[5]))<0)
	return ret_val;
      gValues->append(value);
      break;
    case MEASURE_ELECT:
      if ((ret_val=compute_elect_energy(mlist, molid, atmid, &value, (float) params[0], (float) params[1], (bool) params[2], (bool) params[3], (float) params[4]))<0)
	return ret_val;
      gValues->append(value);
      break;
    }
  }
  
  // reset the current frame
  mol->override_current_frame(orig_ts);
  
  return MEASURE_NOERR;
}
  
// calculate the energy of this geometry
int compute_bond_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy, float k, float x0) {
  int ret_val;
  float dist;

  // Get the coordinates
  if ((ret_val=calculate_bond(mlist, molid, atmid, &dist))<0)
	return ret_val;
  float x = dist-x0;
  *energy = k*x*x;

  return MEASURE_NOERR;
}

// calculate the energy of this geometry
int compute_angle_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy,
			 float k, float x0, float kub, float s0) {
  int ret_val;
  float value;

  // Get the coordinates
  if ((ret_val=calculate_angle(mlist, molid, atmid, &value))<0)
	return ret_val;
  float x = (float) DEGTORAD((value-x0));
  float s = 0.0f;

  if (kub>0.0f) {
    int twoatoms[2];
    twoatoms[0] = atmid[0];
    twoatoms[1] = atmid[2];
    if ((ret_val=calculate_bond(mlist, molid, twoatoms, &value))<0)
      return ret_val;
    s = value-s0;
  }

  *energy = k*x*x + kub*s*s;

  return MEASURE_NOERR;
}

// calculate the energy of this geometry
int compute_dihed_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy,
			 float k, int n, float delta) {
  int ret_val;
  float value;

  // Get the coordinates
  if ((ret_val=calculate_dihed(mlist, molid, atmid, &value))<0)
	return ret_val;
  *energy = k*(1+cosf((float) (DEGTORAD((n*value-delta)))));

  return MEASURE_NOERR;
}

// calculate the energy of this geometry
int compute_imprp_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy,
			 float k, float x0) {
  int ret_val;
  float value;

  // Get the coordinates
  if ((ret_val=calculate_dihed(mlist, molid, atmid, &value))<0)
	return ret_val;
  float x = (float) (DEGTORAD((value-x0)));
  *energy = k*x*x;

  return MEASURE_NOERR;
}

// Calculate the VDW energy for specified pair of atoms
// VDW energy:                                               
// Evdw = eps * ((Rmin/dist)**12 - 2*(Rmin/dist)**6)         
// eps = sqrt(eps1*eps2),  Rmin = Rmin1+Rmin2                
int compute_vdw_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy, float eps1, float rmin1,
			 float eps2, float rmin2, float cutoff, float switchdist) {
  int ret_val;
  float dist;

  // Get the coordinates
  if ((ret_val=calculate_bond(mlist, molid, atmid, &dist))<0)
    return ret_val;

  float sw=1.0;
  if (switchdist>0.0 && cutoff>0.0) {
    if (dist>=cutoff) {
      sw = 0.0;
    } else if (dist>=switchdist) {
      // This is the CHARMM switching function
      float dist2 = dist*dist;
      float cut2 = cutoff*cutoff;
      float switch2 = switchdist*switchdist;
      float s = cut2-dist2;
      float range = cut2-switch2;
      sw = s*s*(cut2+2*dist2-3*switch2)/(range*range*range);
    }
  }

  float term6 = (float) powf((rmin1+rmin2)/dist,6);
  *energy = sqrtf(eps1*eps2)*(term6*term6 - 2.0f*term6)*sw;

  return MEASURE_NOERR;
}

int compute_elect_energy(MoleculeList *mlist, int *molid, int *atmid, float *energy, float q1, float q2,
			 bool flag1, bool flag2, float cutoff) {
  int ret_val;
  float dist;

  // Get the coordinates
  if ((ret_val=calculate_bond(mlist, molid, atmid, &dist))<0)
    return ret_val;

  // Get atom charges
  if (!flag1) q1 = mlist->mol_from_id(molid[0])->charge()[atmid[0]];
  if (!flag2) q2 = mlist->mol_from_id(molid[0])->charge()[atmid[1]];

  if (cutoff>0.0) {
    if (dist<cutoff) {
      float efac = 1.0f-dist*dist/(cutoff*cutoff);
      *energy = 332.0636f*q1*q2/dist*efac*efac;
    } else {
      *energy = 0.0f;
    }
  } else {
    *energy = 332.0636f*q1*q2/dist;
  }

  return MEASURE_NOERR;
}
 

// Compute the center of mass for a given selection.
// The result is put in rcom which has to have a size of at least 3.
static void center_of_mass(AtomSel *sel, MoleculeList *mlist, float *rcom) {
  int i;
  float m = 0, mtot = 0;
  Molecule *mol = mlist->mol_from_id(sel->molid());

  // get atom masses
  const float *mass = mol->mass();

  // get atom coordinates
  const float *pos = sel->coordinates(mlist);

  memset(rcom, 0, 3L*sizeof(float));

  // center of mass
  for (i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      long ind = i * 3L;

      m = mass[i];

      rcom[0] += m*pos[ind    ];
      rcom[1] += m*pos[ind + 1];
      rcom[2] += m*pos[ind + 2];

      // total mass
      mtot += m;
    }
  }

  rcom[0] /= mtot;
  rcom[1] /= mtot;
  rcom[2] /= mtot;
}


// Calculate principle axes and moments of inertia for selected atoms.
// The corresponding eigenvalues are also returned, they can be used
// to see if two axes are equivalent. The center of mass will be put
// in parameter rcom.
// The user can provide his own set of coordinates in coor. If this
// parameter is NULL then the coordinates from the selection are used.
extern int measure_inertia(AtomSel *sel, MoleculeList *mlist, const float *coor, float rcom[3],
			   float priaxes[3][3], float itensor[4][4], float evalue[3]) {
  if (!sel)                     return MEASURE_ERR_NOSEL;
  if (sel->num_atoms == 0)      return MEASURE_ERR_NOATOMS;

  Molecule *mol = mlist->mol_from_id(sel->molid());

  float x, y, z, m;
  float Ixx=0, Iyy=0, Izz=0, Ixy=0, Ixz=0, Iyz=0;
  int i,j=0;

  // need to put 3x3 inertia tensor into 4x4 matrix for jacobi eigensolver
  // itensor = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};
  memset(itensor, 0, 16L*sizeof(float));
  itensor[3][3] = 1.0;

  // compute center of mass
  center_of_mass(sel, mlist, rcom);

  // get atom coordinates
  const float *pos = sel->coordinates(mlist);

  // get atom masses
  const float *mass = mol->mass();


  // moments of inertia tensor
  for (i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      // position relative to COM
      if (coor) {
        // use user provided coordinates
        x = coor[j*3L    ] - rcom[0];
        y = coor[j*3L + 1] - rcom[1];
        z = coor[j*3L + 2] - rcom[2];
        j++;
      } else {
        // use coordinates from selection
        x = pos[i*3L    ] - rcom[0];
        y = pos[i*3L + 1] - rcom[1];
        z = pos[i*3L + 2] - rcom[2];
      }

      m = mass[i];

      Ixx += m*(y*y+z*z);
      Iyy += m*(x*x+z*z);
      Izz += m*(x*x+y*y);
      Ixy -= m*x*y;
      Ixz -= m*x*z;
      Iyz -= m*y*z;
    }
  }

  itensor[0][0] = Ixx;
  itensor[1][1] = Iyy;
  itensor[2][2] = Izz;
  itensor[0][1] = Ixy;
  itensor[1][0] = Ixy;
  itensor[0][2] = Ixz;
  itensor[2][0] = Ixz;
  itensor[1][2] = Iyz;
  itensor[2][1] = Iyz;

  // Find the eigenvalues and eigenvectors of moments of inertia tensor.
  // The eigenvectors correspond to the principle axes of inertia.
  float evector[3][3];
  if (jacobi(itensor,evalue,evector) != 0) return MEASURE_ERR_NONZEROJACOBI;

  // transpose the evector matrix to put the vectors in rows
  float vectmp;
  vectmp=evector[0][1]; evector[0][1]=evector[1][0]; evector[1][0]=vectmp;
  vectmp=evector[0][2]; evector[0][2]=evector[2][0]; evector[2][0]=vectmp;
  vectmp=evector[2][1]; evector[2][1]=evector[1][2]; evector[1][2]=vectmp;


  // sort so that the eigenvalues are from largest to smallest
  // (or rather so a[0] is eigenvector with largest eigenvalue, ...)
  float *a[3];
  a[0] = evector[0];
  a[1] = evector[1];
  a[2] = evector[2];
  // The code for SWAP is copied from measure_fit().
  // It swaps rows in the eigenvector matrix.
#define SWAP(qq,ww) {                                           \
    float v; float *v1;                                         \
    v = evalue[qq]; evalue[qq] = evalue[ww]; evalue[ww] = v;    \
    v1 = a[qq]; a[qq] = a[ww]; a[ww] = v1;                      \
}
  if (evalue[0] < evalue[1]) {
    SWAP(0, 1);
  }
  if (evalue[0] < evalue[2]) {
    SWAP(0, 2);
  }
  if (evalue[1] < evalue[2]) {
    SWAP(1, 2);
  }

#if 0
  // If the 2nd and 3rd eigenvalues are identical and not close to zero
  // then the corresponding axes are not unique. 
  if (evalue[1]/evalue[0]>0.1 && fabs(evalue[1]-evalue[2])/evalue[0]<0.05
      && fabs(evalue[0]-evalue[1])/evalue[0]>0.05) {
    msgInfo << "Principal axes of inertia 2 and 3 are not unique!" << sendmsg;
  }
#endif

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) 
      priaxes[i][j] = a[i][j];
  }

  return MEASURE_NOERR;
}

