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
 *	$RCSfile: BondSearch.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.76 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Distance based bond search code 
 *
 ***************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "BondSearch.h"
#include "Timestep.h"
#include "BaseMolecule.h"
#include "Molecule.h"
#include "Inform.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include <ctype.h>         // needed for isdigit()
#include <string.h>


GridSearchPairlist *vmd_gridsearch_bonds(const float *pos, const float *radii,
                                   int natoms, float pairdist, int maxpairs) {
  float min[3], max[3];
  int i, xb, yb, zb, xytotb, totb;
  int **boxatom, *numinbox, *maxinbox, **nbrlist;
  int numon = 0;
  float sidelen[3], volume;
  int paircount = 0;

  // find bounding box for selected atoms, and number of atoms in selection.
#if 1
  minmax_3fv_aligned(pos, natoms, min, max);
#else
  find_minmax_all(pos, natoms, min, max);
#endif

  // check for NaN coordinates propagating to the bounding box result
  if (!(max[0] >= min[0] && max[1] >= min[1] && max[2] >= min[2])) {
    msgErr << "vmd_gridsearch_bonds: NaN coordinates in bounds, aborting!" << sendmsg;
    return NULL;
  }

  // do sanity checks and complain if we've got bogus atom coordinates,
  // we shouldn't ever have density higher than 0.1 atom/A^3, but we'll
  // be generous and allow much higher densities.  
  if (maxpairs != -1) {
    vec_sub(sidelen, max, min);
    // include estimate for atom radius (1 Angstrom) in volume determination
    volume = fabsf((sidelen[0] + 2.0f) * (sidelen[1] + 2.0f) * (sidelen[2] + 2.0f));
    if ((numon / volume) > 1.0) {
      msgWarn << "vmd_gridsearch_bonds: insane atom density" << sendmsg;
    }
  }

  // I don't want the grid to get too large, otherwise I could run out
  // of memory.  Octrees would be cool, but I'll just limit the grid size
  // and let the performance degrade a little for pathological systems.
  // Note that pairdist^2 is what gets used for the actual distance checks;
  // from here on out pairdist is only used to set the grid size, so we 
  // can set it to anything larger than the original pairdist.
  const int MAXBOXES = 4000000;
  totb = MAXBOXES + 1;

  float newpairdist = pairdist;
  float xrange = max[0]-min[0];
  float yrange = max[1]-min[1];
  float zrange = max[2]-min[2];
  do {
    pairdist = newpairdist;
    const float invpairdist = 1.0f / pairdist; 
    xb = ((int)(xrange*invpairdist))+1;
    yb = ((int)(yrange*invpairdist))+1;
    zb = ((int)(zrange*invpairdist))+1;
    xytotb = yb * xb;
    totb = xytotb * zb;
    newpairdist = pairdist * 1.26f; // cbrt(2) is about 1.26
  } while (totb > MAXBOXES || totb < 1); // check for integer wraparound too
 
  // 2. Sort each atom into appropriate bins
  boxatom = (int **) calloc(1, totb*sizeof(int *));
  numinbox = (int *) calloc(1, totb*sizeof(int));
  maxinbox = (int *) calloc(1, totb*sizeof(int));
  if (boxatom == NULL || numinbox == NULL || maxinbox == NULL) {
    if (boxatom != NULL)
      free(boxatom);
    if (numinbox != NULL)
      free(numinbox);
    if (maxinbox != NULL)
      free(maxinbox);
    msgErr << "Bondsearch memory allocation failed, bailing out" << sendmsg;
    return NULL; // ran out of memory, bail out!
  }

  const float invpairdist = 1.0f / pairdist; 
  for (i=0; i<natoms; i++) {
    int axb, ayb, azb, aindex, num;

    // compute box index for new atom
    const float *loc = pos + 3L*i;
    axb = (int)((loc[0] - min[0])*invpairdist);
    ayb = (int)((loc[1] - min[1])*invpairdist);
    azb = (int)((loc[2] - min[2])*invpairdist);

    // clamp box indices to valid range in case of FP error
    if (axb >= xb) axb = xb-1;
    if (ayb >= yb) ayb = yb-1;
    if (azb >= zb) azb = zb-1;

    aindex = azb * xytotb + ayb * xb + axb;

    // grow box if necessary 
    if ((num = numinbox[aindex]) == maxinbox[aindex]) {
      boxatom[aindex] = (int *) realloc(boxatom[aindex], (num+4)*sizeof(int));
      maxinbox[aindex] += 4;
    }

    // store atom index in box
    boxatom[aindex][num] = i;
    numinbox[aindex]++;
  }
  free(maxinbox);
 
  nbrlist = (int **) calloc(1, totb*sizeof(int *));
  if (make_neighborlist(nbrlist, xb, yb, zb)) {
    if (boxatom != NULL) {
      for (i=0; i<totb; i++) {
        if (boxatom[i] != NULL) free(boxatom[i]);
      }
      free(boxatom);
    }
    if (nbrlist != NULL) {
      for (i=0; i<totb; i++) {
        if (nbrlist[i] != NULL) free(nbrlist[i]);
      }
      free(nbrlist);
    }
    free(numinbox);
    msgErr << "Bondsearch memory allocation failed, bailing out" << sendmsg;
    return NULL; // ran out of memory, bail out!
  }

  // if maxpairs is "unlimited", set it to the biggest positive int
  if (maxpairs < 0) {
    maxpairs = 2147483647;
  }

  // setup head of pairlist
  GridSearchPairlist *head, *cur;
  head = (GridSearchPairlist *) malloc(sizeof(GridSearchPairlist));
  head->next = NULL;
  paircount = vmd_bondsearch_thr(pos, radii, head, totb, 
                                 boxatom, numinbox, nbrlist, 
                                 maxpairs, pairdist);

  for (i=0; i<totb; i++) {
    free(boxatom[i]);
    free(nbrlist[i]);
  }
  free(boxatom);
  free(nbrlist);
  free(numinbox);

  cur = head->next;
  free(head);

  if (paircount > maxpairs) 
    msgErr << "vmdgridsearch_bonds: exceeded pairlist sanity check, aborted" << sendmsg;

  return cur;
}



// bond search thread parameter structure
typedef struct {
  int threadid;
  int threadcount;
  wkf_mutex_t *pairlistmutex;
  GridSearchPairlist * head;
  float *pos;
  float *radii;
  int totb;
  int **boxatom;
  int *numinbox;
  int **nbrlist;  
  int maxpairs;
  float pairdist;
} bondsearchthrparms;

// thread prototype 
extern "C" void * bondsearchthread(void *);

// setup and launch bond search threads
int vmd_bondsearch_thr(const float *pos, const float *radii,
                       GridSearchPairlist * head, 
                       int totb, int **boxatom, 
                       int *numinbox, int **nbrlist, int maxpairs, 
                       float pairdist) {
  int i;
  bondsearchthrparms *parms;
  wkf_thread_t * threads;
  wkf_mutex_t pairlistmutex; ///< guards pairlist
  wkf_mutex_init(&pairlistmutex); // init mutex before use

  int numprocs = wkf_thread_numprocessors();

  /* allocate array of threads */
  threads = (wkf_thread_t *) calloc(numprocs * sizeof(wkf_thread_t), 1);

  /* allocate and initialize array of thread parameters */
  parms = (bondsearchthrparms *) malloc(numprocs * sizeof(bondsearchthrparms));
  for (i=0; i<numprocs; i++) {
    parms[i].threadid = i;
    parms[i].threadcount = numprocs;
    parms[i].pairlistmutex = &pairlistmutex;
    parms[i].head = NULL;
    parms[i].pos = (float *) pos;
    parms[i].radii = (float *) radii;
    parms[i].totb = totb;
    parms[i].boxatom = boxatom;
    parms[i].numinbox = numinbox;
    parms[i].nbrlist = nbrlist;  
    parms[i].maxpairs = maxpairs;
    parms[i].pairdist = pairdist;
  }

#if defined(VMDTHREADS)
  /* spawn child threads to do the work */
  for (i=0; i<numprocs; i++) {
    wkf_thread_create(&threads[i], bondsearchthread, &parms[i]);
  }

  /* join the threads after work is done */
  for (i=0; i<numprocs; i++) {
    wkf_thread_join(threads[i], NULL);
  }
#else
  bondsearchthread(&parms[0]); // single-threaded code
#endif

  // assemble final pairlist from sublists
  for (i=0; i<numprocs; i++) {
    if (parms[i].head != NULL) {
      GridSearchPairlist *tmp = head->next;
      head->next = parms[i].head;
      parms[i].head->next = tmp;
    }
  }

  wkf_mutex_destroy(&pairlistmutex); // destroy mutex when finished

  /* free thread parms */
  free(parms);
  free(threads);

  return 0;
}

extern "C" void * bondsearchthread(void *voidparms) {
  int i, j, aindex;
  int paircount = 0;

  bondsearchthrparms *parms = (bondsearchthrparms *) voidparms;

  const int threadid = parms->threadid;
  const int threadcount = parms->threadcount;
  wkf_mutex_t *pairlistmutex = parms->pairlistmutex;
  const float *pos = parms->pos;
  const float *radii = parms->radii;
  const int totb = parms->totb;
  const int **boxatom = (const int **) parms->boxatom;
  const int *numinbox = parms->numinbox;
  const int **nbrlist = (const int **) parms->nbrlist; 
  const int maxpairs = parms->maxpairs;
  const float pairdist = parms->pairdist;

  ResizeArray<int> *pairs = new ResizeArray<int>;
  float sqdist = pairdist * pairdist;

  wkfmsgtimer *msgt = wkf_msg_timer_create(5);
  for (aindex = threadid; (aindex < totb) && (paircount < maxpairs); aindex+=threadcount) {
    const int *tmpbox, *nbr;
    tmpbox = boxatom[aindex];
    int anbox = numinbox[aindex];

    if (((aindex & 255) == 0) && wkf_msg_timer_timeout(msgt)) {
      char tmpbuf[128];
      sprintf(tmpbuf, "%6.2f", (100.0f * aindex) / (float) totb);
//  XXX: we have to use printf here as msgInfo is not thread-safe.
//  msgInfo << "vmd_gridsearch_bonds (thread " << threadid << "): " 
//          << tmpbuf << "% complete" << sendmsg;
      printf("vmd_gridsearch_bonds (thread %d): %s %% complete\n", 
        threadid, tmpbuf); 
    }

    for (nbr = nbrlist[aindex]; (*nbr != -1) && (paircount < maxpairs); nbr++) {
      int nnbr=*nbr;
      const int *nbrbox = boxatom[nnbr];
      int nbox=numinbox[nnbr];
      int self = (aindex == nnbr) ? 1 : 0;

      for (i=0; (i<anbox) && (paircount < maxpairs); i++) {
        int ind1 = tmpbox[i];
        const float *p1 = pos + 3L*ind1;

        // skip over self and already-tested atoms
        int startj = (self) ? i+1 : 0;

        for (j=startj; (j<nbox) && (paircount < maxpairs); j++) {
          int ind2 = nbrbox[j];
          const float *p2 = pos + 3L*ind2;
          float dx = p1[0] - p2[0];
          float dy = p1[1] - p2[1];
          float dz = p1[2] - p2[2];
          float ds2 = dx*dx + dy*dy + dz*dz;

          // perform distance test, but ignore pairs between atoms
          // with nearly identical coords
          if ((ds2 > sqdist) || (ds2 < 0.001))
            continue;

          if (radii) { // Do atom-specific distance check
            float cut = 0.6f * (radii[ind1] + radii[ind2]);
            if (ds2 > cut*cut)
              continue;
          }

          pairs->append(ind1);
          pairs->append(ind2);
          paircount++;
        }
      }
    }
  }

  // setup results pairlist node
  GridSearchPairlist *head;
  head = (GridSearchPairlist *) malloc(sizeof(GridSearchPairlist));
  head->next = NULL;
  head->pairlist = pairs;

  wkf_mutex_lock(pairlistmutex);   // lock pairlist before update
  parms->head = head;
  wkf_mutex_unlock(pairlistmutex); // unlock pairlist after update

  wkf_msg_timer_destroy(msgt);

  return NULL;
}





// determine bonds from position of atoms previously read.
// If cutoff < 0, use vdw radius to determine if bonded.
int vmd_bond_search(BaseMolecule *mol, const Timestep *ts, 
                    float cutoff, int dupcheck) {
  const float *pos;
  int natoms;
  int i;
  const float *radius = mol->radius();
 
  if (ts == NULL) {
    msgErr << "Internal Error: NULL Timestep in vmd_bond_search" << sendmsg;
    return 0;
  }

  natoms = mol->nAtoms; 
  if (natoms == 0 || cutoff == 0.0)
    return 1;

  msgInfo << "Determining bond structure from distance search ..." << sendmsg;

  if (dupcheck)
    msgInfo << "Eliminating bonds duplicated from existing structure..." << sendmsg;

  // Set box distance to either the cutoff, or 1.2 times the largest VDW radius
  float dist = cutoff; 
  if (cutoff < 0.0) {
    // set minimum cutoff distance for the case when the loaded molecule
    // has radii set to zero.  This must be >0.0 or the grid search will hang.
    dist = 0.833f;
    for (i=0; i<natoms; i++) {  
      float rad = radius[i];
      if (rad > dist) dist = rad;
    }
    dist = 1.2f * dist;
  }
  
  pos = ts->pos; 
 
  // Call the bond search to get all atom pairs within the specified distance
  // XXX set maxpairs to 27 bonds per atom, which ought to be ridiculously high
  //     for any real structure someone would load, but well below N^2
  GridSearchPairlist *pairlist = vmd_gridsearch_bonds(pos, 
                                                 (cutoff < 0) ? radius : NULL,
                                                 natoms, dist, natoms * 27L);

  // Go through the pairlist adding validated bonds freeing nodes as we go. 
  GridSearchPairlist *p, *tmp; 
  for (p = pairlist; p != NULL; p = tmp) {
    int numpairs = p->pairlist->num() / 2;
    
    for (i=0; i<numpairs; i++) {
      int ind1 = (*p->pairlist)[i*2L  ]; 
      int ind2 = (*p->pairlist)[i*2L+1];

      MolAtom *atom1 = mol->atom(ind1);
      MolAtom *atom2 = mol->atom(ind2);

      // don't bond atoms that aren't part of the same conformation
      // or that aren't in the all-conformations part of the structure
      if ((atom1->altlocindex != atom2->altlocindex) &&
          ((mol->altlocNames.name(atom1->altlocindex)[0] != '\0') && 
           (mol->altlocNames.name(atom2->altlocindex)[0] != '\0'))) {
        continue;
      }

      // Prevent hydrogens from bonding with each other.
#if 1
      // XXX must use the atom name strings presently because the
      // hydrogen flags aren't necessarily set by the time the bond search
      // code executes.  It may soon be time to do something a different
      // with per-atom flag storage so that the atom types can be setup
      // during the earliest phase of structure analysis, eliminating this
      // and other potential gotchas.
      if (!IS_HYDROGEN(mol->atomNames.name(atom1->nameindex)) ||
          !IS_HYDROGEN(mol->atomNames.name(atom2->nameindex)) ) {
#else
      // Use atomType info derived during initial molecule analysis for speed.
      if (!(atom1->atomType == ATOMHYDROGEN) ||
          !(atom2->atomType == ATOMHYDROGEN)) {
#endif
        // Add a bond, bondorder defaults to 1, bond type to -1
        if (dupcheck)
          mol->add_bond_dupcheck(ind1, ind2, 1, -1);
        else
          mol->add_bond(ind1, ind2, 1, -1);
      }
    }

    // free this pairlist node and its ResizeArray of pairs
    tmp = p->next;
    delete p->pairlist;
    free(p);
  }

  return 1;
}




