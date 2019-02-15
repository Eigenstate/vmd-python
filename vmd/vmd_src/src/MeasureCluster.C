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
 *      $RCSfile: MeasureCluster.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.19 $       $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Code to find clusters in MD trajectories.
 * Current implementation is based on the quality threshold (QT) algorithm:
 *   http://dx.doi.org/10.1101/gr.9.11.1106
 *   http://en.wikipedia.org/wiki/Cluster_analysis#QT_clustering_algorithm
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
#include "SpatialSearch.h"

class AtomSelThr : public AtomSel
{
public:
  AtomSelThr(AtomSel *osel, wkf_mutex_t *olock)
      : AtomSel(NULL, osel->molid()),
        sel(osel), lock(olock) {
    if (sel) {
      selected=sel->selected;
      num_atoms=sel->num_atoms;
      which_frame=sel->which_frame;
      if (sel->on) {
        on = new int[num_atoms];
        memcpy(on,sel->on,num_atoms*sizeof(int));
      }
    } else {
      selected=-1;
      num_atoms=-1;
      which_frame=-1;
    }
  }

  ~AtomSelThr() {
    sel=NULL;
  }

  // disable these methods
private:
  AtomSelThr() : AtomSel(NULL,-1) {};
  AtomSelThr& operator=(const AtomSelThr &) { return *this; };
  AtomSelThr(AtomSelThr &) : AtomSel(NULL,-1) {};
  int change(const  char *newcmd, DrawMolecule *mol) { return NO_PARSE; }

public:

  /* thread safe selection update */
  void update(/* const */ DrawMolecule *mol, const int frame) {
    if (!sel) return;
    
    wkf_mutex_lock(lock);

    sel->which_frame=frame;
    which_frame=frame;

    if (sel->change(NULL, mol) != AtomSel::PARSE_SUCCESS)
      msgErr << "AtomSelThr::update(): failed to evaluate atom selection update";

    num_atoms=sel->num_atoms;
    selected=sel->selected;
    if (!on) on = new int[num_atoms];
    memcpy(on,sel->on,num_atoms*sizeof(int));
    
    wkf_mutex_unlock(lock);
  }
  
protected:
  AtomSel *sel;
  wkf_mutex_t *lock;
};

/* 
   XXX: below is a custom version of MatrixFitRMS. unlike the
        original in fitrms.c, this one computes and provides
        the RMS and does not output the transformation matrix
        (not needed below).

        this needs to go away as soon as an improved general
        version of MatrixFitRMS is available, where this feature
        would be made an option. 
*/

/*

Code in this file was taken from PyMol v0.90 and used by permissing under
the following license agreement contained in the PyMol distribution.  
Trivial modifications have been made to permit incorporation into VMD.



PyMOL Copyright Notice
======================

The PyMOL source code is copyrighted, but you can freely use and copy
it as long as you don't change or remove any of the copyright notices.

----------------------------------------------------------------------
PyMOL is Copyright 1998-2003 by Warren L. DeLano of 
DeLano Scientific LLC, San Carlos, CA, USA (www.delanoscientific.com).

                        All Rights Reserved

Permission to use, copy, modify, distribute, and distribute modified 
versions of this software and its documentation for any purpose and 
without fee is hereby granted, provided that the above copyright 
notice appear in all copies and that both the copyright notice and 
this permission notice appear in supporting documentation, and that 
the names of Warren L. DeLano or DeLano Scientific LLC not be used in 
advertising or publicity pertaining to distribution of the software 
without specific, written prior permission.

WARREN LYFORD DELANO AND DELANO SCIENTIFIC LLC DISCLAIM ALL WARRANTIES 
WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL WARREN LYFORD DELANO
OR DELANO SCIENTIFIC LLC BE LIABLE FOR ANY SPECIAL, INDIRECT OR 
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE 
USE OR PERFORMANCE OF THIS SOFTWARE.
----------------------------------------------------------------------

Where indicated, portions of the PyMOL system are instead protected
under the copyrights of the respective authors.  However, all code in
the PyMOL system is released as non-restrictive open-source software
under the above license or an equivalent license.  

PyMOL Trademark Notice
======================

PyMOL(TM) is a trademark of DeLano Scientific LLC.  Derivate software
which contains PyMOL source code must be plainly distinguished from
the PyMOL package distributed by DeLano Scientific LLC in all publicity,
advertising, and documentation.

The slogans, "Includes PyMOL(TM).", "Based on PyMOL(TM) technology.",
"Contains PyMOL(TM) source code.", and "Built using PyMOL(TM).", may
be used in advertising, publicity, and documentation of derivate
software provided that the notice, "PyMOL is a trademark of DeLano
Scientific LLC.", is included in a footnote or at the end of the document.

All other endorsements employing the PyMOL trademark require specific,
written prior permission.

--Warren L. DeLano (warren@delanoscientific.com)

*/

#ifdef __cplusplus
extern "C" {
#endif

#ifdef R_SMALL
#undef R_SMALL
#endif
#define R_SMALL 0.000000001

static void normalize3d(double *v) {
  double vlen;
  vlen = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (vlen > R_SMALL) {
    v[0] /= vlen;
    v[1] /= vlen;
    v[2] /= vlen;
  } else {
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
  }
}

/*========================================================================*/
  static float MyMatrixFitRMS(int n, float *v1, float *v2, const float *wt, const double tol)
{
  /*
        Subroutine to do the actual RMS fitting of two sets of vector coordinates
        This routine does not rotate the actual coordinates, but instead returns 
        the RMS fitting value, along with the center-of-mass translation vectors 
        T1 and T2 and the rotation vector M, which rotates the translated 
        coordinates of molecule 2 onto the translated coordinates of molecule 1.
  */

  float *vv1,*vv2;
  double m[3][3],aa[3][3];
  double sumwt, sig, gam;
  double sg, bb, cc, tmp, err, etmp;
  int a, b, c, maxiter, iters, iy, iz;
  double t1[3],t2[3];
  double aatmp[9];

  /* Initialize arrays. */

  for(a=0;a<3;a++) {
    for(b=0;b<3;b++) {
      m[a][b] = 0.0;
      aa[a][b] = 0.0;
      aatmp[3*a+b] = 0;
    }
    m[a][a] = 1.0;
    t1[a]=0.0;
    t2[a]=0.0;
  }

  /* maximum number of fitting iterations */
  maxiter = 1000;

  /* Calculate center-of-mass vectors */

  vv1=v1;
  vv2=v2;
  sumwt = 0.0;

  for(c=0;c<n;c++) {
    double w = wt ? wt[c] : 1;
    t1[0] += w * vv1[0];
    t1[1] += w * vv1[1];
    t1[2] += w * vv1[2];
    t2[0] += w * vv2[0];
    t2[1] += w * vv2[1];
    t2[2] += w * vv2[2];
    sumwt += w;
    vv1+=3;
    vv2+=3;
  }
  for(a=0;a<3;a++) {
    t1[a] /= sumwt;
    t2[a] /= sumwt;
  }

  /* Calculate correlation matrix */
  vv1=v1;
  vv2=v2;
  for(c=0;c<n;c++) {
    double w = wt ? wt[c] : 1;
    double x1 = w * (vv1[0] - t1[0]);
    double y1 = w * (vv1[1] - t1[1]);
    double z1 = w * (vv1[2] - t1[2]);

    /* don't multply x2/y2/z2 by w, otherwise weights get squared */
    double x2 =     (vv2[0] - t2[0]); 
    double y2 =     (vv2[1] - t2[1]);
    double z2 =     (vv2[2] - t2[2]);
    aatmp[0] += x2 * x1;
    aatmp[1] += x2 * y1;
    aatmp[2] += x2 * z1;
    aatmp[3] += y2 * x1;
    aatmp[4] += y2 * y1;
    aatmp[5] += y2 * z1;
    aatmp[6] += z2 * x1;
    aatmp[7] += z2 * y1;
    aatmp[8] += z2 * z1;
    vv1+=3;
    vv2+=3;
  }

  for (a=0; a<3; a++) 
    for (b=0; b<3; b++) 
      aa[a][b] = aatmp[3*a+b];

  if(n>1) {
    /* Primary iteration scheme to determine rotation matrix for molecule 2 */
    iters = 0;
    while(1) {
      /* IX, IY, and IZ rotate 1-2-3, 2-3-1, 3-1-2, etc.*/
      iz = (iters+1) % 3;
      iy = (iz+1) % 3;
      // unused...
      // ix = (iy+1) % 3;
      sig = aa[iz][iy] - aa[iy][iz];
      gam = aa[iy][iy] + aa[iz][iz];

      if(iters>=maxiter) {
        fprintf(stderr,
                " Matrix: Warning: no convergence (%.15f>%.15f after %d iterations).\n",
                fabs(sig),tol*fabs(gam),iters);
        break;
      }

      /* Determine size of off-diagonal element.  If off-diagonals exceed the
         diagonal elements * tolerance, perform Jacobi rotation. */
      tmp = sig*sig + gam*gam;
      sg = sqrt(tmp);
      if( (sg > 0.0) && (fabs(sig)>(tol*fabs(gam))) ) {
        sg = 1.0 / sg;
        for(a=0;a<3;a++) {
          bb = gam*aa[iy][a] + sig*aa[iz][a];
          cc = gam*aa[iz][a] - sig*aa[iy][a];
          aa[iy][a] = bb*sg;
          aa[iz][a] = cc*sg;

          bb = gam*m[iy][a] + sig*m[iz][a];
          cc = gam*m[iz][a] - sig*m[iy][a];
          m[iy][a] = bb*sg;
          m[iz][a] = cc*sg;
        }
      } else {
        break;
      }
      iters++;
    }
  }
  /* At this point, we should have a converged rotation matrix (M).  Calculate
         the weighted RMS error. */
  err=0.0;
  vv1=v1;
  vv2=v2;

  normalize3d(m[0]);
  normalize3d(m[1]);
  normalize3d(m[2]);

  for(c=0;c<n;c++) {
	etmp = 0.0;
	for(a=0;a<3;a++) {
	  tmp = m[a][0]*(vv2[0]-t2[0])
		+ m[a][1]*(vv2[1]-t2[1])
		+ m[a][2]*(vv2[2]-t2[2]);
	  tmp = (vv1[a]-t1[a])-tmp;
	  etmp += tmp*tmp;
	}

	if(wt)
	  err += wt[c] * etmp;
	else 
	  err += etmp;

	vv1+=3;
	vv2+=3;
  }

  err=err/sumwt;
  err=sqrt(err);
  return (float)err;
}

#ifdef __cplusplus
}
#endif

/* XXX: end of customized MatrixFitRMS */


// compute weighted RMSD between selected atoms in two frames
// this is a simple wrapper around measure_rmsd.
static float cluster_get_rmsd(const float *Frame1Pos, const float *Frame2Pos, 
                              AtomSel *sel, float *weights) {
  float distance = 0.0f;
  measure_rmsd(sel, sel, sel->num_atoms, Frame1Pos, Frame2Pos, weights, &distance);
  return distance;
}


// compute weighted difference between radius of gyration
// of the selected atoms in the two frames.
static float cluster_get_rgyrd(const float *Frame1Pos, const float *Frame2Pos, 
                               AtomSel *sel, float *weights) {

  float distance = 10000000.0f;

  // compute the center of mass with the current weights
  float com1[3], com2[3];
  int ret_val;

  ret_val = measure_center(sel, Frame1Pos, weights, com1);
  if (ret_val < 0) 
    return distance;

  ret_val = measure_center(sel, Frame2Pos, weights, com2);
  if (ret_val < 0) 
    return distance;
  
  // measure center of gyration
  int i, j;
  float total_w, w, sum1, sum2;
  total_w=sum1=sum2=0.0f;
  for (j=0,i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      w = weights[j];
      total_w += w;
      sum1 += w * distance2(Frame1Pos + 3*i, com1);
      sum2 += w * distance2(Frame2Pos + 3*i, com2);
      j++;
    }
  }

  if (total_w == 0.0f)
    return distance;

  // and finalize the computation
  distance = sqrtf(sum1/total_w) - sqrtf(sum2/total_w);
  return fabsf(distance);
}


// This is a stripped down version of measure fit supporting only
// selections with 4 atoms or larger. not much value in clustering
// smaller systems with fit and rmsd.
// This algorithm comes from Kabsch, Acta Cryst. (1978) A34, 827-828.
static float cluster_get_fitrmsd(const float *Frame1Pos, const float *Frame2Pos, 
                                 AtomSel *sel, float *weights, const double tol) {

  int num = sel->selected;
  
  // failure of the fit+rmsd is indicated by a very large distance.
  float distance = 10000000.0f;

  // use the new RMS fit implementation only
  // fit+clustering with 3 or less atoms doesn't make much sense.
  // the Kabsch method won't work of the number of atoms is less than 4
  // (and won't work in some cases of n > 4; I think it works so long as
  // three or more planes are needed to intersect all the data points

  if (sel->selected < 4)
    return distance;
    
  int i, j, k;
  float *v1, *v2, *wt;
  v1 = new float[3*num];
  v2 = new float[3*num];
  wt = new float[num];
  for (j=0,k=0,i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      int ind = 3 * i;
      wt[j] = weights[i];
      ++j;
      v1[k] = Frame1Pos[ind];
      v2[k] = Frame2Pos[ind];
      v1[k+1] = Frame1Pos[ind+1];
      v2[k+1] = Frame2Pos[ind+1];
      v1[k+2] = Frame1Pos[ind+2];
      v2[k+2] = Frame2Pos[ind+2];
      k+=3;
    }
  }
  distance = MyMatrixFitRMS(num, v1, v2, wt, tol);

  delete [] v1;
  delete [] v2;
  delete [] wt;

  return distance;
}




typedef struct {
  int threadid;
  int threadcount;

  int max_cluster_size;
  const int *skip_list;
  int *new_skip_list;
  int *max_cluster;

  int istart;
  int iend;
  int *frames_list;
  int numframes;

  AtomSelThr *sel;
  Molecule *mol;
  int selupdate;
  float cutoff;
  int likeness;
  float *weights;
} clusterparms_t;


// cluster search thread worker function
extern "C" void * find_cluster_thr(void *voidparms)
{

  clusterparms_t *parms = (clusterparms_t *)voidparms;
  const int istart = parms->istart;
  const int iend   = parms->iend;
  int *framesList = parms->frames_list;
  const int numframes = parms->numframes;

  const int selupdate = parms->selupdate;
  const int likeness = parms->likeness;
  float cutoff = parms->cutoff;
  float *weights = parms->weights;

  AtomSelThr *sel = parms->sel;
  Molecule *mymol = parms->mol;
  const int *skipList = parms->skip_list;

  int *maxCluster = parms->max_cluster;
  memset(maxCluster, 0, numframes*sizeof(int));
  int *newSkipList = parms->new_skip_list;
  memset(newSkipList, 0, numframes*sizeof(int));

  int maxClusterSize = 0, tempClusterSize = 0;
  int *tempCluster = new int[numframes];
  int *tempSkipList = new int[numframes];


  // MatrixFitRMS returns RMS distance of fitted molecule. 
  /* RMS fit tolerance */
  double tol = 1e-15;
  const char *TOL = getenv( "VMDFITRMSTOLERANCE" );
  if (TOL)
    tol = atof(TOL);

  // Loops through assigned frames find the one with the max cluster size
  int i,j;
  for (i = istart; i < iend; i++) {
    memset(tempSkipList, 0, numframes*sizeof(int));
    memset(tempCluster, 0, numframes*sizeof(int));

    if (skipList[i]==0) {
      if (selupdate)
        sel->update(mymol,framesList[i]);
      
      const Timestep *tsMain = mymol->get_frame(framesList[i]);
      const float *framePos = tsMain->pos;

      tempCluster[0] = i;
      tempSkipList[i] = 1;
      tempClusterSize = 1;

      // Loops through all frames other then frame i and computes frame i's distance to them
      for (j = 0; j < numframes; j++) {
        if (skipList[j]==0 && j != i) {
          const Timestep *ts2;
          ts2 = mymol->get_frame(framesList[j]);
          float distance;

          // branch to the implemented likeness algorithms
          switch(likeness) {
            case MEASURE_DIST_RMSD:
              distance = cluster_get_rmsd(framePos, ts2->pos, sel, weights);
              break;

            case MEASURE_DIST_FITRMSD:
              distance = cluster_get_fitrmsd(framePos, ts2->pos, sel, weights, tol);
              break;
              
            case MEASURE_DIST_RGYRD:
              distance = cluster_get_rgyrd(framePos, ts2->pos, sel, weights);
              break;
              
            default:
              distance = 10000000.0f;
          }

          if (distance <= cutoff) {
            tempCluster[tempClusterSize] = j;
            ++tempClusterSize;
            tempSkipList[j] = 1;
          }
        }
      }

      // If size of temp cluster > max cluster, temp cluster becomes max cluster
      if (tempClusterSize > maxClusterSize) {
        int *temp;
        maxClusterSize = tempClusterSize;

        temp = maxCluster;
        maxCluster = tempCluster;
        tempCluster = temp;

        temp = newSkipList;
        newSkipList = tempSkipList;
        tempSkipList = temp;
      }
    }
  }

  // update parameter struct with results
  parms->max_cluster_size = maxClusterSize;
  parms->max_cluster = maxCluster;
  parms->new_skip_list = newSkipList;
  
  // cleanup
  delete[] tempCluster;
  delete[] tempSkipList;

  return MEASURE_NOERR;
}
    

/// find the next largest cluster for the selected range of time steps
static int *find_next_cluster(Molecule *mymol, int *framesList, const int numframes, 
                              const int remframes, const int *skipList, int **newSkipList,
                              const int likeness, AtomSel *sel, const int selupdate, 
                              const double cutoff, float *weights)
{
  int i,j;
  
  // threading setup.
  wkf_thread_t   *threads;
  clusterparms_t *parms;

#if defined(VMDTHREADS)
  int numprocs = wkf_thread_numprocessors();
#else
  int numprocs = 1;
#endif

  int delta = remframes / numprocs;
  int istart = 0;
  int iend = 0;
  
  // not enough work to do, force serial execution
  if (delta < 1) {
    numprocs=1;
    delta=numframes;
  }

  threads = new wkf_thread_t[numprocs];
  memset(threads, 0, numprocs * sizeof(wkf_thread_t));
  wkf_mutex_t *atomsel_lock = new wkf_mutex_t;
  wkf_mutex_init(atomsel_lock);

  // allocate and (partially) initialize array of per-thread parameters
  parms = new clusterparms_t[numprocs];
  for (i=0; i<numprocs; ++i) {
    parms[i].threadid = i;
    parms[i].threadcount = numprocs;

    parms[i].max_cluster_size = 1;
    parms[i].skip_list = skipList;
    parms[i].new_skip_list = new int[numframes];
    parms[i].max_cluster = new int[numframes];

    // use a thread-safe wrapper to access "the one" 
    // AtomSel class. The wrapper uses mutexes to 
    // prevent from updating the global selection from
    // multiple threads at the same time. The whole data 
    // access infrastructure in VMD is currently not thread-safe.
    parms[i].sel = new AtomSelThr(sel,atomsel_lock);
    parms[i].mol = mymol;

    // load balancing. scatter the remaining frames evenly
    // by skipping over eliminated frames.
    parms[i].istart = istart;
    int nframe=0;
    for (j=istart; (j < numframes) && (nframe < delta); ++j) {
      if (skipList[framesList[j]]==0) 
        ++nframe;
      iend=j;
    }
    parms[i].iend = iend;
    istart=iend;
                   
    parms[i].frames_list = framesList;
    parms[i].numframes = numframes;
    parms[i].selupdate = selupdate;
    parms[i].likeness = likeness;
    parms[i].cutoff = (float) cutoff;
    parms[i].weights = weights;
  }
  parms[numprocs-1].iend=numframes;


#if defined(VMDTHREADS)
  if (numprocs > 1) {
    for (i=0; i<numprocs; ++i) {
      wkf_thread_create(&threads[i], find_cluster_thr, &parms[i]);
    }
    for (i=0; i<numprocs; ++i) {
      wkf_thread_join(threads[i], NULL);
    }
  } else
#endif
  find_cluster_thr(&parms[0]);
  
  int maxClusterSize = parms[0].max_cluster_size;
  int *maxCluster = parms[0].max_cluster;
  delete[] *newSkipList;
  *newSkipList= parms[0].new_skip_list;

  // retrieve results from additional threads,
  // override, if needed, and free temporary storage.
  if (numprocs > 1) {
    for (i = 1; i < numprocs; i++) {
      if (parms[i].max_cluster_size > maxClusterSize) {
        maxClusterSize = parms[i].max_cluster_size;
        delete[] maxCluster;
        maxCluster = parms[i].max_cluster;
        delete[] *newSkipList;
        *newSkipList = parms[i].new_skip_list;
      } else {
        delete[] parms[i].max_cluster;
        delete[] parms[i].new_skip_list;
      }
    }
  }
    
  // Transform cluster list back to real frame numbers
  for (i = 0; i < numframes; i++) {
    maxCluster[i] = framesList[maxCluster[i]];
  }

  // cleanup.
  wkf_mutex_destroy(atomsel_lock);
  delete atomsel_lock;

  if (selupdate) {
    for (i=0; i<numprocs; ++i)
      delete parms[i].sel;
  }
  delete[] threads;
  delete[] parms;

  return maxCluster;
}

int measure_cluster(AtomSel *sel, MoleculeList *mlist,
                    const int numcluster, const int algorithm,
                    const int likeness, const double cutoff,
                    int *clustersize, int **clusterlist,
                    int first, int last, int step, int selupdate, 
                    float *weights)
{
  Molecule *mymol = mlist->mol_from_id(sel->molid());
  int maxframe = mymol->numframes()-1;

  if (last == -1) last = maxframe;

  if ((last < first) || (last < 0) || (step <=0) || (first < 0)
      || (last > maxframe)) {
    msgErr << "measure cluster: bad frame range given."
           << " max. allowed frame#: " << maxframe << sendmsg;
    return MEASURE_ERR_BADFRAMERANGE;
  }

  int numframes = (last-first+1)/step;
  int remframes = numframes;

  // create list with frames numbers selected to process
  int *framesList = new int[numframes];
  int frame_count = 0;
  int n;

  for(n = first; n <= last; n += step)
    framesList[frame_count++] = n;

  // accumulated list of frames to skip because they belong to a cluster
  int *skipList = new int[numframes];
  // new list of frames to skip to be added to the existing skip list.
  int *newSkipList = new int[numframes];
  // initially we want all frames.
  memset(skipList, 0, numframes*sizeof(int));
  // timer for progress messages
  wkfmsgtimer *msgtp = wkf_msg_timer_create(5);

  // compute the numcluster largest clusters
  for(n = 0; n < numcluster; ++n){

    // wipe out list of frames to be added to the global skiplist
    memset(newSkipList, 0, numframes*sizeof(int));
    clusterlist[n] = find_next_cluster(mymol, framesList, numframes, remframes,
                                       skipList, &newSkipList, likeness,
                                       sel, selupdate, cutoff, weights);
    int n_cluster=0;
    for(int i = 0; i < numframes; ++i){
      if (newSkipList[i] == 1) {
        skipList[i] = 1;
        n_cluster++;
      }
    }
    clustersize[n]=n_cluster;
    remframes -= n_cluster;

    // print progress messages for long running tasks
    if (msgtp && wkf_msg_timer_timeout(msgtp)) {
      char tmpbuf[1024];
      sprintf(tmpbuf, "cluster %d of %d: (%6.2f%% complete). %d frames of %d left.", 
              n+1, numcluster, 100.0f*(n+1)/((float) numcluster), remframes, numframes);
      msgInfo << "measure cluster: " << tmpbuf << sendmsg;
    }
  }

  // combine unclustered frames to form the last cluster
  int *unclustered = new int[numframes];
  int numunclustered = 0;
  for (n = 0; n < numframes; ++n) {
    if (skipList[n] == 0) {
      unclustered[numunclustered] = framesList[n];
      ++numunclustered;
    }
  }
  // NOTE: both lists have been allocated
  // to the length of numcluster+1
  clusterlist[numcluster] = unclustered;
  clustersize[numcluster] = numunclustered;

  // Cleanup
  delete[] newSkipList;
  delete[] skipList;
  wkf_msg_timer_destroy(msgtp);

  return MEASURE_NOERR;
}


/**************************************************************************/

typedef ResizeArray<int> intlist;

// helper function for cluster size analysis
// build index list for the next cluster.
static void assemble_cluster(intlist &cluster_list, intlist &candidate_list,
               intlist **neighbor_grid, int atom, int numshared, int *idxmap) {
  int idx, nn, i,j;
  
  // clear lists and add initial atom pairs to candidates list.
  candidate_list.clear();
  cluster_list.clear();

  idx = idxmap[atom];
  nn  = neighbor_grid[idx]->num();
  for (i = 0; i < nn; i++) {
    int bn = (*neighbor_grid[idx])[i];
    if (neighbor_grid[idxmap[bn]]) {
      candidate_list.append(atom);
      candidate_list.append(bn);
    }
  }

  // pointer to the currently processed cluster candidate list entry.
  int curidx=0;
  
  while (curidx < candidate_list.num()) {

    // a pair of neighbors has to share at least numshared
    // neighbors to be added to the cluster.
    // at least numshared neighbors.
    int count = 0;

    if (numshared > 0) {
      int ida = idxmap[candidate_list[curidx]];
      int idb = idxmap[candidate_list[curidx+1]];
      int nna  = neighbor_grid[ida]->num();
      int nnb  = neighbor_grid[idb]->num();

      for (i = 0; i < nna; i++) {
        if (neighbor_grid[ida]) {
          for (j = 0; j < nnb; j++) {
            if (neighbor_grid[idb]) {
              if ( (*neighbor_grid[ida])[i] == (*neighbor_grid[idb])[j] ) {
                ++count;
                if (count == numshared) 
                  goto exit;
              }
            }
          }
        }
      }
    }
    exit:
    
    if (count == numshared) {

      // add central atom of group of neighbors.
      // its neighbors had already been added to
      // the candidate list.
      int atma = candidate_list[curidx];
      if (cluster_list.find(atma) < 0) 
        cluster_list.append(atma);

      // add neighbor of central atom to cluster and
      // add neighbors of this atom to candidate list, 
      // if they are not already in it.
      int atmb = candidate_list[curidx+1];
      idx = idxmap[atmb];

      if (cluster_list.find(atmb) < 0) {
        cluster_list.append(atmb);
        
        int nnb = neighbor_grid[idx]->num();
        for (i = 0; i < nnb; i++) {
          int bn = (*neighbor_grid[idx])[i];
          if ((neighbor_grid[idxmap[bn]]) && (cluster_list.find(bn) < 0)) {
            candidate_list.append(atmb);
            candidate_list.append(bn);
          }
        }
      }
    }

    ++curidx;++curidx; // next candidate pair
  }
  
  return;
}

// perform cluster size analysis
int measure_clustsize(const AtomSel *sel, MoleculeList *mlist,
                      const double cutoff, int *clustersize,
                      int *clusternum, int *clusteridx, 
                      int minsize, int numshared, int usepbc) {
  int i,j;

  const float *framepos  = sel->coordinates(mlist);
  const int num_selected = sel->selected;
  const int num_atoms    = sel->num_atoms;
  const int *selected    = sel->on;

  // forward and reverse index maps for relative position
  // of atoms in the selection and in the global arrays.
  int *idxmap      = new int[num_atoms];
  int *idxrev      = new int[num_selected];

  for (i=0; i<sel->firstsel; i++) 
    idxmap[i]=-1;

  for(j=0,i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      idxrev[j]=i;
      idxmap[i]=j++;
    } else {
      idxmap[i]=-1;
    }
  }

  for (i=sel->lastsel+1; i<sel->num_atoms; i++) 
    idxmap[i]=-1;

  // allocate list of neighbor lists.
  intlist **neighbor_grid;
  neighbor_grid = new intlist *[num_selected];
  for (i = 0; i < num_selected; i++)
    neighbor_grid[i] = new intlist;

  // compile list of pairs for selection.
  GridSearchPair *pairlist, *currentPair, *nextPair;
  pairlist = vmd_gridsearch1(framepos, num_atoms, selected, (float)cutoff, 0, -1);

  // populate the neighborlist grid.
  for (currentPair = pairlist; currentPair != NULL; currentPair = nextPair) {
    neighbor_grid[idxmap[currentPair->ind1]]->append(currentPair->ind2);
    neighbor_grid[idxmap[currentPair->ind2]]->append(currentPair->ind1);
    nextPair = currentPair->next;
    free(currentPair);
  }

  // collect the cluster size information.
  int currentClusterNum = 0;
  int currentPosition = 0;
  intlist cluster_list(64);
  intlist candidate_list(128);

  while (currentPosition < num_selected) {
    if (neighbor_grid[currentPosition]) {
      // pick next atom that has not been processed yet and build list of 
      // all indices for this cluster. by looping over the neighbors of 
      // the first atom and adding all pairs of neighbors that share at
      // least numshared neighbors. continue with neighbors of neighbors 
      // accordingly until no more new unique neighbors are found.
      // entries of atoms added to a cluster are removed from neighbor_grid
      if (neighbor_grid[currentPosition]->num() > numshared) {
        assemble_cluster(cluster_list, candidate_list, neighbor_grid, 
                         idxrev[currentPosition], numshared, idxmap);

        if (minsize <= cluster_list.num()) {
          // these atoms have been processed. remove from global list
          for (i = 0; i < cluster_list.num(); i++) {
            int idx = idxmap[cluster_list[i]];
            delete neighbor_grid[idx];
            neighbor_grid[idx] = 0;
          }
      
          // store the cluster size, cluster index and atom index information in 
          // the designated arrays.
          for (i = 0; i < cluster_list.num(); i++) {
            int atom = idxmap[cluster_list[i]];
            clusteridx[atom] = cluster_list[i];
            clusternum[atom] = currentClusterNum;
            clustersize[atom] = cluster_list.num();
          }
          currentClusterNum++;
        }
      }
    }
    ++currentPosition;
  }

  for(i=0; i < num_selected; ++i) {
    if (neighbor_grid[i])
      delete neighbor_grid[i];
  }
  delete[] neighbor_grid;

  return MEASURE_NOERR;
}
