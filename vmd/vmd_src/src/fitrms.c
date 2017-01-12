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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>


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
void MatrixFitRMS(int n, float *v1, float *v2, const float *wt, float *ttt)
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
  double sumwt, tol, sig, gam;
  double sg, bb, cc, tmp;
  int a, b, c, maxiter, iters, /* ix, */ iy, iz;
  double t1[3],t2[3];
  double aatmp[9];
  char *TOL;

  /* Initialize arrays. */

  for(a=0;a<3;a++) {
	for(b=0;b<3;b++) {
	  m[a][b] = 0.0F;
	  aa[a][b] = 0.0F;
    aatmp[3*a+b] = 0;
	}
	m[a][a] = 1.0F;
	t1[a]=0.0F;
	t2[a]=0.0F;
  }

  sumwt = 0.0F;

  /* RMS fit tolerance */
  TOL = getenv( "VMDFITRMSTOLERANCE" );
  if (TOL) {
    tol = atof(TOL);
  } else {
    tol = 1e-15;
  }

  /* maximum number of fitting iterations */
  maxiter = 1000;

  /* Calculate center-of-mass vectors */

  vv1=v1;
  vv2=v2;

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
  for (a=0; a<3; a++) for (b=0; b<3; b++) aa[a][b] = aatmp[3*a+b];

  if(n>1) {
    /* Primary iteration scheme to determine rotation matrix for molecule 2 */
    iters = 0;
    while(1) {
      /* IX, IY, and IZ rotate 1-2-3, 2-3-1, 3-1-2, etc.*/
      iz = (iters+1) % 3;
      iy = (iz+1) % 3;
/*      ix = (iy+1) % 3; */
      sig = aa[iz][iy] - aa[iy][iz];
      gam = aa[iy][iy] + aa[iz][iz];

      if(iters>=maxiter) 
        {
            fprintf(stderr,
            " Matrix: Warning: no convergence (%1.8f<%1.8f after %d iterations).\n",(float)tol,(float)gam,iters);
          break;
        }

      /* Determine size of off-diagonal element.  If off-diagonals exceed the
         diagonal elements * tolerance, perform Jacobi rotation. */
      tmp = sig*sig + gam*gam;
      sg = sqrt(tmp);
      if((sg!=0.0F) &&(fabs(sig)>(tol*fabs(gam)))) {
        sg = 1.0F / sg;
        for(a=0;a<3;a++)
          {
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
  vv1=v1;
  vv2=v2;

  normalize3d(m[0]);
  normalize3d(m[1]);
  normalize3d(m[2]);

  ttt[0]=(float)m[0][0];
  ttt[1]=(float)m[0][1];
  ttt[2]=(float)m[0][2];
  ttt[3]=(float)-t1[0];
  ttt[4]=(float)m[1][0];
  ttt[5]=(float)m[1][1];
  ttt[6]=(float)m[1][2];
  ttt[7]=(float)-t1[1];
  ttt[8]=(float)m[2][0];
  ttt[9]=(float)m[2][1];
  ttt[10]=(float)m[2][2];
  ttt[11]=(float)-t1[2];
  ttt[12]=(float)t2[0];
  ttt[13]=(float)t2[1];
  ttt[14]=(float)t2[2];
  ttt[15]=1.0F; /* for compatibility with normal 4x4 matrices */
}

#ifdef __cplusplus
}
#endif

