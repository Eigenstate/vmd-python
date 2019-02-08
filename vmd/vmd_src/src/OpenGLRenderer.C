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
 *	$RCSfile: OpenGLRenderer.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.467 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Subclass of DisplayDevice, this object has routines used by all the
 * different display devices that use OpenGL for rendering.
 * Will render drawing commands into a window.
 * This is not the complete definition,
 * however, of a DisplayDevice; something must provide routines to open
 * windows, reshape, clear, set perspective, etc.  This object contains the
 * code to render a display command list.
 ***************************************************************************/

#include "OpenGLRenderer.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"
#include "VMDDisplayList.h"
#include "Hershey.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "OpenGLStipples.h"

// enable WireGL support
#define VMDWIREGL 1

// enable Intel SWR support
#define VMDINTELSWR 1

#if defined(VMDUSEOPENGLSHADER)
#define VMDUSEGLSLSPHERES 1
#if defined(GL_ARB_point_sprite)
#define VMDUSEGLSLSPHERESPRITES 1
#endif
#endif

#if 0
#define OGLERR { GLenum err; if ((err = glGetError()) != GL_NO_ERROR) {  \
	msgErr << __FILE__ << " line " << __LINE__ << " " << \
	(const char *) gluErrorString(err) << sendmsg; }}
#else 
#define OGLERR
#endif

#define MIN_SPHERE_RES 4
#define MAX_SPHERE_RES 30

#if defined(VMDUSELIBGLU) 
#define vmd_Project   gluProject
#define vmd_UnProject gluUnProject
#else
//
// VMD-internal replacements for GLU routines
//
static void vmd_matmult_4x4d(GLdouble *r, const GLdouble *m1, 
                             const GLdouble *m2) {
  r[ 0]=m1[0]*m2[ 0] + m1[4]*m2[ 1] + m1[ 8]*m2[ 2] + m1[12]*m2[ 3];
  r[ 4]=m1[0]*m2[ 4] + m1[4]*m2[ 5] + m1[ 8]*m2[ 6] + m1[12]*m2[ 7];
  r[ 8]=m1[0]*m2[ 8] + m1[4]*m2[ 9] + m1[ 8]*m2[10] + m1[12]*m2[11];
  r[12]=m1[0]*m2[12] + m1[4]*m2[13] + m1[ 8]*m2[14] + m1[12]*m2[15];

  r[ 1]=m1[1]*m2[ 0] + m1[5]*m2[ 1] + m1[ 9]*m2[ 2] + m1[13]*m2[ 3];
  r[ 5]=m1[1]*m2[ 4] + m1[5]*m2[ 5] + m1[ 9]*m2[ 6] + m1[13]*m2[ 7];
  r[ 9]=m1[1]*m2[ 8] + m1[5]*m2[ 9] + m1[ 9]*m2[10] + m1[13]*m2[11];
  r[13]=m1[1]*m2[12] + m1[5]*m2[13] + m1[ 9]*m2[14] + m1[13]*m2[15];

  r[ 2]=m1[2]*m2[ 0] + m1[6]*m2[ 1] + m1[10]*m2[ 2] + m1[14]*m2[ 3];
  r[ 6]=m1[2]*m2[ 4] + m1[6]*m2[ 5] + m1[10]*m2[ 6] + m1[14]*m2[ 7];
  r[10]=m1[2]*m2[ 8] + m1[6]*m2[ 9] + m1[10]*m2[10] + m1[14]*m2[11];
  r[14]=m1[2]*m2[12] + m1[6]*m2[13] + m1[10]*m2[14] + m1[14]*m2[15];

  r[ 3]=m1[3]*m2[ 0] + m1[7]*m2[ 1] + m1[11]*m2[ 2] + m1[15]*m2[ 3];
  r[ 7]=m1[3]*m2[ 4] + m1[7]*m2[ 5] + m1[11]*m2[ 6] + m1[15]*m2[ 7];
  r[11]=m1[3]*m2[ 8] + m1[7]*m2[ 9] + m1[11]*m2[10] + m1[15]*m2[11];
  r[15]=m1[3]*m2[12] + m1[7]*m2[13] + m1[11]*m2[14] + m1[15]*m2[15];
}


static void vmd_matmultvec_4x4d(GLdouble *npoint, const GLdouble *opoint, 
                               const GLdouble *mat) {
  npoint[0]=opoint[0]*mat[0]+opoint[1]*mat[4]+opoint[2]*mat[8]+opoint[3]*mat[12];
  npoint[1]=opoint[0]*mat[1]+opoint[1]*mat[5]+opoint[2]*mat[9]+opoint[3]*mat[13];
  npoint[2]=opoint[0]*mat[2]+opoint[1]*mat[6]+opoint[2]*mat[10]+opoint[3]*mat[14];
  npoint[3]=opoint[0]*mat[3]+opoint[1]*mat[7]+opoint[2]*mat[11]+opoint[3]*mat[15];
}

#define SWAP_ROWS_DOUBLE(a, b) { double *_tmp = a; (a)=(b); (b)=_tmp; }
#define SWAP_ROWS_FLOAT(a, b)  { float *_tmp = a; (a)=(b); (b)=_tmp; }
#define SWAP_ROWS SWAP_ROWS_DOUBLE
#define MAT(m,r,c) (m)[(c)*4+(r)]

 
static int vmd_invert_mat_4x4d(const GLdouble *m, GLdouble *out) {
  double wtmp[4][8];
  double m0, m1, m2, m3, s;
  double *r0, *r1, *r2, *r3;

  r0 = wtmp[0], r1 = wtmp[1], r2 = wtmp[2], r3 = wtmp[3];

  r0[0] = MAT(m, 0, 0), r0[1] = MAT(m, 0, 1),
  r0[2] = MAT(m, 0, 2), r0[3] = MAT(m, 0, 3),
  r0[4] = 1.0, r0[5] = r0[6] = r0[7] = 0.0,
  r1[0] = MAT(m, 1, 0), r1[1] = MAT(m, 1, 1),
  r1[2] = MAT(m, 1, 2), r1[3] = MAT(m, 1, 3),
  r1[5] = 1.0, r1[4] = r1[6] = r1[7] = 0.0,
  r2[0] = MAT(m, 2, 0), r2[1] = MAT(m, 2, 1),
  r2[2] = MAT(m, 2, 2), r2[3] = MAT(m, 2, 3),
  r2[6] = 1.0, r2[4] = r2[5] = r2[7] = 0.0,
  r3[0] = MAT(m, 3, 0), r3[1] = MAT(m, 3, 1),
  r3[2] = MAT(m, 3, 2), r3[3] = MAT(m, 3, 3),
  r3[7] = 1.0, r3[4] = r3[5] = r3[6] = 0.0;

  /* choose pivot - or die */
  if (fabs(r3[0]) > fabs(r2[0]))
    SWAP_ROWS(r3, r2);
  if (fabs(r2[0]) > fabs(r1[0]))
    SWAP_ROWS(r2, r1);
  if (fabs(r1[0]) > fabs(r0[0]))
    SWAP_ROWS(r1, r0);
  if (0.0 == r0[0])
    return 0;

  /* eliminate first variable     */
  m1 = r1[0] / r0[0];
  m2 = r2[0] / r0[0];
  m3 = r3[0] / r0[0];
  s = r0[1];
  r1[1] -= m1 * s;
  r2[1] -= m2 * s;
  r3[1] -= m3 * s;
  s = r0[2];
  r1[2] -= m1 * s;
  r2[2] -= m2 * s;
  r3[2] -= m3 * s;
  s = r0[3];
  r1[3] -= m1 * s;
  r2[3] -= m2 * s;
  r3[3] -= m3 * s;
  s = r0[4];
  if (s != 0.0) {
    r1[4] -= m1 * s;
    r2[4] -= m2 * s;
    r3[4] -= m3 * s;
  }
  s = r0[5];
  if (s != 0.0) {
    r1[5] -= m1 * s;
    r2[5] -= m2 * s;
    r3[5] -= m3 * s;
  }
  s = r0[6];
  if (s != 0.0) {
    r1[6] -= m1 * s;
    r2[6] -= m2 * s;
    r3[6] -= m3 * s;
  }
  s = r0[7];
  if (s != 0.0) {
    r1[7] -= m1 * s;
    r2[7] -= m2 * s;
    r3[7] -= m3 * s;
  }

  /* choose pivot - or die */
  if (fabs(r3[1]) > fabs(r2[1]))
    SWAP_ROWS(r3, r2);
  if (fabs(r2[1]) > fabs(r1[1]))
    SWAP_ROWS(r2, r1);
  if (0.0 == r1[1])
    return 0;

  /* eliminate second variable */
  m2 = r2[1] / r1[1];
  m3 = r3[1] / r1[1];
  r2[2] -= m2 * r1[2];
  r3[2] -= m3 * r1[2];
  r2[3] -= m2 * r1[3];
  r3[3] -= m3 * r1[3];
  s = r1[4];
  if (0.0 != s) {
    r2[4] -= m2 * s;
    r3[4] -= m3 * s;
  }
  s = r1[5];
  if (0.0 != s) {
    r2[5] -= m2 * s;
    r3[5] -= m3 * s;
  }
  s = r1[6];
  if (0.0 != s) {
    r2[6] -= m2 * s;
    r3[6] -= m3 * s;
  }
  s = r1[7];
  if (0.0 != s) {
    r2[7] -= m2 * s;
    r3[7] -= m3 * s;
  }

  /* choose pivot - or die */
  if (fabs(r3[2]) > fabs(r2[2]))
    SWAP_ROWS(r3, r2);
  if (0.0 == r2[2])
    return 0;

  /* eliminate third variable */
  m3 = r3[2] / r2[2];
  r3[3] -= m3 * r2[3], r3[4] -= m3 * r2[4],
  r3[5] -= m3 * r2[5], r3[6] -= m3 * r2[6], r3[7] -= m3 * r2[7];

  /* last check */
  if (0.0 == r3[3])
    return 0;

  s = 1.0 / r3[3];		/* now back substitute row 3 */
  r3[4] *= s;
  r3[5] *= s;
  r3[6] *= s;
  r3[7] *= s;

  m2 = r2[3];			/* now back substitute row 2 */
  s = 1.0 / r2[2];
  r2[4] = s * (r2[4] - r3[4] * m2), r2[5] = s * (r2[5] - r3[5] * m2),
  r2[6] = s * (r2[6] - r3[6] * m2), r2[7] = s * (r2[7] - r3[7] * m2);

  m1 = r1[3];
  r1[4] -= r3[4] * m1, r1[5] -= r3[5] * m1,
  r1[6] -= r3[6] * m1, r1[7] -= r3[7] * m1;

  m0 = r0[3];
  r0[4] -= r3[4] * m0, r0[5] -= r3[5] * m0,
  r0[6] -= r3[6] * m0, r0[7] -= r3[7] * m0;

  m1 = r1[2];			/* now back substitute row 1 */
  s = 1.0 / r1[1];
  r1[4] = s * (r1[4] - r2[4] * m1), r1[5] = s * (r1[5] - r2[5] * m1),
  r1[6] = s * (r1[6] - r2[6] * m1), r1[7] = s * (r1[7] - r2[7] * m1);
  m0 = r0[2];
  r0[4] -= r2[4] * m0, r0[5] -= r2[5] * m0,
  r0[6] -= r2[6] * m0, r0[7] -= r2[7] * m0;

  m0 = r0[1];			/* now back substitute row 0 */
  s = 1.0 / r0[0];
  r0[4] = s * (r0[4] - r1[4] * m0), r0[5] = s * (r0[5] - r1[5] * m0),
  r0[6] = s * (r0[6] - r1[6] * m0), r0[7] = s * (r0[7] - r1[7] * m0);

  MAT(out, 0, 0) = r0[4];
  MAT(out, 0, 1) = r0[5], MAT(out, 0, 2) = r0[6];
  MAT(out, 0, 3) = r0[7], MAT(out, 1, 0) = r1[4];
  MAT(out, 1, 1) = r1[5], MAT(out, 1, 2) = r1[6];
  MAT(out, 1, 3) = r1[7], MAT(out, 2, 0) = r2[4];
  MAT(out, 2, 1) = r2[5], MAT(out, 2, 2) = r2[6];
  MAT(out, 2, 3) = r2[7], MAT(out, 3, 0) = r3[4];
  MAT(out, 3, 1) = r3[5], MAT(out, 3, 2) = r3[6];
  MAT(out, 3, 3) = r3[7];

  return 1;
}


static GLdouble * vmd_vec_normalize_3d(GLdouble *vect) {
  GLdouble len2 = vect[0]*vect[0] + vect[1]*vect[1] + vect[2]*vect[2];

  // prevent division by zero
  if (len2 > 0) {
    GLdouble rescale = 1.0 / sqrt(len2);
    vect[0] *= rescale;
    vect[1] *= rescale;
    vect[2] *= rescale;
  }

  return vect;
}


static GLdouble * vmd_cross_3d(GLdouble *x1, const GLdouble *x2, 
                               const GLdouble *x3) {
  x1[0] =  x2[1]*x3[2] - x3[1]*x2[2];
  x1[1] = -x2[0]*x3[2] + x3[0]*x2[2];
  x1[2] =  x2[0]*x3[1] - x3[0]*x2[1];

  return x1;
}


static void vmd_mattrans_d(GLdouble *m, GLdouble x, GLdouble y, GLdouble z) {
  m[12] = m[0]*x + m[4]*y + m[ 8]*z + m[12];
  m[13] = m[1]*x + m[5]*y + m[ 9]*z + m[13];
  m[14] = m[2]*x + m[6]*y + m[10]*z + m[14];
  m[15] = m[3]*x + m[7]*y + m[11]*z + m[15];
}


static void vmd_mat_identity_4x4d(GLdouble *m) {
  memset((void *)m, 0, 16*sizeof(GLdouble));
  m[0]=1.0;
  m[5]=1.0;
  m[10]=1.0;
  m[15]=1.0;
}


#define SPHEREMAXRES 30
static void vmd_DrawSphere(float rad, int res, int solid) {
  int i, j;
  float zLo, zHi, radLo, radHi, stn, ctn;

  float sinLong[SPHEREMAXRES];
  float cosLong[SPHEREMAXRES];
  float sinLatVert[SPHEREMAXRES];
  float cosLatVert[SPHEREMAXRES];
  float sinLatNorm[SPHEREMAXRES];
  float cosLatNorm[SPHEREMAXRES];

  if (res < 2)
    res = 2;

  if (res >= SPHEREMAXRES)
    res = SPHEREMAXRES-1;

  // longitudinal "slices"
  float ang_twopi_res = VMD_TWOPI / res;
  for (i=0; i<res; i++) {
    float angle = i * ang_twopi_res;
    sinLong[i] = sinf(angle);
    cosLong[i] = cosf(angle);
  }
  // ensure that longitude end point exactly matches start
  sinLong[res] = 0.0f; // sinLong[0]
  cosLong[res] = 1.0f; // cosLong[0]

  // latitude "stacks"
  float ang_pi_res = VMD_PI / res;
  for (i=0; i<=res; i++) {
    float angle = i * ang_pi_res;
    sinLatNorm[i] = sinf(angle);
    cosLatNorm[i] = cosf(angle);
    sinLatVert[i] = rad * sinLatNorm[i];
    cosLatVert[i] = rad * cosLatNorm[i];
  }
  // ensure top and bottom poles come to points
  sinLatVert[0] = 0;
  sinLatVert[res] = 0;

  // draw sphere caps as triangle fans, lower cap, j==0
  if (solid) {
    radLo = sinLatVert[1];
    zLo   = cosLatVert[1];
    stn   = sinLatNorm[1];
    ctn   = cosLatNorm[1];

    glNormal3f(sinLong[0] * sinLatNorm[0], 
               cosLong[0] * sinLatNorm[0], 
               cosLatNorm[0]);

    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0.0, 0.0, rad);
    for (i=res; i>=0; i--) {
      glNormal3f(sinLong[i] * stn, cosLong[i] * stn, ctn);
      glVertex3f(radLo * sinLong[i], radLo * cosLong[i], zLo);
    }
    glEnd();
  }

  // draw sphere caps as triangle fans, high cap, j==(res-1)
  radHi = sinLatVert[res-1];
  zHi   = cosLatVert[res-1];
  stn   = sinLatNorm[res-1];
  ctn   = cosLatNorm[res-1];

  if (solid) {
    glNormal3f(sinLong[res] * sinLatNorm[res], 
               cosLong[res] * sinLatNorm[res], 
               cosLatNorm[res]);

    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0.0, 0.0, -rad);
    for (i=0; i<=res; i++) {
      glNormal3f(sinLong[i] * stn, cosLong[i] * stn, ctn);
      glVertex3f(radHi * sinLong[i], radHi * cosLong[i], zHi);
    }
    glEnd();
  } else {
    glBegin(GL_POINTS);
    glVertex3f(0.0, 0.0,  rad); // draw both apex and base points at once
    glVertex3f(0.0, 0.0, -rad);
    for (i=0; i<=res; i++)
      glVertex3f(radHi * sinLong[i], radHi * cosLong[i], zHi);
    glEnd();
  }
  for (j=1; j<res-1; j++) {
    zLo = cosLatVert[j];
    zHi = cosLatVert[j+1];

    float stv1 = sinLatVert[j];
    float stv2 = sinLatVert[j+1];

    float stn1 = sinLatNorm[j];
    float ctn1 = cosLatNorm[j];
    float stn2 = sinLatNorm[j+1];
    float ctn2 = cosLatNorm[j+1];

    if (solid) {
      glBegin(GL_QUAD_STRIP);
      for (i=0; i<=res; i++) {
        glNormal3f(sinLong[i] * stn2, cosLong[i] * stn2, ctn2);
        glVertex3f(stv2 * sinLong[i], stv2 * cosLong[i], zHi);
        glNormal3f(sinLong[i] * stn1, cosLong[i] * stn1, ctn1);
        glVertex3f(stv1 * sinLong[i], stv1 * cosLong[i], zLo);
      }
      glEnd();
    } else {
      glBegin(GL_POINTS);
      for (i=0; i<=res; i++)
        glVertex3f(stv1 * sinLong[i], stv1 * cosLong[i], zLo);
      glEnd();
    }
  }
}


// Routine to draw a truncated cone with caps, adapted from the fallback
// triangulated code path in FileRenderer::cone_trunc()
static void vmd_DrawConic(float *base, float *apex, float radius, float radius2, int numsides) {
  int h;
  float theta, incTheta, cosTheta, sinTheta;
  float axis[3], temp[3], perp[3], perp2[3];
  float vert0[3], vert1[3], vert2[3], edge0[3], edge1[3], face0[3], face1[3], norm0[3], norm1[3];

  axis[0] = base[0] - apex[0];
  axis[1] = base[1] - apex[1];
  axis[2] = base[2] - apex[2];
  vec_normalize(axis);

  // Find an arbitrary vector that is not the axis and has non-zero length
  temp[0] = axis[0] - 1.0f;
  temp[1] = 1.0f;
  temp[2] = 1.0f;

  // use the cross product to find orthogonal vectors
  cross_prod(perp, axis, temp);
  vec_normalize(perp);
  cross_prod(perp2, axis, perp); // shouldn't need normalization

  // Draw the triangles
  incTheta = (float) VMD_TWOPI / numsides;
  theta = 0.0;

  // if radius2 is larger than zero, we will draw quadrilateral
  // panels rather than triangular panels
  if (radius2 > 0) {
    float negaxis[3], offsetL[3], offsetT[3], vert3[3];
    int filled=1;
    vec_negate(negaxis, axis);
    memset(vert0, 0, sizeof(vert0));
    memset(vert1, 0, sizeof(vert1));
    memset(norm0, 0, sizeof(norm0));

    glBegin(GL_TRIANGLES);
    for (h=0; h <= numsides+3; h++) {
      // project 2-D unit circles onto perp/perp2 3-D basis vectors
      // and scale to desired radii
      cosTheta = (float) cosf(theta);
      sinTheta = (float) sinf(theta);
      offsetL[0] = radius2 * (cosTheta*perp[0] + sinTheta*perp2[0]);
      offsetL[1] = radius2 * (cosTheta*perp[1] + sinTheta*perp2[1]);
      offsetL[2] = radius2 * (cosTheta*perp[2] + sinTheta*perp2[2]);
      offsetT[0] = radius  * (cosTheta*perp[0] + sinTheta*perp2[0]);
      offsetT[1] = radius  * (cosTheta*perp[1] + sinTheta*perp2[1]);
      offsetT[2] = radius  * (cosTheta*perp[2] + sinTheta*perp2[2]);

      // copy old vertices
      vec_copy(vert2, vert0);
      vec_copy(vert3, vert1);
      vec_copy(norm1, norm0);

      // calculate new vertices
      vec_add(vert0, base, offsetT);
      vec_add(vert1, apex, offsetL);

      // Use the new vertex to find new edges
      edge0[0] = vert0[0] - vert1[0];
      edge0[1] = vert0[1] - vert1[1];
      edge0[2] = vert0[2] - vert1[2];
      edge1[0] = vert0[0] - vert2[0];
      edge1[1] = vert0[1] - vert2[1];
      edge1[2] = vert0[2] - vert2[2];

      // Use the new edge to find a new facet normal
      cross_prod(norm0, edge1, edge0);
      vec_normalize(norm0);

      if (h > 2) {
        // Use the new normal to draw the previous side
        glNormal3fv(norm0);
        glVertex3fv(vert0);
        glNormal3fv(norm1);
        glVertex3fv(vert3);
        glNormal3fv(norm0);
        glVertex3fv(vert1);

        glNormal3fv(norm1);
        glVertex3fv(vert3);
        glNormal3fv(norm0);
        glVertex3fv(vert0);
        glNormal3fv(norm1);
        glVertex3fv(vert2);

        // Draw cylinder caps
        if (filled & CYLINDER_LEADINGCAP) {
          glNormal3fv(axis);
          glVertex3fv(vert1);
          glNormal3fv(axis);
          glVertex3fv(vert3);
          glNormal3fv(axis);
          glVertex3fv(apex);
        }
        if (filled & CYLINDER_TRAILINGCAP) {
          glNormal3fv(negaxis);
          glVertex3fv(vert0);
          glNormal3fv(negaxis);
          glVertex3fv(vert2);
          glNormal3fv(negaxis);
          glVertex3fv(base);
        }
      }

      theta += incTheta;
    }
    glEnd();
  } else {
    // radius2 is zero, so we draw triangular panels joined at the apex
    glBegin(GL_TRIANGLES);
    for (h=0; h < numsides+3; h++) {
      // project 2-D unit circle onto perp/perp2 3-D basis vectors
      // and scale to desired radius
      cosTheta = (float) cosf(theta);
      sinTheta = (float) sinf(theta);
      vert0[0] = base[0] + radius * (cosTheta*perp[0] + sinTheta*perp2[0]);
      vert0[1] = base[1] + radius * (cosTheta*perp[1] + sinTheta*perp2[1]);
      vert0[2] = base[2] + radius * (cosTheta*perp[2] + sinTheta*perp2[2]);

      // Use the new vertex to find a new edge
      edge0[0] = vert0[0] - apex[0];
      edge0[1] = vert0[1] - apex[1];
      edge0[2] = vert0[2] - apex[2];

      if (h > 0) {
        // Use the new edge to find a new face
        cross_prod(face0, edge0, edge1);
        vec_normalize(face0);

        if (h > 1) {
          // Use the new face to find the normal of the previous triangle
          norm0[0] = (face1[0] + face0[0]) * 0.5f;
          norm0[1] = (face1[1] + face0[1]) * 0.5f;
          norm0[2] = (face1[2] + face0[2]) * 0.5f;
          vec_normalize(norm0);

          if (h > 2) {
            // Use the new normal to draw the previous side and base of the cone
            glNormal3fv(norm0);
            glVertex3fv(vert1);
            glNormal3fv(norm1);
            glVertex3fv(vert2);
            glNormal3fv(face1);
            glVertex3fv(apex);

            glNormal3fv(axis);
            glVertex3fv(vert2);
            glNormal3fv(axis);
            glVertex3fv(vert1);
            glNormal3fv(axis);
            glVertex3fv(base);
          }
        }

        // Copy the old values
        memcpy(norm1, norm0, 3*sizeof(float));
        memcpy(vert2, vert1, 3*sizeof(float));
        memcpy(face1, face0, 3*sizeof(float));
      }
      memcpy(vert1, vert0, 3*sizeof(float));
      memcpy(edge1, edge0, 3*sizeof(float));

      theta += incTheta;
    }
    glEnd();
  }
}


static GLint vmd_Project(GLdouble objX,
                         GLdouble objY,
                         GLdouble objZ,
                         const GLdouble *model,
                         const GLdouble *proj,
                         const GLint *view,
                         GLdouble *winX,
                         GLdouble *winY,
                         GLdouble *winZ) {
#if !defined(VMDUSELIBGLU) 
  // replaced previous implementation with one that also works correctly
  // for orthographic projections
  double in[4], tmp[4], out[4];

  in[0]=objX;
  in[1]=objY;
  in[2]=objZ;
  in[3]=1.0;

  vmd_matmultvec_4x4d(tmp,  in, model);
  vmd_matmultvec_4x4d(out, tmp, proj);

  if (out[3] == 0.0) 
    return 0;

  // efficiently map coordinates to range 0-1, and then to the viewport
  double tinv = 0.5 / out[3];
  *winX = (out[0] * tinv + 0.5) * view[2] + view[0];
  *winY = (out[1] * tinv + 0.5) * view[3] + view[1]; 
  *winZ = out[2] * tinv + 0.5;

  return 1;
#else
  return gluProject(objX, objY, objZ, model, proj, view, winX, winY, winZ);
#endif
}


static GLint vmd_UnProject(GLdouble winX,
                           GLdouble winY,
                           GLdouble winZ,
                           const GLdouble *model,
                           const GLdouble *proj,
                           const GLint *view,
                           GLdouble *objX,
                           GLdouble *objY,
                           GLdouble *objZ) {
#if !defined(VMDUSELIBGLU) 
  // based on opengl.org wiki sample
  GLdouble m[16], A[16], in[4], out[4];
  memset(m, 0, sizeof(m));

  // invert matrix, compute projection * modelview, store in A
  vmd_matmult_4x4d(A, proj, model);
  if (vmd_invert_mat_4x4d(A, m) == 0)
    return 0;

  in[0]=((winX-(double)view[0])/(double)view[2])*2.0 - 1.0;
  in[1]=((winY-(double)view[1])/(double)view[3])*2.0 - 1.0;
  in[2]=winZ*2.0 - 1.0;
  in[3]=1.0;

  vmd_matmultvec_4x4d(out, in, m);
  if (out[3]==0.0)
    return 0;

  out[3]=1.0/out[3];
  *objX=out[0]*out[3];
  *objY=out[1]*out[3];
  *objZ=out[2]*out[3];

  return 1;  
#else
  return gluUnProject(winX, winY, winZ, model, proj, view, objX, objY, objZ);
#endif
}
#endif


static void vmd_LookAt(GLdouble eyeX,
                       GLdouble eyeY,
                       GLdouble eyeZ,
                       GLdouble centerX,
                       GLdouble centerY,
                       GLdouble centerZ,
                       GLdouble upX,
                       GLdouble upY,
                       GLdouble upZ) {
#if !defined(VMDUSELIBGLU) 
  // initialize to identity matrix
  GLdouble matrix[16];
  vmd_mat_identity_4x4d(matrix);

  // now compute transform for look at point
  GLdouble f[3], s[3], u[3];
  GLdouble matrix2[16], resmat[16];

  f[0] = centerX - eyeX;
  f[1] = centerY - eyeY;
  f[2] = centerZ - eyeZ;
  vmd_vec_normalize_3d(f);

  // side = forward x up
  u[0] = upX; u[1] = upY; u[2] = upZ;
  vmd_cross_3d(s, f, u);
  vmd_vec_normalize_3d(s);

  // recompute orthogonal up dir: up = side x forward
  vmd_cross_3d(u, s, f);

  matrix2[ 0] = s[0];
  matrix2[ 4] = s[1];
  matrix2[ 8] = s[2];
  matrix2[12] = 0.0;

  matrix2[ 1] = u[0];
  matrix2[ 5] = u[1];
  matrix2[ 9] = u[2];
  matrix2[13] = 0.0;

  matrix2[ 2] = -f[0];
  matrix2[ 6] = -f[1];
  matrix2[10] = -f[2];
  matrix2[14] = 0.0;

  matrix2[3] = matrix2[7] = matrix2[11] = 0.0;
  matrix2[15] = 1.0;

  vmd_matmult_4x4d(resmat, matrix, matrix2);
  vmd_mattrans_d(resmat, -eyeX, -eyeY, -eyeZ);

  GLfloat tmpmat[16];
  for (int i=0; i<16; i++) 
    tmpmat[i]= (GLfloat)(resmat[i]);

  glLoadIdentity();
  glMultMatrixf(tmpmat);
#else
  glLoadIdentity();
  gluLookAt(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ);
#endif
}


#if defined(VMD_NANOHUB)
bool OpenGLRenderer::init_offscreen_framebuffer(int winWidth, int winHeight) {
    if (_finalColorTex != 0) {
        glDeleteTextures(1, &_finalColorTex);
    }
    if (_finalDepthRb != 0) {
        glDeleteRenderbuffersEXT(1, &_finalDepthRb);
    }
    if (_finalFbo != 0) {
        glDeleteFramebuffersEXT(1, &_finalFbo);
    }

    // Initialize a fbo for final display.
    glGenFramebuffersEXT(1, &_finalFbo);

    glGenTextures(1, &_finalColorTex);
    glBindTexture(GL_TEXTURE_2D, _finalColorTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, winWidth, winHeight, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _finalFbo);
    glGenRenderbuffersEXT(1, &_finalDepthRb);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, _finalDepthRb);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, 
                             winWidth, winHeight);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, _finalColorTex, 0);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                 GL_RENDERBUFFER_EXT, _finalDepthRb);

    GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
        msgWarn << "FBO Setup failed" << sendmsg;
        return false;
    }
    
    return true;
}
#endif


void OpenGLRenderer::setup_initial_opengl_state(void) {
  int i; 

  if (getenv("VMDSIMPLEGRAPHICS") == NULL) {
    simplegraphics = 0; // use all available OpenGL features and extensions
  } else {
    simplegraphics = 1; // limit VMD to OpenGL ~1.0 with no extensions
    msgWarn << "Simple graphics mode: OpenGL 1.0, no extensions" << sendmsg;
  }

  // Create quadric objects for spheres, cylinders, and disks
  // Default to drawing filled objects, may be changed by the detail level
#if defined(VMDUSELIBGLU)
  objQuadric = gluNewQuadric();
  pointsQuadric = gluNewQuadric();
  gluQuadricDrawStyle(objQuadric, (GLenum)GLU_FILL);
  gluQuadricDrawStyle(pointsQuadric, (GLenum)GLU_POINT);
#endif

  // display list caching
  ogl_cachelistbase = 2000;
  ogl_cacheenabled = 0;
  ogl_cachedebug = 0;
  if (getenv("VMDCACHEDLDEBUG") != NULL) {
    ogl_cachedebug = 1; // enable verbose output for display list caching
  }

  wiregl = 0;          // wiregl not detected yet
  intelswr = 0;        // Intel SWR not detected yet
  immersadeskflip = 0; // Immersadesk right-eye X-axis mode off by default
  shearstereo = 0;     // use stereo based on eye rotation by default

  // initialize state caching variables so they get overwritten on 
  // first reference in the call to render()
  oglmaterialindex = -1;
  oglopacity = -1.0f;
  oglambient = -1.0f;
  ogldiffuse = -1.0f;
  oglspecular = -1.0f;
  oglshininess = -1.0f;
  ogloutline = -1.0f;
  ogloutlinewidth = -1.0f;
  ogltransmode = -1;

  ogl_useblendedtrans = 0;
  ogl_transpass = 0; 
  ogl_useglslshader = 0;
  ogl_acrobat3dcapture = 0;
  ogl_lightingenabled = 0;
  ogl_rendstateserial = 1;    // force GLSL update on 1st pass
  ogl_glslserial = 0;         // force GLSL update on 1st pass
  ogl_glsltoggle = 1;         // force GLSL update on 1st pass
  ogl_glslmaterialindex = -1; // force GLSL update on 1st pass
  ogl_glslprojectionmode = DisplayDevice::PERSPECTIVE; 
  ogl_glsltexturemode = 0;    // initialize GLSL projection to off

  // identify the rendering hardware we're using
  ext->find_renderer();

  // find all available OpenGL extensions, unless the user doesn't want us to.
  if (!simplegraphics) {
    ext->find_extensions(); ///< register available OpenGL extensions
  }


#if 0
// XXX the performance workaround aspect of this must still be tested
// on Linux to verify that the existing code path is actually a benefit.

// not tested on other platforms yet
#if defined(__APPLE__)
  // Detect NVidia graphics cards, which have a semi-broken stereo 
  // implementation that favors rendering in stereo mode all the time, 
  // as the alternative is 20-50x performance hit on Linux.
  // XXX on MacOS X, the behavior is much more serious than just a performance
  // hit, they actually fail to draw/clear the back right color buffer 
  // when drawing to GL_BACK.
  if (ext->hasstereo && ext->oglrenderer == OpenGLExtensions::NVIDIA) {
    msgInfo << "nVidia card detected, enabling mono drawing performance workaround" << sendmsg;

    // force drawing in stereo even when VMD is set for mono mode
    ext->stereodrawforced = 1;
  }
#endif
#endif

// XXX recent ATI/AMD graphics drivers are greatly improved, so this safety
//     check is disabled for the time being...
#if 0 && defined(__linux)
  // Detect ATI Linux driver and disable unsafe extensions
  if (ext->oglrenderer == OpenGLExtensions::ATI) {
    if (getenv("VMDDISABLEATILINUXWORKAROUND") == NULL) {
      msgInfo << "ATI Linux driver detected, limiting features to avoid driver bugs." << sendmsg;
      msgInfo << "  Set the environment variable VMDDISABLEATILINUXWORKAROUND" << sendmsg;
      msgInfo << "  to enable full functionality on a known-safe driver version." << sendmsg;

      simplegraphics = 1; 
    }
  }
#endif

#if defined(VMDWIREGL)
  // Detect WireGL and shut off unsupported rendering features if so.
  if (ext->oglrenderer == OpenGLExtensions::WIREGL ||
      (getenv("VMDWIREGL") != NULL)) {
    msgInfo << "WireGL renderer detected, disabling unsupported OpenGL features." << sendmsg;
    wiregl=1;

    // Shut off unsupported rendering features if so.
    ext->hastex2d = 0;
    ext->hastex3d = 0;
  }
#endif

#if defined(VMDINTELSWR)
  // Detect Intel OpenSWR and shut off unsupported rendering features if so.
  if (ext->oglrenderer == OpenGLExtensions::INTELSWR) {
    msgInfo << "Intel OpenSWR renderer detected, disabling unsupported OpenGL features." << sendmsg;
    intelswr=1;

    // the current alpha version of SWR has lots of missing functionality
    simplegraphics = 1; 

    // Shut off unsupported rendering features if so.
    ext->hastex2d = 0;
    ext->hastex3d = 0;
  }
#endif

  glDepthFunc(GL_LEQUAL);
  glEnable(GL_DEPTH_TEST);           // use zbuffer for hidden-surface removal
  glClearDepth(1.0);
 
#if 1
  // VMD now renormalizes all the time since non-uniform scaling 
  // operations must be applied for drawing ellipsoids and other 
  // warped geometry, and this is a final cure for people remote
  // displaying on machines with broken rescale normal extensions
  glEnable(GL_NORMALIZE);            // automatically normalize normals
#else
  // Rescale normals, assumes they are initially normalized to begin with, 
  // and that only uniform scaling is applied, and non-warping modelview
  // matrices are used. (i.e. no shear in the modelview matrix...)
  // Gets rid of a square root on every normal, but still tracks
  // uniform scaling operations.  If not available, enable full
  // normalization.
  if (simplegraphics || wiregl || ogl_acrobat3dcapture) {
    // Renormalize normals if we're in 'simplegraphics' mode.
    // WireGL doesn't work correctly with the various normal rescaling
    // features and extensions, so we have to make it use GL_NORMALIZE.
    glEnable(GL_NORMALIZE);            // automatically normalize normals
  } else {
#if defined(_MSC_VER) || defined(__irix) || defined(__APPLE__)
    // XXX The Microsoft "GDI Generic", MacOS X, and IRIX OpenGL renderers
    // malfunction when we use GL_RESCALE_NORMAL, so we disable it here
    glEnable(GL_NORMALIZE);            // automatically normalize normals
#else
#if defined(GL_VERSION_1_2)
    ext->hasrescalenormalext = 1;
    glEnable(GL_RESCALE_NORMAL);       // automatically rescale normals
#elif defined(GL_RESCALE_NORMAL_EXT)
    if (ext->vmdQueryExtension("GL_RESCALE_NORMAL_EXT")) {
      ext->hasrescalenormalext = 1;
      glEnable(GL_RESCALE_NORMAL_EXT); // automatically rescale normals
    } else {
      glEnable(GL_NORMALIZE);          // automatically normalize normals
    }
#else
    glEnable(GL_NORMALIZE);            // automatically normalize normals
#endif
#endif
  }  
#endif

  // configure for dashed lines ... but initially show solid lines
  glLineStipple(1, 0x3333);
  glDisable(GL_LINE_STIPPLE);

  // configure the fogging characteristics ... color and position of fog
  // are set during the clear routine
  glFogi(GL_FOG_MODE, GL_EXP2);
  glFogf(GL_FOG_DENSITY, 0.40f);

  // configure the light model
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);

  glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
  glEnable(GL_COLOR_MATERIAL);       // have materials set by curr color
  glDisable(GL_POLYGON_SMOOTH);      // make sure not to antialias polygons

  // disable all lights by default
  for (i=0; i < DISP_LIGHTS; i++) {
    ogl_lightstate[i] = 0; // off by default
  }

  // disable all clipping planes by default
  for (i=0; i < VMD_MAX_CLIP_PLANE; i++) {
    ogl_clipmode[i] = 0; // off by default
    glDisable((GLenum) (GL_CLIP_PLANE0 + i));
  }

  // load transformation matrices on stack, initially with identity transforms
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // generate sphere display lists
  glMatrixMode(GL_MODELVIEW);
  for (i=MIN_SPHERE_RES; i<=MAX_SPHERE_RES; i++) {
    GLuint solidlist = glGenLists(1);
    glNewList(solidlist, GL_COMPILE);
#if defined(VMDUSELIBGLU)
    gluSphere(objQuadric, 1.0, i, i);
#else
    vmd_DrawSphere(1.0, i, 1);
#endif
    glEndList(); 
    solidSphereLists.append(solidlist);

    GLuint pointlist = glGenLists(1);
    glNewList(pointlist, GL_COMPILE);
#if defined(VMDUSELIBGLU)
    gluSphere(pointsQuadric, 1.0, i, i);
#else
    vmd_DrawSphere(1.0, i, 0);
#endif
    glEndList(); 
    pointSphereLists.append(pointlist);
  }

  // create font display lists for use in displaying text
  ogl_textMat.identity();

  // display list for 1-pixel wide non-antialiased font rendering,
  // which doesn't have points at each font stroke vector endpoint
  font1pxListBase = glGenLists(256);
  glListBase(font1pxListBase);
  for (i=0 ; i<256 ; i++) {
    glNewList(font1pxListBase+i, GL_COMPILE);
    hersheyDrawLetterOpenGL(i, 0); // draw vector lines only
    glEndList();
  }

  // display list for N-pixel wide antialiased font rendering,
  // which has added points at each font stroke vector endpoint
  // to prevent "cracks" from showing up with large font sizes
  fontNpxListBase = glGenLists(256);
  glListBase(fontNpxListBase);
  for (i=0 ; i<256 ; i++) {
    glNewList(fontNpxListBase+i, GL_COMPILE);
    hersheyDrawLetterOpenGL(i, 1); // draw with lines+points
    glEndList();
  }

  // display lists are now initialized, so they must be freed when this
  // OpenGL context is destroyed
  dpl_initialized = 1;

#if defined(GL_VERSION_1_1)
  if (!(simplegraphics || ogl_acrobat3dcapture)) {
    // enable vertex arrays.
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
  }
#endif


#if defined(GL_VERSION_1_1)
  if (ext->hastex2d) {
    int i, sz;
    GLint x, y;

    // test actual maximums for desired format
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max2DtexSize);
    
    for (i=0; (sz = 1 << i) <= max2DtexSize; i++) {
      glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGB8,
                   sz, sz, 0, 
                   GL_RGB, GL_UNSIGNED_BYTE, NULL);
       
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &x);
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &y);
    
      if (x > 0 && y > 0) { 
        max2DtexX = x;
        max2DtexY = y;
      }
    }

    if (max2DtexX > max2DtexSize)
      max2DtexX = max2DtexSize;
        
    if (max2DtexY > max2DtexSize)
      max2DtexY = max2DtexSize;
  } 
#endif

#if defined(GL_VERSION_1_2)
  if (ext->hastex3d) {
    int i, sz;
    GLint x, y, z;

    // test actual maximums for desired format
    max3DtexSize = 0; // until successfully queried from OpenGL
    max3DtexX = 0;
    max3DtexY = 0;
    max3DtexZ = 0;
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max3DtexSize);
    
    for (i=0; (sz = 1 << i) <= max3DtexSize; i++) {
      GLTEXIMAGE3D(GL_PROXY_TEXTURE_3D, 0, GL_RGB8, 
                   sz, sz, sz, 0, 
                   GL_RGB, GL_UNSIGNED_BYTE, NULL);
 
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D, 0, GL_TEXTURE_WIDTH,  &x);
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D, 0, GL_TEXTURE_HEIGHT, &y);
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D, 0, GL_TEXTURE_DEPTH,  &z);
   
      if (x > 0 && y > 0 && z > 0) {
        max3DtexX = x; 
        max3DtexY = y; 
        max3DtexZ = z; 
      }  
    }

    if (max3DtexX > max3DtexSize)
      max3DtexX = max3DtexSize;
        
    if (max3DtexY > max3DtexSize)
      max3DtexY = max3DtexSize;

    if (max3DtexZ > max3DtexSize)
      max3DtexZ = max3DtexSize;

    // disable 3-D texturing on cards that return unusable max texture sizes
    if (max3DtexX < 1 || max3DtexY < 1 || max3DtexZ < 1) {
      ext->hastex3d = 0;
    }

  }  
#endif



// MacOS X has had a broken implementation of GL_SEPARATE_SPECULAR_COLOR
// for some time.
#if defined(GL_VERSION_1_2) && !defined(__APPLE__)
  if (((ext->oglmajor == 1) && (ext->oglminor >= 2)) || (ext->oglmajor >= 2)) {
    if (ext->hastex2d || ext->hastex3d) {
      // Make specular color calculations follow texture mapping operations
      glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);
    } else {
      // Do the specular color calculations at the same time as the rest 
      glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SINGLE_COLOR);
    }
  }
#endif

  ext->PrintExtensions(); ///< print out information on OpenGL Extensions

#if defined(VMDUSEOPENGLSHADER)
  int glslextensionsavailable=0;

  // Enable OpenGL programmable shading if it is available
  if (!(simplegraphics || ogl_acrobat3dcapture) &&
      ext->hasglshadinglangarb &&
      ext->hasglfragmentshaderarb && 
      ext->hasglvertexshaderarb   &&
      ext->hasglshaderobjectsarb &&
      (getenv("VMDNOGLSL") == NULL)) {
    glslextensionsavailable=1; // GLSL is available
  }

  if (glslextensionsavailable) {
    mainshader = new OpenGLShader(ext);

    if (mainshader) {
      char *shaderpath = NULL;

      if (getenv("VMDOGLSHADER") != NULL) {
        shaderpath = (char *) calloc(1, strlen(getenv("VMDOGLSHADER")) + 512);
        strcpy(shaderpath, getenv("VMDOGLSHADER"));
      } else if (getenv("VMDDIR") != NULL) {
        shaderpath = (char *) calloc(1, strlen(getenv("VMDDIR")) + 512);
        strcpy(shaderpath, getenv("VMDDIR"));
        strcat(shaderpath, "/shaders/vmd");
      } else {
        msgErr << "Unable to locate VMD vertex and fragment shader path, " 
               << "VMDDIR environment variable not set" << sendmsg;
        delete mainshader;
        mainshader = NULL;
      }   

      if (mainshader) {
#if defined(_MSC_VER)
        // convert '/' to '\' for Windows...
        int i, len;
        len=strlen(shaderpath);
        for (i=0; i<len; i++) {
          if (shaderpath[i] == '\\') {
            shaderpath[i] = '/';
          }
        }
#endif

        if (mainshader->LoadShader(shaderpath)) {
          mainshader->UseShader(0); // if glsl is available, turn off initially
          // OpenGL rendering state gets propagated on-demand at render time 
          // whenever ogl_renderstateserial != ogl_glslserial, thus no need to
          // enable the shader immediately at startup anymore.
        } else {
          msgWarn << "GPU driver failed to compile shader: " << sendmsg;
          msgWarn << "  " << shaderpath << sendmsg;
          delete mainshader;
          mainshader = NULL;
        }
      }
 
      if (shaderpath)
        free(shaderpath);
    }
    OGLERR // enable OpenGL debugging code
  }

#if defined(VMDUSEGLSLSPHERES)
  // if the main shader compiled successfully, try loading up the 
  // sphere shader also
  if (mainshader) {
    sphereshader = new OpenGLShader(ext);
    char *shaderpath = NULL;

    if (getenv("VMDOGLSPHERESHADER") != NULL) {
      shaderpath = (char *) calloc(1, strlen(getenv("VMDOGLSPHERESHADER")) + 512);
      strcpy(shaderpath, getenv("VMDOGLSPHERESHADER"));
    } else if (getenv("VMDDIR") != NULL) {
      shaderpath = (char *) calloc(1, strlen(getenv("VMDDIR")) + 512);
      strcpy(shaderpath, getenv("VMDDIR"));
      strcat(shaderpath, "/shaders/vmdsphere");
    } else {
      msgWarn << "Unable to locate VMD sphere vertex and fragment shaders, " 
              << "VMDDIR environment variable not set" << sendmsg;
      delete sphereshader;
      sphereshader = NULL;
    }   

    if (sphereshader) {
#if defined(_MSC_VER)
      // convert '/' to '\' for Windows...
      int i, len;
      len=strlen(shaderpath);
      for (i=0; i<len; i++) {
        if (shaderpath[i] == '\\') {
          shaderpath[i] = '/';
        }
      }
#endif

      if (sphereshader->LoadShader(shaderpath)) {
        sphereshader->UseShader(0); // if glsl is available, turn off initially
        // OpenGL rendering state gets propagated on-demand at render time 
        // whenever ogl_renderstateserial != ogl_glslserial, thus no need to
        // enable the shader immediately at startup anymore.
      } else {
        msgWarn << "GPU driver failed to compile shader: " << sendmsg;
        msgWarn << "  " << shaderpath << sendmsg;
        delete sphereshader;
        sphereshader = NULL;
      }
    }

    if (shaderpath)
      free(shaderpath);

    OGLERR // enable OpenGL debugging code
  }
#endif


#if defined(VMDUSEGLSLSPHERESPRITES)
  // if the main shader compiled successfully, try loading up the 
  // sphere shader also
  if (mainshader 
#if 0
&& getenv("VMDUSESPHERESPRITES")
#endif
      ) {
    spherespriteshader = new OpenGLShader(ext);
    char *shaderpath = NULL;

    if (getenv("VMDOGLSPHERESPRITESHADER") != NULL) {
      shaderpath = (char *) calloc(1, strlen(getenv("VMDOGLSPHERESPRITESHADER")) + 512);
      strcpy(shaderpath, getenv("VMDOGLSPHERESPRITESHADER"));
    } else if (getenv("VMDDIR") != NULL) {
      shaderpath = (char *) calloc(1, strlen(getenv("VMDDIR")) + 512);
      strcpy(shaderpath, getenv("VMDDIR"));
      strcat(shaderpath, "/shaders/vmdspheresprite");
    } else {
      msgWarn << "Unable to locate VMD sphere sprite vertex and fragment shaders, " 
              << "VMDDIR environment variable not set" << sendmsg;
      delete spherespriteshader;
      spherespriteshader = NULL;
    }   

    if (spherespriteshader) {
#if defined(_MSC_VER)
      // convert '/' to '\' for Windows...
      int i, len;
      len=strlen(shaderpath);
      for (i=0; i<len; i++) {
        if (shaderpath[i] == '\\') {
          shaderpath[i] = '/';
        }
      }
#endif

      if (spherespriteshader->LoadShader(shaderpath)) {
        spherespriteshader->UseShader(0); // if glsl is available, turn off initially
        // OpenGL rendering state gets propagated on-demand at render time 
        // whenever ogl_renderstateserial != ogl_glslserial, thus no need to
        // enable the shader immediately at startup anymore.
      } else {
        msgWarn << "GPU driver failed to compile shader: " << sendmsg;
        msgWarn << "  " << shaderpath << sendmsg;
        delete spherespriteshader;
        spherespriteshader = NULL;
      }
    }

    if (shaderpath)
      free(shaderpath);

    OGLERR // enable OpenGL debugging code
  }
#endif

  if (mainshader && sphereshader 
#if defined(VMDUSEGLSLSPHERESPRITES)
      && ((spherespriteshader != 0) 
#if 0
           == (getenv("VMDUSESPHERESPRITES") != NULL)
#endif
         )
#endif
      ) {
    msgInfo << "  Full GLSL rendering mode is available." << sendmsg;
  } else if (mainshader) {
    if (glslextensionsavailable) {
      msgWarn << "This GPU/driver is buggy, or doesn't fully implement GLSL." << sendmsg;
      msgWarn << "Set environment variable VMDGLSLVERBOSE for more info." << sendmsg;
    }
    msgInfo << "  Basic GLSL rendering mode is available." << sendmsg;
  } else {
    if (glslextensionsavailable) {
      msgWarn << "This GPU/driver is buggy, or doesn't fully implement GLSL." << sendmsg;
      msgWarn << "Set environment variable VMDGLSLVERBOSE for more info." << sendmsg;
    }
    msgInfo << "  GLSL rendering mode is NOT available." << sendmsg;
  }
#endif

  // print information on OpenGL texturing hardware
  if (ext->hastex2d || ext->hastex3d) {
    msgInfo << "  Textures: ";
  
    if (ext->hastex2d) 
      msgInfo << "2-D (" << max2DtexX << "x" << max2DtexY << ")"; 

    if (ext->hastex2d && ext->hastex3d)
      msgInfo << ", ";

    if (ext->hastex3d) 
      msgInfo << "3-D (" << max3DtexX << "x" << max3DtexY << "x" << max3DtexZ << ")";

    if ((ext->hastex2d || ext->hastex3d) && ext->multitextureunits > 0)
      msgInfo << ", ";

    if (ext->multitextureunits > 0)
      msgInfo << "Multitexture (" << ext->multitextureunits << ")";

    msgInfo << sendmsg;
  }

  // print information about special stereo configuration
  if (getenv("VMDIMMERSADESKFLIP") != NULL) {
    immersadeskflip = 1;
    msgInfo << "  Enabled Immersadesk right-eye reflection stereo mode" << sendmsg;
  }

  // print information about special stereo configuration
  if (getenv("VMDSHEARSTEREO") != NULL) {
    shearstereo = 1;
    msgInfo << "  Enabled shear matrix stereo projection mode" << sendmsg;
  }

  OGLERR // enable OpenGL debugging code
}


void OpenGLRenderer::update_lists(void) {
  // point SphereList to the proper list
  ResizeArray<GLuint> *lists = (sphereMode == 
      ::SOLIDSPHERE) ? &solidSphereLists : &pointSphereLists;
  int ind = sphereRes - MIN_SPHERE_RES;
  if (ind < 0) 
    ind = 0;
  else if (ind >= lists->num())
    ind = lists->num()-1;
  SphereList = (*lists)[ind];
}

/////////////////////////  constructor and destructor  
// constructor ... initialize some variables
OpenGLRenderer::OpenGLRenderer(const char *nm) : DisplayDevice(nm) {
#if defined(VMD_NANOHUB)
  _finalFbo = _finalColorTex = _finalDepthRb = 0;
#endif

  // initialize data
#if defined(VMDUSELIBGLU)
  objQuadric = NULL;
  pointsQuadric = NULL;
#endif

#if defined(VMDUSEOPENGLSHADER)
  mainshader = NULL;         // init shaders to NULL until they're loaded
  sphereshader = NULL;       // init shaders to NULL until they're loaded
  spherespriteshader = NULL; // init shaders to NULL until they're loaded
#endif
  ext = new OpenGLExtensions;

  dpl_initialized = 0; // display lists need to be initialized still
}


// destructor
OpenGLRenderer::~OpenGLRenderer(void) {
#if defined(VMDUSELIBGLU)
  if (objQuadric != NULL)
    gluDeleteQuadric(objQuadric);     // delete the quadrics

  if (pointsQuadric != NULL)
    gluDeleteQuadric(pointsQuadric);  // delete the quadrics
#endif

  delete ext;                         // delete OpenGL extensions

#if defined(VMDUSEOPENGLSHADER)
  delete mainshader;                  // delete programmable shaders
  delete sphereshader;                // delete programmable shaders
  delete spherespriteshader;          // delete programmable shaders
#endif
}

// prepare to free OpenGL context (should be called from subclass destructor)
void OpenGLRenderer::free_opengl_ctx() {
  int i;
  GLuint tag;

  // delete all cached display lists
  displaylistcache.markUnused();
  while ((tag = displaylistcache.deleteUnused()) != GLCACHE_FAIL) {
    glDeleteLists(tag, 1);
  }

  // delete all cached textures
  texturecache.markUnused();
  while ((tag = texturecache.deleteUnused()) != GLCACHE_FAIL) {
    glDeleteTextures(1, &tag);
  }

  if (dpl_initialized) { 
    // free sphere display lists
    for (i=MIN_SPHERE_RES; i<=MAX_SPHERE_RES; i++) {
      glDeleteLists(solidSphereLists[i-MIN_SPHERE_RES], 1);
      glDeleteLists(pointSphereLists[i-MIN_SPHERE_RES], 1);
    } 

    // free the display lists used for the 3-D label/text font
    glDeleteLists(font1pxListBase, 256);
    glDeleteLists(fontNpxListBase, 256);
  }
}


/////////////////////////  protected nonvirtual routines  

// change current line width
void OpenGLRenderer::set_line_width(int w) {
  if(w > 0) {
    glLineWidth((GLfloat)w);
    lineWidth = w;
  }
}

// change current line style
void OpenGLRenderer::set_line_style(int s) {
  if(s == ::DASHEDLINE) {
    lineStyle = s;
    glEnable(GL_LINE_STIPPLE);
  } else {
    lineStyle = ::SOLIDLINE;
    glDisable(GL_LINE_STIPPLE);
  }
}


// change current sphere resolution
void OpenGLRenderer::set_sphere_res(int r) {
  // avoid unnecessary display list state changes, helps avoid some serious
  // OpenGL performance problems on MacOS X.
  if (sphereRes == r)
    return; 

  if (r > 2)
    sphereRes = r;
  else
    sphereRes = 2;

  update_lists();
}


// change current sphere type
void OpenGLRenderer::set_sphere_mode(int m) {
  // avoid unnecessary display list state changes, helps avoid some serious
  // OpenGL performance problems on MacOS X.
  if (sphereMode == m)
    return; 

  sphereMode = m;
  update_lists();
}


// this routine draws a cylinder from start to end, using rod_res panels,
// of radius rod_radius
void OpenGLRenderer::cylinder(float *end, float *start, int rod_res,
                              float rod_radius, float rod_top_radius) {
#if !defined(VMDUSELIBGLU)
  vmd_DrawConic(start, end, rod_radius, rod_top_radius, rod_res);
#else
  float R, RXY, phi, theta, lenaxis[3];

  // need to do some preprocessing ... find length of vector
  lenaxis[0] = end[0] - start[0];
  lenaxis[1] = end[1] - start[1];
  lenaxis[2] = end[2] - start[2];

  R = lenaxis[0]*lenaxis[0]+lenaxis[1]*lenaxis[1]+lenaxis[2]*lenaxis[2];

  if (R <= 0.0)
    return; // early exit if cylinder is of length 0;

  R = sqrtf(R); // evaluation of sqrt() _after_ early exit 

  // determine phi rotation angle, amount to rotate about y
  phi = acosf(lenaxis[2] / R);

  // determine theta rotation, amount to rotate about z
  RXY = sqrtf(lenaxis[0]*lenaxis[0]+lenaxis[1]*lenaxis[1]);
  if (RXY <= 0.0f) {
    theta = 0.0f;
  } else {
    theta = acosf(lenaxis[0] / RXY);
    if (lenaxis[1] < 0.0f)
      theta = (float) (2.0f * VMD_PI) - theta;
  }

  glPushMatrix(); // setup transform moving cylinder from Z-axis to position
  glTranslatef((GLfloat)(start[0]), (GLfloat)(start[1]), (GLfloat)(start[2]));
  if (theta != 0.0f)
    glRotatef((GLfloat) ((theta / VMD_PI) * 180.0f), 0.0f, 0.0f, 1.0f);
  if (phi != 0.0f)
    glRotatef((GLfloat) ((phi / VMD_PI) * 180.0f), 0.0f, 1.0f, 0.0f);

  // call utility routine to draw cylinders
  gluCylinder(objQuadric, (GLdouble)rod_radius, (GLdouble)rod_top_radius,
	      (GLdouble)R, (GLint)rod_res, 1);

  // if this is a cone, we also draw a disk at the bottom
  gluQuadricOrientation(objQuadric, (GLenum)GLU_INSIDE);
  gluDisk(objQuadric, (GLdouble)0, (GLdouble)rod_radius, (GLint)rod_res, 1);
  gluQuadricOrientation(objQuadric, (GLenum)GLU_OUTSIDE);

  glPopMatrix(); // restore the previous transformation matrix
#endif
}


// this routine also draws a cylinder.  However, it assumes that
// the cylinder drawing command has precomputed the data.  This
// uses more memory, but is faster
// the data are: num == number of edges
//  edges = a normal, start, and end 
static void cylinder_full(int num, float *edges, int filled) {
  int n = num;
  float *start = edges;

  if (num < 2)
     return;

  glBegin(GL_QUAD_STRIP);
    while (n-- > 0) {
      glNormal3fv(edges);
      glVertex3fv(edges+6);
      glVertex3fv(edges+3);
      edges += 9;
    }
    glNormal3fv(start);  // loop back to the beginning
    glVertex3fv(start+6);
    glVertex3fv(start+3);
  glEnd();

  // and fill in the top and bottom, if needed
  if (filled) {
    float axis[3];
    axis[0] = start[6] - start[3];
    axis[1] = start[7] - start[4];
    axis[2] = start[8] - start[5];
    vec_normalize(axis);

    if (filled & CYLINDER_LEADINGCAP) { // do the first side
      n = num;            // get one side
      edges = start + 3;
      glBegin(GL_POLYGON);
        glNormal3fv(axis);
        while (--n >= 0) {
          glVertex3fv(edges);
          edges += 9;
        }
      glEnd();
    }
    if (filled & CYLINDER_TRAILINGCAP) { // do the other side
      n = num;          // and the other
      edges = start + 6;
      glBegin(GL_POLYGON);
        glNormal3fv(axis);       // I'm going the other direction, so
        while (--n >= 0) {
          glVertex3fv(edges);
          edges += 9;
        }
      glEnd();
    }
  }
}


/////////////////////////  protected virtual routines  

// define a new light source ... return success of operation
int OpenGLRenderer::do_define_light(int n, float *color, float *position) {
  int i;
 
  for(i=0; i < 3; i++)  {
    ogl_lightcolor[n][i] = color[i];
    ogl_lightpos[n][i] = position[i];
  }
  ogl_lightpos[n][3] = 0.0; // directional lights require w=0.0 otherwise
                            // OpenGL assumes they are positional lights.
  ogl_lightcolor[n][3] = 1.0;

  // normalize the light direction vector
  vec_normalize(&ogl_lightpos[n][0]); // 4th element is left alone

  glLightfv((GLenum)(GL_LIGHT0 + n), GL_POSITION, &ogl_lightpos[n][0]);
  glLightfv((GLenum)(GL_LIGHT0 + n), GL_SPECULAR, &ogl_lightcolor[n][0]);
  
  ogl_rendstateserial++; // cause GLSL cached state to update when necessary
  _needRedraw = 1;
  return TRUE;
}

// activate a given light source ... return success of operation
int OpenGLRenderer::do_activate_light(int n, int turnon) {
  if (turnon) {
    glEnable((GLenum)(GL_LIGHT0 + n));
    ogl_lightstate[n] = 1;
  } else {
    glDisable((GLenum)(GL_LIGHT0 + n));
    ogl_lightstate[n] = 0;
  }

  ogl_rendstateserial++; // cause GLSL cached state to update when necessary
  _needRedraw = 1;
  return TRUE;
}

void OpenGLRenderer::loadmatrix(const Matrix4 &m) {
  GLfloat tmpmat[16];
  for (int i=0; i<16; i++) tmpmat[i]=(GLfloat)(m.mat[i]);
  glLoadMatrixf(tmpmat);
}

void OpenGLRenderer::multmatrix(const Matrix4 &m) {
  GLfloat tmpmat[16];
  for (int i=0; i<16; i++) tmpmat[i]=(GLfloat)(m.mat[i]);
  glMultMatrixf(tmpmat);
}

// virtual routines to return 2D screen coordinates, given 2D or 3D world
// coordinates.  These assume the proper GL window has focus, etc.
// The xy coordinates returned are absolute screen coords, relative to 
// the lower left corner of the display monitor.  The returned Z coordinate
// has been normalized according to its position within the view frustum
// between the front and back clipping planes.  The normalized Z coordinate
// is used to avoid picking points that are outside of the visible portion
// of the view frustum.
void OpenGLRenderer::abs_screen_loc_3D(float *loc, float *spos) {
  GLdouble modelMatrix[16], projMatrix[16];
  GLdouble pos[3];
  int i;

  // get current matrices and viewport for project call
  for (i=0; i<16; i++) {
    modelMatrix[i] = ogl_mvmatrix[i];
    projMatrix[i] = ogl_pmatrix[i];
  }

  // call the GLU routine to project the object coord to world coords
  if(!vmd_Project((GLdouble)(loc[0]), (GLdouble)(loc[1]), (GLdouble)(loc[2]),
     modelMatrix, projMatrix, ogl_viewport, pos, pos + 1, pos + 2)) {
    msgErr << "Cannot determine window position of world coordinate.";
    msgErr << sendmsg;
  } else {
    spos[0] = (float) (pos[0] + (float)xOrig);
    spos[1] = (float) (pos[1] + (float)yOrig);
    spos[2] = (float) (pos[2]);
  }
}

void OpenGLRenderer::abs_screen_loc_2D(float *loc, float *spos) {
  float newloc[3];
  newloc[0] = loc[0];
  newloc[1] = loc[1];
  newloc[2] = 0.0f;
  abs_screen_loc_3D(newloc, spos);
}

// Given a 3D point (pos A),
// and a 2D rel screen pos point (for pos B), computes the 3D point
// which goes with the second 2D point at pos B.  Result returned in B3D.
// NOTE: currently, this algorithm only supports the simple case where the
// eye look direction is along the Z-axis.  A more sophisticated version
// requires finding the plane to which the look direction is normal, which is
// assumed here to be the Z-axis (for simplicity in coding).
void OpenGLRenderer::find_3D_from_2D(const float *A3D, const float *B2D,
				     float *B3D) {
  GLdouble modelMatrix[16], projMatrix[16], w1[3], w2[3];
  int i;
  float lsx, lsy;		// used to convert rel screen -> abs

  // get current matrices and viewport for unproject call
  for (i=0; i<16; i++) {
    modelMatrix[i] = ogl_mvmatrix[i];
    projMatrix[i] = ogl_pmatrix[i];
  }

  // get window coordinates of 2D point
  lsx = B2D[0];
  lsy = B2D[1];
  lsx = lsx * (float)xSize;
  lsy = lsy * (float)ySize;

  // call the GLU routine to unproject the window coords to world coords
  if (!vmd_UnProject((GLdouble)lsx, (GLdouble)lsy, 0,
      modelMatrix, projMatrix, ogl_viewport, w1, w1 + 1, w1 + 2)) {
    msgErr << "Can't determine world coords of window position 1." << sendmsg;
    return;
  }
  if (!vmd_UnProject((GLdouble)lsx, (GLdouble)lsy, 1.0,
      modelMatrix, projMatrix, ogl_viewport, w2, w2 + 1, w2 + 2)) {
    msgErr << "Can't determine world coords of window position2." << sendmsg;
    return;
  }

  // finally, find the point where line returned as w1..w2 intersects the 
  // given 3D point's plane (this plane is assumed to be parallel to the X-Y
  // plane, i.e., with a normal along the Z-axis.  A more general algorithm
  // would need to find the plane which is normal to the eye look direction,
  // and which contains the given 3D point.)
  
  // special case: w1z = w2z ==> just return given 3D point, since there
  //		is either no solution, or the line is in the given plane
  if(w1[2] == w2[2]) {
    memcpy(B3D, A3D, 3*sizeof(float));
  } else {
    float relchange = (float) ((A3D[2] - w1[2]) / (w2[2] - w1[2]));
    B3D[0] = (float) ((w2[0] - w1[0]) * relchange + w1[0]);
    B3D[1] = (float) ((w2[1] - w1[1]) * relchange + w1[1]);
    B3D[2] = A3D[2];
  }
}


//
// antialiasing and depth-cueing
//

// turn on antialiasing effect
void OpenGLRenderer::aa_on(void) {
  if (inStereo == OPENGL_STEREO_STENCIL_CHECKERBOARD ||
      inStereo == OPENGL_STEREO_STENCIL_COLUMNS ||
      inStereo == OPENGL_STEREO_STENCIL_ROWS) {
    msgInfo << "Antialiasing must be disabled for stencil-based stereo modes."
<< sendmsg;
    msgInfo << "You may re-enable antialiasing when stereo is turned off." << sendmsg;
    aa_off();
    return;
  }

  if (aaAvailable && !aaEnabled) {
#if defined(GL_ARB_multisample)
    if (ext->hasmultisample) {
      glEnable(GL_MULTISAMPLE_ARB);
      aaEnabled = TRUE;
      _needRedraw = 1;
      return;
    } 
#endif
    // could implement accumulation buffer antialiasing here someday
    aaEnabled = TRUE;
  }
}

// turn off antialiasing effect
void OpenGLRenderer::aa_off(void) {
  if(aaAvailable && aaEnabled) {
#if defined(GL_ARB_multisample)
    if (ext->hasmultisample) {
      glDisable(GL_MULTISAMPLE_ARB);
      aaEnabled = FALSE;
      _needRedraw = 1;
      return;
    } 
#else
#endif
    // could implement accumulation buffer antialiasing here someday
    aaEnabled = FALSE;
  }
}

// turn on hardware depth-cueing
void OpenGLRenderer::cueing_on(void) {
  if (cueingAvailable && !cueingEnabled) {
    glEnable(GL_FOG);
    cueingEnabled = TRUE;
    _needRedraw = 1;
  }
}

// turn off hardware depth-cueing
void OpenGLRenderer::cueing_off(void) {
  if (cueingAvailable && cueingEnabled) {
    glDisable(GL_FOG);
    cueingEnabled = FALSE;
    _needRedraw = 1;
  }
}


void OpenGLRenderer::culling_on(void) {
  if (cullingAvailable && !cullingEnabled) {
    glFrontFace(GL_CCW);              // set CCW as fron face direction
    glPolygonMode(GL_FRONT, GL_FILL); // set front face fill mode
    glPolygonMode(GL_BACK,  GL_LINE); // set back face fill mode
    glCullFace(GL_BACK);              // set for culling back faces
    glEnable(GL_CULL_FACE);           // enable face culling
    cullingEnabled = TRUE; 
    _needRedraw = 1;
  }
}

void OpenGLRenderer::culling_off(void) {
  if (cullingAvailable && cullingEnabled) {
    glPolygonMode(GL_FRONT, GL_FILL); // set front face fill mode
    glPolygonMode(GL_BACK,  GL_FILL); // set back face fill mode
    glCullFace(GL_BACK);              // set for culling back faces
    glDisable(GL_CULL_FACE);          // disable face culling
    cullingEnabled = FALSE; 
    _needRedraw = 1;
  }
}

void OpenGLRenderer::set_background(const float *newback) {
  GLfloat r, g, b;
  r = (GLfloat)newback[0];
  g = (GLfloat)newback[1];
  b = (GLfloat)newback[2];

  // set fog color used for depth cueing
  GLfloat fogcol[4];
  fogcol[0] = r;
  fogcol[1] = g;
  fogcol[2] = b;
  fogcol[3] = 1.0;

  glFogfv(GL_FOG_COLOR, fogcol);

  // set clear color
  glClearColor((GLclampf)r,
               (GLclampf)g,
               (GLclampf)b, 1.0);
}

void OpenGLRenderer::set_backgradient(const float *topcol, 
                                      const float *botcol) {
  int i;
  for (i=0; i<3; i++) {
    ogl_backgradient[0][i] = topcol[i]; 
    ogl_backgradient[1][i] = botcol[i]; 
  }
  ogl_backgradient[0][3] = 1.0;
  ogl_backgradient[1][3] = 1.0;
}

// change to a different stereo mode
void OpenGLRenderer::set_stereo_mode(int newMode) {
  if (inStereo == newMode)
    return;   // do nothing if current mode is same as specified mode

  if (inStereo == OPENGL_STEREO_STENCIL_CHECKERBOARD ||
      inStereo == OPENGL_STEREO_STENCIL_COLUMNS ||
      inStereo == OPENGL_STEREO_STENCIL_ROWS)
    disable_stencil_stereo(); 

  if (newMode == OPENGL_STEREO_STENCIL_CHECKERBOARD ||
      newMode == OPENGL_STEREO_STENCIL_COLUMNS ||
      newMode == OPENGL_STEREO_STENCIL_ROWS)
    enable_stencil_stereo(newMode); 

  inStereo = newMode;  // set new mode
  reshape();           // adjust the viewport width/height
  normal();            // adjust the viewport size/projection matrix
                       // this is reset again when left/right are called.
  clear();             // clear the screen
  update();            // redraw

  _needRedraw = 1;
}

// change to a different caching mode
void OpenGLRenderer::set_cache_mode(int newMode) {
  cacheMode = newMode; // set new mode;
  ogl_cacheenabled = newMode;
}

// change to a different rendering mode
void OpenGLRenderer::set_render_mode(int newMode) {
  if (renderMode == newMode)
    return;   // do nothing if current mode is same as specified mode

  renderMode = newMode;  // set new mode

  switch (renderMode) {
    case OPENGL_RENDER_NORMAL:
      ogl_useblendedtrans = 0;
      ogl_useglslshader = 0;
      ogl_acrobat3dcapture = 0;
      break;

    case OPENGL_RENDER_GLSL:
#if defined(VMDUSEOPENGLSHADER)
      // GLSL shader state variables must now be updated to match the 
      // active fixed-pipeline state before/during the next rendering pass. 
      if (mainshader) {
        ogl_useblendedtrans = 1;
        ogl_useglslshader = 1;
      } else
#endif
      {
        ogl_useblendedtrans = 0;
        ogl_useglslshader = 0;
        msgWarn << "OpenGL Programmable Shading not available." << sendmsg;
      }
      ogl_acrobat3dcapture = 0;
      break;

    case OPENGL_RENDER_ACROBAT3D:
      ogl_useblendedtrans = 0;
      ogl_useglslshader = 0;
      ogl_acrobat3dcapture = 1;
      break;
  }

  reshape();           // adjust the viewport width/height
  normal();            // adjust the viewport size/projection matrix
                       // this is reset again when left/right are called.
  clear();             // clear the screen
  update();            // redraw

  _needRedraw = 1;
}


// set up for normal (non-stereo) drawing.  Sets the viewport and perspective.
void OpenGLRenderer::normal(void) {
  glViewport(0, 0, (GLsizei)xSize, (GLsizei)ySize);
  set_persp();

  // draw the background gradient if necessary
  draw_background_gradient();
}


void OpenGLRenderer::enable_stencil_stereo(int newMode) {
  int i;
  
  if (!ext->hasstencilbuffer) {
    set_stereo_mode(OPENGL_STEREO_OFF); 
    msgInfo << "Stencil Buffer Stereo is NOT available." << sendmsg;
    return;
  } 

  if (aaEnabled) {
    msgInfo << "Antialiasing must be disabled for stencil-based stereo modes." << sendmsg;
    msgInfo << "Antialiasing will be re-enabled when stereo is turned off." << sendmsg;
    aaPrevious = aaEnabled;
    aa_off();
  }

  glPushMatrix();
  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, (GLsizei)xSize, (GLsizei)ySize);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity();
  
  glOrtho(0.0, xSize, 0.0, ySize, -1.0, 1.0); // 2-D orthographic projection

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // clearing and configuring stencil drawing
  glDrawBuffer(GL_BACK);

  glEnable(GL_STENCIL_TEST);
  glClearStencil(0);
  glClear(GL_STENCIL_BUFFER_BIT);
  glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE); 
  glStencilFunc(GL_ALWAYS, 1, 1); 

  glColor4f(1,1,1,0); // set draw color to all 1s

  // According to Appendix G. of the OpenGL 1.2 Programming Guide
  // correct 2-D line rasterization requires placing vertices at half
  // pixel offsets.  This is mentioned specifically on page 677.
  glDisable(GL_LINE_STIPPLE); // ensure stippling is off
  glLineWidth(1);
  glBegin(GL_LINES);
  if (newMode == OPENGL_STEREO_STENCIL_CHECKERBOARD) {
    // Draw the stencil pattern on every other pixel of the window
    // in a checkerboard pattern, by drawing diagonal lines.
    for (i = -ySize; i < xSize+ySize; i += 2) {
      glVertex2f((GLfloat) i         + 0.5f, (GLfloat)         0.5f);
      glVertex2f((GLfloat) i + ySize + 0.5f, (GLfloat) ySize + 0.5f);
    }
  } else if (newMode == OPENGL_STEREO_STENCIL_COLUMNS) {
    // Draw the stencil pattern on every other column of the window.
    for (i=0; i<xSize; i+=2) {
      glVertex2f((GLfloat) i + 0.5f,            0.0f);
      glVertex2f((GLfloat) i + 0.5f, (GLfloat) ySize);
    }
  } else if (newMode == OPENGL_STEREO_STENCIL_ROWS) {
    // draw the stencil pattern on every other row of the window.
    for (i=0; i<ySize; i+=2) {
      glVertex2f(           0.0f, (GLfloat) i + 0.5f);
      glVertex2f((GLfloat) xSize, (GLfloat) i + 0.5f);
    }
  }
  glEnd();

  glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP); // disable changes to stencil buffer

  glEnable(GL_DEPTH_TEST);
  
  glPopMatrix();
}

void OpenGLRenderer::disable_stencil_stereo(void) {
  glDisable(GL_STENCIL_TEST);
  if (aaPrevious) {
    // XXX hack to please aa_on() so it'll reenable stereo even though
    // inStereo isn't quite back to normal yet.
    int foo = inStereo;    
    inStereo = OPENGL_STEREO_OFF;
    aa_on(); // re-enable antialiasing if we're leaving stenciling mode
    inStereo = foo;
    msgInfo << "Antialiasing re-enabled." << sendmsg;
  }
}


// set up for drawing the left eye image.  Assume always the left eye is
// drawn first (so no zclear is needed before it)
void OpenGLRenderer::left(void) {
  DisplayEye cureye = LEFTEYE;
  if (stereoSwap) {
    switch (inStereo) {
      case OPENGL_STEREO_HDTVSIDE:
      case OPENGL_STEREO_SIDE:
      case OPENGL_STEREO_ABOVEBELOW:
      case OPENGL_STEREO_QUADBUFFER:
      case OPENGL_STEREO_STENCIL_CHECKERBOARD:
      case OPENGL_STEREO_STENCIL_COLUMNS:
      case OPENGL_STEREO_STENCIL_ROWS:
      case OPENGL_STEREO_ANAGLYPH:
        cureye = RIGHTEYE;
        break;
    }
  }

  switch (inStereo) {
    case OPENGL_STEREO_HDTVSIDE:
      glViewport(0, 0, (GLsizei)xSize / 2, (GLsizei)ySize);
      set_persp(cureye);
      break;

    case OPENGL_STEREO_SIDE:
      glViewport(0, 0, (GLsizei)xSize / 2, (GLsizei)ySize);
      set_persp(cureye);
      break;

    case OPENGL_STEREO_ABOVEBELOW:
      glViewport(0, 0, (GLsizei)xSize, (GLsizei)ySize / 2);
      set_persp(cureye);
      break;

    case OPENGL_STEREO_LEFT:
      set_persp(LEFTEYE);
      break;

    case OPENGL_STEREO_RIGHT:
      set_persp(RIGHTEYE);
      break;

    case OPENGL_STEREO_QUADBUFFER:
      if (ext->hasstereo) {
        glDrawBuffer(GL_BACK_LEFT); // Z-buffer must be cleared already
      } else {
        // XXX do something since we don't support non-quad buffered modes
        glViewport(0, (GLint)ySize / 2, (GLsizei)xSize, (GLsizei)ySize / 2);
      }
      set_persp(cureye);
      break;

    case OPENGL_STEREO_STENCIL_CHECKERBOARD:
    case OPENGL_STEREO_STENCIL_COLUMNS:
    case OPENGL_STEREO_STENCIL_ROWS:
      glStencilFunc(GL_NOTEQUAL,1,1); // draws if stencil <> 1
      set_persp(cureye);
      break;

    case OPENGL_STEREO_ANAGLYPH:
      if(ext->hasstereo) {
        glDrawBuffer(GL_BACK_LEFT); // Z-buffer must be cleared already
      }
      // Prevailing default anaglyph format is left-eye-red
      glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE); 
      set_persp(cureye);
      break;

    default:
      normal(); // left called even though we're non-stereo
// not tested on other platforms yet
#if defined(__APPLE__)
      if (ext->hasstereo && ext->stereodrawforced)
        glDrawBuffer(GL_BACK_LEFT); // draw to back-left
#endif
      break;
  }

  // draw the background gradient if necessary
  draw_background_gradient();
}


// set up for drawing the right eye image.  Assume always the right eye is
// drawn last (so a zclear IS needed before it)
void OpenGLRenderer::right(void) {
  DisplayEye cureye = RIGHTEYE;
  if (stereoSwap) {
    switch (inStereo) {
      case OPENGL_STEREO_HDTVSIDE:
      case OPENGL_STEREO_SIDE:
      case OPENGL_STEREO_ABOVEBELOW:
      case OPENGL_STEREO_QUADBUFFER:
      case OPENGL_STEREO_STENCIL_CHECKERBOARD:
      case OPENGL_STEREO_STENCIL_COLUMNS:
      case OPENGL_STEREO_STENCIL_ROWS:
      case OPENGL_STEREO_ANAGLYPH:
        cureye = LEFTEYE;
        break;
    }
  }

  switch (inStereo) {
    case OPENGL_STEREO_HDTVSIDE:
      glViewport((GLsizei)xSize / 2, 0, (GLsizei)xSize / 2, (GLsizei)ySize);
      set_persp(cureye);
      break;

    case OPENGL_STEREO_SIDE:
      glViewport((GLsizei)xSize / 2, 0, (GLsizei)xSize / 2, (GLsizei)ySize);
      set_persp(cureye);
      break;

    case OPENGL_STEREO_ABOVEBELOW:
      glViewport(0, (GLsizei)ySize / 2, (GLsizei)xSize, (GLsizei)ySize / 2);
      set_persp(cureye);
      break;

    case OPENGL_STEREO_LEFT:
    case OPENGL_STEREO_RIGHT:
      // no need to do anything, already done in call to left
      break;

    case OPENGL_STEREO_QUADBUFFER:
      if (ext->hasstereo) {
        glDepthMask(GL_TRUE);  // make Z-buffer writable
#if defined(__APPLE__)
        // XXX This hack is required by MacOS X because their 
        // Quadro 4500 stereo drivers are broken such that the 
        // clear on both right/left buffers doesn't actually work.
        // This explicitly forces a second clear on the back right buffer.
        glDrawBuffer(GL_BACK_RIGHT);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#else
        // all other platforms work fine
        glClear(GL_DEPTH_BUFFER_BIT);
#endif
        glDrawBuffer(GL_BACK_RIGHT);
      } else {
        // XXX do something since we don't support non-quad buffered modes
        glViewport(0, 0, (GLsizei)xSize, (GLsizei)ySize / 2);
      }
      set_persp(cureye);
      break;

    case OPENGL_STEREO_STENCIL_CHECKERBOARD:
    case OPENGL_STEREO_STENCIL_COLUMNS:
    case OPENGL_STEREO_STENCIL_ROWS:
      glDepthMask(GL_TRUE);  // make Z-buffer writable
      glClear(GL_DEPTH_BUFFER_BIT);
      glStencilFunc(GL_EQUAL,1,1); // draws if stencil <> 0
      set_persp(cureye);
      break;

    case OPENGL_STEREO_ANAGLYPH:
      if(ext->hasstereo) {
        glDrawBuffer(GL_BACK_RIGHT);
      }

      // Prevailing default anaglyph format is left-eye-red
#if 1
      // Use both green and blue components on right-eye, to yield
      // a more "full color" effect for red-blue and red-cyan glasses
      glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE); 
#else
      // Use blue channel only for reduced ghosting with cheap filters
      glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_TRUE); 
#endif
      glDepthMask(GL_TRUE);  // make Z-buffer writable
      glClear(GL_DEPTH_BUFFER_BIT);
      set_persp(cureye);
      break;

    default:
      normal(); // right called even though we're non-stereo
// not tested on other platforms yet
#if defined(__APPLE__)
      if (ext->hasstereo && ext->stereodrawforced)
        glDrawBuffer(GL_BACK_RIGHT); // draw to back-right
#endif
      break;
  }

  // draw the background gradient if necessary
  draw_background_gradient();
}


// set the current perspective, based on the eye position and where we
// are looking.
void OpenGLRenderer::set_persp(DisplayEye my_eye) {
  // define eye and look at some point.  Assumes up vector = (0,1,0)
  GLdouble ep[3];
  switch (my_eye) {
    case LEFTEYE:
      ep[0] = eyePos[0] - eyeSepDir[0];
      ep[1] = eyePos[1] - eyeSepDir[1];
      ep[2] = eyePos[2] - eyeSepDir[2];
      DisplayDevice::left();
      break;
    case RIGHTEYE: 
      ep[0] = eyePos[0] + eyeSepDir[0];
      ep[1] = eyePos[1] + eyeSepDir[1];
      ep[2] = eyePos[2] + eyeSepDir[2];
      DisplayDevice::right();
      break;

    case NOSTEREO:
    default:
      ep[0] = eyePos[0];
      ep[1] = eyePos[1];
      ep[2] = eyePos[2];
      DisplayDevice::normal();
      break;
  }

  // setup camera system and projection transformations
  if (projection() == PERSPECTIVE) {
    ogl_glslprojectionmode = DisplayDevice::PERSPECTIVE; 

    if (shearstereo) {
      // XXX almost ready for prime time, when testing is done we may
      // make shear stereo the default and eye rotation a backwards 
      // compatibility option.
      // Use the "frustum shearing" method for creating a stereo view.  
      // The frustum shearing method is preferable to eye rotation in general.

      // Calculate the eye shift (half eye separation distance)
      // XXX hack, needs to be more general
      float eyeshift = float(ep[0] - eyePos[0]);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      // Shifts viewing frustum horizontally in the image plane
      // according to the stereo eye separation if rendering in stereo.
      // XXX hack, the parameterization of this projection still 
      // needs work, but the fact that it incorporates eyeDist is nice.
      glFrustum((GLdouble)cpLeft  + (eyeshift * nearClip / eyeDist),
                (GLdouble)cpRight + (eyeshift * nearClip / eyeDist),
                (GLdouble)cpDown, 
                (GLdouble)cpUp,
                (GLdouble)nearClip, 
                (GLdouble)farClip);

      // Shift the eye position horizontally by half the eye separation
      // distance if rendering in stereo.
      glTranslatef(-eyeshift, 0.0, 0.0); 

      glMatrixMode(GL_MODELVIEW);
      // set modelview identity and then applies transform
      vmd_LookAt(eyePos[0], eyePos[1], eyePos[2],
                 (GLdouble)(eyePos[0] + eyeDir[0]),
                 (GLdouble)(eyePos[1] + eyeDir[1]),
                 (GLdouble)(eyePos[2] + eyeDir[2]),
                 (GLdouble)(upDir[0]),
                 (GLdouble)(upDir[1]),
                 (GLdouble)(upDir[2]));
    } else {
      // Use the "eye rotation" method for creating a stereo view.  
      // The frustum shearing method would be preferable.
      // XXX this implementation is not currently using the eyeDist
      // parameter, though it probably should.
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glFrustum((GLdouble)cpLeft,  (GLdouble)cpRight,  (GLdouble)cpDown,
                (GLdouble)cpUp,   (GLdouble)nearClip, (GLdouble)farClip);

      // Reflect the X axis of the right eye for the new LCD panel immersadesks
      // XXX experimental hack that needs more work to get lighting
      // completely correct for the Axes, Title Screen, etc.
      if (immersadeskflip && my_eye == RIGHTEYE) {
        // Scale the X axis by -1 in the GL_PROJECTION matrix
        glScalef(-1, 1, 1);
      }

      glMatrixMode(GL_MODELVIEW);
      // set modelview identity and then applies transform
      vmd_LookAt(ep[0], ep[1], ep[2],
                 (GLdouble)(eyePos[0] + eyeDir[0]),
                 (GLdouble)(eyePos[1] + eyeDir[1]),
                 (GLdouble)(eyePos[2] + eyeDir[2]),
                 (GLdouble)(upDir[0]),
                 (GLdouble)(upDir[1]),
                 (GLdouble)(upDir[2]));
    }
  } else { // ORTHOGRAPHIC
    ogl_glslprojectionmode = DisplayDevice::ORTHOGRAPHIC; 
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(-0.25 * vSize * Aspect, 0.25 * vSize * Aspect,
            -0.25 * vSize,          0.25 * vSize,
            nearClip, farClip);

    // Use the "eye rotation" method for creating a stereo view.  
    // The frustum shearing method won't work with orthographic 
    // views since the eye rays are parallel, so the rotation method 
    // is ok in this case.
    glMatrixMode(GL_MODELVIEW);
    // set modelview identity and then applies transform
    vmd_LookAt(ep[0], ep[1], ep[2],
               (GLdouble)(eyePos[0] + eyeDir[0]),
               (GLdouble)(eyePos[1] + eyeDir[1]),
               (GLdouble)(eyePos[2] + eyeDir[2]),
               (GLdouble)(upDir[0]),
               (GLdouble)(upDir[1]),
               (GLdouble)(upDir[2]));
  }

  // update the cached transformation matrices for use in text display, etc.
  glGetFloatv(GL_PROJECTION_MATRIX, ogl_pmatrix);
  glGetFloatv(GL_MODELVIEW_MATRIX, ogl_mvmatrix);
  glGetIntegerv(GL_VIEWPORT, ogl_viewport);
  ogl_textMat.identity();
  ogl_textMat.multmatrix(ogl_pmatrix);
  ogl_textMat.multmatrix(ogl_mvmatrix);
}


// prepare to draw a 3D image
int OpenGLRenderer::prepare3D(int do_clear) {
  if (do_clear) {
    clear();
  } else {
    glDepthMask(GL_TRUE);  // make Z-buffer writable
    glClear(GL_DEPTH_BUFFER_BIT);
  }

  // invalidate the OpenGL material index cache since a new frame is
  // being drawn and the material state for the previous index may 
  // have changed.  
  oglmaterialindex = -1;

  // start a new frame, marking all cached IDs as "unused"
  displaylistcache.markUnused();
  texturecache.markUnused();

  return TRUE; // must return true for normal (non file-based) renderers
}


// prepare to draw opaque objects
int OpenGLRenderer::prepareOpaque(void) {
  if (ogl_useblendedtrans) {
    glDepthMask(GL_TRUE); // make Z-buffer writable
    ogl_transpass = 0;
  }

  return 1;
}

// prepare to draw transparent objects
int OpenGLRenderer::prepareTrans(void) {
  if (ogl_useblendedtrans) {
    glDepthMask(GL_FALSE); // make Z-buffer read-only while drawing trans objs
    ogl_transpass = 1;
    return 1;
  }

  return 0;
}

// clear the display
void OpenGLRenderer::clear(void) {
  // clear the whole viewport, not just one side 
  switch (inStereo) {
    case OPENGL_STEREO_HDTVSIDE:
    case OPENGL_STEREO_SIDE:
    case OPENGL_STEREO_ABOVEBELOW:
    case OPENGL_STEREO_QUADBUFFER:
    case OPENGL_STEREO_ANAGLYPH:
      glViewport(0, 0, (GLsizei)xSize, (GLsizei)ySize);
      break;
  }

  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);    // reset color mask 
  glDepthMask(GL_TRUE);                               // make Z-buffer writable
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color/depth bufs

// not tested on other platforms yet
#if defined(__APPLE__)
  if (ext->hasstereo && ext->stereodrawforced) {
    glDrawBuffer(GL_BACK_RIGHT);
    glClear(GL_COLOR_BUFFER_BIT); // force separate clear of right buffer
    glDrawBuffer(GL_BACK);
  }
#endif
}


// draw the background gradient
void OpenGLRenderer::draw_background_gradient(void) {
  // if the background mode is set for gradient, then draw the gradient
  // note: this must be done immediately after clearing the viewport
  if (backgroundmode != 0) {
    int i;

    // disable all clipping planes by default
    for (i=0; i < VMD_MAX_CLIP_PLANE; i++) {
      ogl_clipmode[i] = 0; // off by default
      glDisable((GLenum) (GL_CLIP_PLANE0 + i));
    }

    glDisable(GL_LIGHTING);           // disable lighting
    ogl_lightingenabled=0;            // update state var
#if defined(VMDUSEOPENGLSHADER)
    if (mainshader && ogl_useglslshader) {
      mainshader->UseShader(0);       // use fixed-func pipeline
    }
#endif
    glDisable(GL_DEPTH_TEST);         // disable depth test
    glDepthMask(GL_FALSE);            // make Z-buffer read-only

    // turn off any transparent rendering state
    glDisable(GL_POLYGON_STIPPLE);    // make sure stippling is disabled
    glDisable(GL_BLEND);              // disable blending

    glMatrixMode(GL_MODELVIEW);       // save existing transformation state
    glPushMatrix();
    glLoadIdentity();                 // prepare for 2-D orthographic drawing

    glMatrixMode (GL_PROJECTION);     // save existing transformation state
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0); // 2-D orthographic projection

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();                 // add one more modelview

    // draw the background polygon
    glBegin(GL_QUADS);
      glColor3fv(&ogl_backgradient[1][0]);
      glVertex2f(0.0f, 0.0f);
      glColor3fv(&ogl_backgradient[1][0]);
      glVertex2f(1.0f, 0.0f);
      glColor3fv(&ogl_backgradient[0][0]);
      glVertex2f(1.0f, 1.0f);
      glColor3fv(&ogl_backgradient[0][0]);
      glVertex2f(0.0f, 1.0f);
    glEnd();

    glPopMatrix();                     // pop off top modelview

    glMatrixMode (GL_PROJECTION);
    glPopMatrix();                     // restore projection transform state

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();                     // restore modelview transform state

    glEnable(GL_DEPTH_TEST);           // restore depth testing
    glDepthMask(GL_TRUE);              // make Z-buffer writeable
    glEnable(GL_LIGHTING);             // restore lighting
    ogl_lightingenabled=1;             // update state var
#if defined(VMDUSEOPENGLSHADER)
    if (mainshader && ogl_useglslshader) {
      mainshader->UseShader(1);        // re-enable glsl mainshader
    }
#endif
  }
}


//**********************  the rendering routine  ***********************
//* This scans the given command list until the end, doing the commands
//* in the order they appear
//**********************************************************************
void OpenGLRenderer::render(const VMDDisplayList *cmdList) {
  char *cmdptr = NULL;  // ptr to current display command data
  int tok;              // what display command was encountered
  _needRedraw = 0;      // reset the flag now that we are drawing

  // early exit if any of these conditions are true. 
  if (!cmdList) 
    return;

  if (ogl_useblendedtrans) {
    if (ogl_transpass) {
      // skip rendering mostly Opaque objects on transparent pass
      if (cmdList->opacity > 0.50) 
        return;
    } else {
      // skip rendering mostly transparent objects on opaque pass
      if (cmdList->opacity <= 0.50)
        return;
    }
  } else {
    if (cmdList->opacity < 0.0625)
      return;
  }

  // if we're rendering for Acrobat3D capture, emit materials and other
  // state changes at every opportunity, caching next to nothing by 
  // invalidating materials on every object we draw
  if (ogl_acrobat3dcapture) {
    oglmaterialindex = -1;
    oglambient   = -1;
    ogldiffuse   = -1;
    oglspecular  = -1;
    oglshininess = -1;
    ogloutline = -1;
    ogloutlinewidth = -1;
    ogltransmode = -1;
  } 

  //
  // set the material - only changing those items that have been updated.
  //
  if (oglmaterialindex != cmdList->materialtag) {
    float matbuf[4];
    matbuf[3] = 1.0f; 
    int recalcambientlights = 0;
    int recalcdiffuselights = 0;

    oglmaterialindex = cmdList->materialtag;
    if (oglopacity != cmdList->opacity) {
      oglopacity = cmdList->opacity; // update for next time through

      if (ogl_useblendedtrans) {
        glDisable(GL_POLYGON_STIPPLE);   
        if (oglopacity > 0.999) {
          // disable alpha-blended transparency
          glDisable(GL_BLEND);
        } else {
          glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
          glEnable(GL_BLEND);
        }
      } else {
        // disable alpha-blended transparency
        glDisable(GL_BLEND);

        // use stipple-based transparency
        if (oglopacity > 0.9375) {  
          glDisable(GL_POLYGON_STIPPLE);   
        } else {
          // here's our transparency: opacity < 0.9375  -> transparent
          if (oglopacity > 0.875) 
            glPolygonStipple(ninesixteentone);               
          else if (oglopacity > 0.75) 
            glPolygonStipple(seveneighthtone);               
          else if (oglopacity > 0.5) 
            glPolygonStipple(threequartertone);               
          else if (oglopacity > 0.25)
            glPolygonStipple(halftone);               
          else if (oglopacity > 0.125)
            glPolygonStipple(quartertone);               
          else if (oglopacity > 0.0625)
            glPolygonStipple(eighthtone);               
          else 
            return; // skip rendering the geometry if entirely transparent
    
          glEnable(GL_POLYGON_STIPPLE);                
        }
      }
    }

    if (ogloutline != cmdList->outline) { 
      ogloutline = cmdList->outline;
    }

    if (ogloutlinewidth != cmdList->outlinewidth) { 
      ogloutlinewidth = cmdList->outlinewidth;
    }

    if (ogltransmode != (int) cmdList->transmode) { 
      ogltransmode = (int) cmdList->transmode;
    }

    if (oglambient != cmdList->ambient) { 
      oglambient = cmdList->ambient;
      recalcambientlights = 1;  // force recalculation of ambient lighting
      matbuf[0] = matbuf[1] = matbuf[2] = oglambient; 
      glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, matbuf);
    }

    if (ogldiffuse != cmdList->diffuse) { 
      ogldiffuse = cmdList->diffuse;
      recalcdiffuselights = 1;  // force recalculation of diffuse lighting
      matbuf[0] = matbuf[1] = matbuf[2] = ogldiffuse; 
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, matbuf);
    }

    if (oglspecular != cmdList->specular) { 
      oglspecular = cmdList->specular;
      matbuf[0] = matbuf[1] = matbuf[2] = oglspecular; 
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, matbuf);
    }

    if (oglshininess != cmdList->shininess) {
      oglshininess = cmdList->shininess;
  
      // clamp shininess parameter to what OpenGL 1.x can deal with
      // XXX there are new OpenGL extensions that allow a broader range
      // of Phong exponents.
      glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 
                  (GLfloat) (oglshininess < 128.0f) ? oglshininess : 128.0f);
    }
 
    // change lighting to match new diffuse/ambient factors
    if (recalcambientlights) { 
      for (int z=0; z<DISP_LIGHTS; z++) {
        GLfloat d[4];
        d[0] = ogl_lightcolor[z][0] * oglambient;
        d[1] = ogl_lightcolor[z][1] * oglambient;
        d[2] = ogl_lightcolor[z][2] * oglambient;
        d[3] = 1.0;
        glLightfv((GLenum)(GL_LIGHT0 + z), GL_AMBIENT, d);
      }
    }
 
    if (recalcdiffuselights) { 
      for (int z=0; z<DISP_LIGHTS; z++) {
        GLfloat d[4];
        d[0] = ogl_lightcolor[z][0] * ogldiffuse;
        d[1] = ogl_lightcolor[z][1] * ogldiffuse;
        d[2] = ogl_lightcolor[z][2] * ogldiffuse;
        d[3] = 1.0;
        glLightfv((GLenum)(GL_LIGHT0 + z), GL_DIFFUSE, d);
      }
    }
  }
  // 
  // end material processing code for fixed-function OpenGL pipeline
  //

  // XXX shouldn't be testing this every rep, but for now this works ok 
  ogl_fogmode = 0; // fogmode for shaders

  if (cueingEnabled) {
    switch (cueMode) {
      case CUE_LINEAR: 
        glFogi(GL_FOG_MODE, GL_LINEAR);
        ogl_fogmode = 1;
        break;
  
      case CUE_EXP:
        glFogi(GL_FOG_MODE, GL_EXP);
        ogl_fogmode = 2;
        break;
  
      case CUE_EXP2:
        glFogi(GL_FOG_MODE, GL_EXP2);
        ogl_fogmode = 3;
        break;

      case NUM_CUE_MODES:
        // this should never happen
        break;
    }

    glFogf(GL_FOG_DENSITY, (GLfloat) get_cue_density());
    glFogf(GL_FOG_START,   (GLfloat) get_cue_start());
    glFogf(GL_FOG_END,     (GLfloat) get_cue_end());
  }

#if defined(VMDUSEOPENGLSHADER)
  // setup programmable shader for this object
  if (mainshader) {
    if (ogl_useglslshader) {
      mainshader->UseShader(1); // if glsl is available and enabled, use it
  
      if ((ogl_glslmaterialindex != cmdList->materialtag) || ogl_glsltoggle) {
        ogl_glslmaterialindex = cmdList->materialtag;
        ogl_glsltoggle = 0;
        update_shader_uniforms(mainshader, 1);
      }
    } else {
      mainshader->UseShader(0); // if glsl is available but disabled, turn it off
    }
  }
#endif

  // save transformation matrix
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  multmatrix(cmdList->mat);

  // set up text matrices
  GLfloat textsize = 1;
  Matrix4 textMat(ogl_textMat);
  textMat.multmatrix(cmdList->mat);
  
  // XXX Display list caching begins here
  GLuint ogl_cachedid = 0;    // reset OpenGL display list ID for cached list
  int ogl_cachecreated = 0;  // reset display list creation flag
  int ogl_cacheskip;

  // Disable display list caching if GLSL is enabled or we encounter
  // a non-cacheable representation (such as an animating structure).
  ogl_cacheskip = (cmdList->cacheskip || ogl_useglslshader);

  // enable/disable clipping planes
  for (int cp=0; cp<VMD_MAX_CLIP_PLANE; cp++) {
    // don't cache 'on' state since the parameters will likely differ,
    // just setup the clip plane from the new state
    if (cmdList->clipplanes[cp].mode) {
      GLdouble cpeq[4];
      cpeq[0] = cmdList->clipplanes[cp].normal[0];
      cpeq[1] = cmdList->clipplanes[cp].normal[1];
      cpeq[2] = cmdList->clipplanes[cp].normal[2];
  
      // Convert specification to OpenGL plane equation
      cpeq[3] = 
      -(cmdList->clipplanes[cp].normal[0] * cmdList->clipplanes[cp].center[0] +
        cmdList->clipplanes[cp].normal[1] * cmdList->clipplanes[cp].center[1] +
        cmdList->clipplanes[cp].normal[2] * cmdList->clipplanes[cp].center[2]);
      glClipPlane((GLenum) (GL_CLIP_PLANE0 + cp), cpeq);
      glEnable((GLenum) (GL_CLIP_PLANE0 + cp)); 

      // XXX if the clipping plane mode is set for rendering
      // capped clipped solids, we will have to perform several
      // rendering passes using the stencil buffer and Z-buffer 
      // in order to get the desired results.
      // http://www.nigels.com/research/wscg2002.pdf 
      // http://citeseer.ist.psu.edu/stewart02lineartime.html
      // http://citeseer.ist.psu.edu/stewart98improved.html
      // http://www.sgi.com/software/opengl/advanced97/notes/node10.html
      // http://www.opengl.org/resources/tutorials/sig99/advanced99/notes/node21.html
      // http://www.ati.com/developer/sdk/rage128sdk/OpenGL/Samples/Rage128StencilCap.html
      // The most common algorithm goes something like what is described here:
      //   0) clear stencil/color/depth buffers
      //   1) disable color buffer writes
      //   2) render clipping plane polygon writing to depth buffer
      //   3) disable depth buffer writes
      //   4) set stencil op to increment when depth test passes
      //   5) draw molecule with glCullFace(GL_BACK)
      //   6) set stencil op to decrement when depth test passes
      //   7) draw molecule with glCullFace(GL_FRONT)
      //   8) clear depth buffer
      //   9) enable color buffer writes 
      //  10) set stencil function to GL_EQUAL of 1
      //  11) draw clipping plane polygon with appropriate materials
      //  12) disable stencil buffer
      //  13) enable OpenGL clipping plane
      //  14) draw molecule
    } else {
      // if its already off, no need to disable it again.
      if (ogl_clipmode[cp] != cmdList->clipplanes[cp].mode) {
        glDisable((GLenum) (GL_CLIP_PLANE0 + cp)); 
      }
    }

    // update clip mode cache
    ogl_clipmode[cp] = cmdList->clipplanes[cp].mode;
  }

  // initialize text offset variables.  These values should never be set in one
  // display list and applied in another, so we make them local variables here
  // rather than OpenGLRenderer state variables.
  float textoffset_x = 0, textoffset_y = 0;

  // Compute periodic image transformation matrices
  ResizeArray<Matrix4> pbcImages;
  find_pbc_images(cmdList, pbcImages);
  int npbcimages = pbcImages.num();

  // Retreive instance image transformation matrices
  ResizeArray<Matrix4> instanceImages;
  find_instance_images(cmdList, instanceImages);
  int ninstances = instanceImages.num();

for (int pbcimage = 0; pbcimage < npbcimages; pbcimage++) {
  glPushMatrix();
  multmatrix(pbcImages[pbcimage]);

for (int instanceimage = 0; instanceimage < ninstances; instanceimage++) {
  glPushMatrix();
  multmatrix(instanceImages[instanceimage]);

  if (ogl_cachedebug) {
    msgInfo << "Rendering scene: cache enable=" << ogl_cacheenabled 
            << ", created=" << ogl_cachecreated << ", serial=" << (int)cmdList->serial
            << ", id=" << (int)ogl_cachedid << ", skip=" << ogl_cacheskip << sendmsg;
  }

  // find previously cached display list for this object
  if (ogl_cacheenabled && !ogl_cacheskip) {
    ogl_cachedid = displaylistcache.markUsed(cmdList->serial);

    // add to the cache and regenerate if we didn't find it
    if (ogl_cachedid == GLCACHE_FAIL) {
      ogl_cachedid = glGenLists(1);      
      displaylistcache.encache(cmdList->serial, ogl_cachedid);

      // create the display list, and execute it.
      glNewList(ogl_cachedid, GL_COMPILE_AND_EXECUTE);
      ogl_cachecreated = 1; // a new display list was created 
    } 
  }

  // XXX Draw OpenGL geometry only when caching is disabled or when
  //     we have new geometry to cache
  if ((!ogl_cacheenabled) || ogl_cacheskip || (ogl_cacheenabled && ogl_cachecreated)) {

  // scan through the list, getting each command and executing it, until
  // the end of commands token is found
  VMDDisplayList::VMDLinkIter cmditer;
  cmdList->first(&cmditer);
  while((tok = cmdList->next(&cmditer, cmdptr)) != DLASTCOMMAND) {
    OGLERR // enable OpenGL debugging code

    switch (tok) {
      case DPOINT:
        // plot a point at the given position
        glBegin(GL_POINTS);
          glVertex3fv(((DispCmdPoint *)cmdptr)->pos);
        glEnd();
        break;

      case DPOINTARRAY: 
        {
          DispCmdPointArray *pa = (DispCmdPointArray *)cmdptr;
          float *centers;
          float *colors;
          pa->getpointers(centers, colors);
#if defined(GL_VERSION_1_1)
        if (!(simplegraphics || ogl_acrobat3dcapture)) {
          // Vertex array implementation 
          glDisable(GL_LIGHTING); 
          ogl_lightingenabled=0;
          glEnableClientState(GL_VERTEX_ARRAY);
          glEnableClientState(GL_COLOR_ARRAY);
          glDisableClientState(GL_NORMAL_ARRAY);
          glVertexPointer(3, GL_FLOAT, 12, (void *) centers);
          glColorPointer(3, GL_FLOAT, 12, (void *)  colors);

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLLOCKARRAYSEXT(0, pa->numpoints);
          }
#endif

          // set point size, enable blending and point antialiasing
          glPointSize(pa->size); 
          glEnable(GL_POINT_SMOOTH);
          glEnable(GL_BLEND);
          glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

#if defined(VMDUSEGLSLSPHERESPRITES) && defined(GL_ARB_point_sprite)
          // XXX enable point sprites
          if (ext->hasglpointspritearb &&
              spherespriteshader && ogl_useglslshader) {
            glEnable(GL_POINT_SPRITE_ARB);
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
            mainshader->UseShader(0);   // switch to sphere shader
            spherespriteshader->UseShader(1); // switch to sphere sprite shader
            update_shader_uniforms(spherespriteshader, 1); // force update of shader

            // define sprite size in pixels
            GLint loc;
            loc = GLGETUNIFORMLOCATIONARB(spherespriteshader->ProgramObject,
                                          "vmdspritesize");
            GLfloat sz = pa->size;
            GLUNIFORM1FVARB(loc, 1, &sz);

            // Specify point sprite texture coordinate replacement mode
            glPushAttrib(GL_TEXTURE_BIT);
            glTexEnvf(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
            OGLERR;
          }
#endif

#if defined(GL_ARB_point_parameters) 
          int dodepthscaling = 0;
  
          // enable distance based point attenuation
          if (ext->hasglpointparametersext  && (projection() == PERSPECTIVE)) {
            dodepthscaling = 1;
  
            GLfloat abc[4] = {0.0, 0.0, 1.0};
            GLPOINTPARAMETERFVARB(GL_POINT_DISTANCE_ATTENUATION_ARB, abc);
          }
#endif

          glDrawArrays(GL_POINTS, 0, pa->numpoints);

#if defined(GL_ARB_point_parameters) 
          // disable distance based point attenuation
          if (dodepthscaling) {
            GLfloat abc[4] = {1.0, 0.0, 0.0};
            GLPOINTPARAMETERFVARB(GL_POINT_DISTANCE_ATTENUATION_ARB, abc);
          }
#endif

          // disable blending and point antialiasing
          glDisable(GL_BLEND);
          glDisable(GL_POINT_SMOOTH);

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLUNLOCKARRAYSEXT();
          }
#endif

#if defined(VMDUSEGLSLSPHERESPRITES) && defined(GL_ARB_point_sprite)
          // XXX enable point sprites
          if (ext->hasglpointspritearb &&
              spherespriteshader && ogl_useglslshader) {
            glPopAttrib(); // return previous texturing state
            glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
            glDisable(GL_POINT_SPRITE_ARB);
            spherespriteshader->UseShader(0); // switch back to the main shader
            mainshader->UseShader(1);   // switch back to the main shader
            OGLERR;
          }
#endif

          glEnableClientState(GL_NORMAL_ARRAY);
          glPointSize(1.0); // reset point size to default
        } else {
#endif
          // Immediate mode implementation
          int i, ind;
          glBegin(GL_POINTS);
          ind = 0;
          for (i=0; i<pa->numpoints; i++) {
            glColor3fv(&colors[ind]);
            glVertex3fv(&centers[ind]); 
            ind += 3;
          }
          glEnd();
#if defined(GL_VERSION_1_1)
        }
#endif
        }
        break;

      case DLITPOINTARRAY: 
        {
        DispCmdLitPointArray *pa = (DispCmdLitPointArray *)cmdptr;
        float *centers;
        float *normals;
        float *colors;
        pa->getpointers(centers, normals, colors);
#if defined(GL_VERSION_1_1)
        if (!(simplegraphics || ogl_acrobat3dcapture)) {
          // Vertex array implementation 
          glEnableClientState(GL_VERTEX_ARRAY);
          glEnableClientState(GL_COLOR_ARRAY);
          glEnableClientState(GL_NORMAL_ARRAY);
          glVertexPointer(3, GL_FLOAT, 12, (void *) centers);
          glNormalPointer(GL_FLOAT, 12, (void *) normals);
          glColorPointer(3, GL_FLOAT, 12, (void *)  colors);

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLLOCKARRAYSEXT(0, pa->numpoints);
          }
#endif

          // set point size, enable blending and point antialiasing
          glPointSize(pa->size); 
          glEnable(GL_POINT_SMOOTH);

#if defined(GL_ARB_point_parameters) 
          int dodepthscaling = 0;
          // enable distance based point attenuation
          if (ext->hasglpointparametersext  && (projection() == PERSPECTIVE)) {
            dodepthscaling = 1;
            GLfloat abc[4] = {0.0, 0.0, 1.0};
            GLPOINTPARAMETERFVARB(GL_POINT_DISTANCE_ATTENUATION_ARB, abc);
          }
#endif

          glDrawArrays(GL_POINTS, 0, pa->numpoints);

#if defined(GL_ARB_point_parameters) 
          // disable distance based point attenuation
          if (dodepthscaling) {
            GLfloat abc[4] = {1.0, 0.0, 0.0};
            GLPOINTPARAMETERFVARB(GL_POINT_DISTANCE_ATTENUATION_ARB, abc);
          }
#endif

          // disable blending and point antialiasing
          glDisable(GL_BLEND);
          glDisable(GL_POINT_SMOOTH);

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLUNLOCKARRAYSEXT();
          }
#endif

          glPointSize(1.0); // reset point size to default
        } else {
#endif
          // Immediate mode implementation 
          int i, ind;
          glBegin(GL_POINTS);
          ind = 0;
          for (i=0; i<pa->numpoints; i++) {
            glColor3fv(&colors[ind]);
            glNormal3fv(&normals[ind]); 
            glVertex3fv(&centers[ind]); 
            ind += 3;
          }
          glEnd();
#if defined(GL_VERSION_1_1)
        }
#endif
        }
        break;

      case DLINE:
        // plot a line
        glBegin(GL_LINES);
          glVertex3fv(((DispCmdLine *)cmdptr)->pos1);
          glVertex3fv(((DispCmdLine *)cmdptr)->pos2);
        glEnd();
        break;

      case DLINEARRAY:
        {
          float *v = (float *)(cmdptr);
          int nlines = (int)v[0];
          v++; // move pointer forward before traversing vertex data

#if defined(GL_VERSION_1_1)
          if (!(simplegraphics || ogl_acrobat3dcapture)) {
            // Vertex array implementation
            glInterleavedArrays(GL_V3F, 0, v);

#if defined(GL_EXT_compiled_vertex_array) 
            if (ext->hascompiledvertexarrayext) {
              GLLOCKARRAYSEXT(0, 2*nlines);
            }
#endif

            glDrawArrays(GL_LINES, 0, 2*nlines); 

#if defined(GL_EXT_compiled_vertex_array) 
            if (ext->hascompiledvertexarrayext) {
              GLUNLOCKARRAYSEXT();
            }
#endif
          } else {
#endif
            // Immediate mode implementation
            glBegin(GL_LINES);
            for (int i=0; i<nlines; i++) {
              glVertex3fv(v);
              glVertex3fv(v+3);
              v += 6;
            }
            glEnd();
#if defined(GL_VERSION_1_1)
          }
#endif
        }
        break;    

      case DPOLYLINEARRAY:
        {
          float *v = (float *)(cmdptr);
          int nverts = (int)v[0];
          v++; // move pointer forward before traversing vertex data

#if defined(GL_VERSION_1_1)
          if (!(simplegraphics || ogl_acrobat3dcapture)) {
            // Vertex array implementation
            glInterleavedArrays(GL_V3F, 0, v);

#if defined(GL_EXT_compiled_vertex_array) 
            if (ext->hascompiledvertexarrayext) {
              GLLOCKARRAYSEXT(0, nverts);
            }
#endif

            glDrawArrays(GL_LINE_STRIP, 0, nverts); 

#if defined(GL_EXT_compiled_vertex_array) 
            if (ext->hascompiledvertexarrayext) {
              GLUNLOCKARRAYSEXT();
            }
#endif
          } else {
#endif
            // Immediate mode implementation
            glBegin(GL_LINE_STRIP);
            for (int i=0; i<nverts; i++) {
              glVertex3fv(v);
              v += 3;
            }
            glEnd();
#if defined(GL_VERSION_1_1)
          }
#endif
        }
        break;    

      case DSPHERE: 
        {
        float *p = (float *)cmdptr;
        glPushMatrix();
        glTranslatef(p[0], p[1], p[2]); 
        glScalef(p[3], p[3], p[3]);
        glCallList(SphereList);
        glPopMatrix();
        }
        break;

      case DSPHEREARRAY: 
        {
        DispCmdSphereArray *sa = (DispCmdSphereArray *)cmdptr;
        int i, ind;
        float * centers;
        float * radii;
        float * colors;
        sa->getpointers(centers, radii, colors);

#if defined(VMDUSEGLSLSPHERES) 
        // Render the sphere using programmable shading via ray-casting
        if (sphereshader && ogl_useglslshader) {
          // coordinates of unit bounding box
          GLfloat v0[] = {-1.0, -1.0, -1.0}; 
          GLfloat v1[] = { 1.0, -1.0, -1.0}; 
          GLfloat v2[] = {-1.0,  1.0, -1.0}; 
          GLfloat v3[] = { 1.0,  1.0, -1.0}; 
          GLfloat v4[] = {-1.0, -1.0,  1.0}; 
          GLfloat v5[] = { 1.0, -1.0,  1.0}; 
          GLfloat v6[] = {-1.0,  1.0,  1.0}; 
          GLfloat v7[] = { 1.0,  1.0,  1.0}; 
      
          mainshader->UseShader(0);   // switch to sphere shader
          sphereshader->UseShader(1); // switch to sphere shader
          update_shader_uniforms(sphereshader, 1); // force update of shader

          // Update projection parameters for OpenGL shader
          GLfloat projparms[4];
          projparms[0] = nearClip;
          projparms[1] = farClip; 
          projparms[2] = 0.5f * (farClip + nearClip);
          projparms[3] = 1.0f / (farClip - nearClip);
          GLint projloc = GLGETUNIFORMLOCATIONARB(sphereshader->ProgramObject, "vmdprojparms");
          GLUNIFORM4FVARB(projloc, 1, projparms);
          OGLERR;

          ind = 0;
          for (i=0; i<sa->numspheres; i++) {
            glPushMatrix();
            glTranslatef(centers[ind], centers[ind + 1], centers[ind + 2]); 
            glScalef(radii[i], radii[i], radii[i]);
            glColor3fv(&colors[ind]);

            // Draw the bounding box containing the sphere, gauranteeing 
            // that it will be correctly rendered regardless of the 
            // perspective projection used, viewing direction, etc.
            // If enough is known about the projection being used, this
            // could be done with simple billboard polygons, or perhaps even
            // a large OpenGL point primitive instead of a whole cube
            glBegin(GL_QUADS);
              glVertex3fv((GLfloat *) v0); /* -Z face */
              glVertex3fv((GLfloat *) v1);
              glVertex3fv((GLfloat *) v3);
              glVertex3fv((GLfloat *) v2);

              glVertex3fv((GLfloat *) v4); /* +Z face */
              glVertex3fv((GLfloat *) v5);
              glVertex3fv((GLfloat *) v7);
              glVertex3fv((GLfloat *) v6);

              glVertex3fv((GLfloat *) v0); /* -Y face */
              glVertex3fv((GLfloat *) v1);
              glVertex3fv((GLfloat *) v5);
              glVertex3fv((GLfloat *) v4);

              glVertex3fv((GLfloat *) v2); /* +Y face */
              glVertex3fv((GLfloat *) v3);
              glVertex3fv((GLfloat *) v7);
              glVertex3fv((GLfloat *) v6);

              glVertex3fv((GLfloat *) v0); /* -X face */
              glVertex3fv((GLfloat *) v2);
              glVertex3fv((GLfloat *) v6);
              glVertex3fv((GLfloat *) v4);

              glVertex3fv((GLfloat *) v1); /* +X face */
              glVertex3fv((GLfloat *) v3);
              glVertex3fv((GLfloat *) v7);
              glVertex3fv((GLfloat *) v5);
            glEnd();
            glPopMatrix();
            ind += 3; // next sphere
          }

          sphereshader->UseShader(0); // switch back to the main shader
          mainshader->UseShader(1);   // switch back to the main shader
          OGLERR;
        } else {
#endif
          // OpenGL display listed sphere rendering implementation
          set_sphere_res(sa->sphereres); // set the current sphere resolution

          // use single-sided lighting when drawing spheres for 
          // peak rendering speed.
          glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
          ind = 0;
          for (i=0; i<sa->numspheres; i++) {
            glPushMatrix();
            glTranslatef(centers[ind], centers[ind + 1], centers[ind + 2]); 
            glScalef(radii[i], radii[i], radii[i]);
            glColor3fv(&colors[ind]);
            glCallList(SphereList);
            glPopMatrix();
            ind += 3; // next sphere
          }
          glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
#if defined(VMDUSEGLSLSPHERES)
        }
#endif

        }
        break;

      case DCUBEARRAY: 
        {
          DispCmdLatticeCubeArray *ca = (DispCmdLatticeCubeArray *)cmdptr;
          int i, ind;
          float * centers;
          float * radii;
          float * colors;
          ca->getpointers(centers, radii, colors);

          // Render the cube 
          // coordinates of unit cube
          GLfloat v0[] = {-1.0, -1.0, -1.0}; 
          GLfloat v1[] = { 1.0, -1.0, -1.0}; 
          GLfloat v2[] = {-1.0,  1.0, -1.0}; 
          GLfloat v3[] = { 1.0,  1.0, -1.0}; 
          GLfloat v4[] = {-1.0, -1.0,  1.0}; 
          GLfloat v5[] = { 1.0, -1.0,  1.0}; 
          GLfloat v6[] = {-1.0,  1.0,  1.0}; 
          GLfloat v7[] = { 1.0,  1.0,  1.0}; 
      
          ind = 0;
          for (i=0; i<ca->numcubes; i++) {
            glPushMatrix();
            glTranslatef(centers[ind], centers[ind + 1], centers[ind + 2]); 
            glScalef(radii[i], radii[i], radii[i]);
            glColor3fv(&colors[ind]);

            // Draw the unit cube
            glBegin(GL_QUADS);
              glNormal3f(0.0f, 0.0f, 1.0f);
              glVertex3fv((GLfloat *) v0); /* -Z face */
              glVertex3fv((GLfloat *) v1);
              glVertex3fv((GLfloat *) v3);
              glVertex3fv((GLfloat *) v2);

              glNormal3f(0.0f, 0.0f, 1.0f);
              glVertex3fv((GLfloat *) v4); /* +Z face */
              glVertex3fv((GLfloat *) v5);
              glVertex3fv((GLfloat *) v7);
              glVertex3fv((GLfloat *) v6);

              glNormal3f(0.0f, -1.0f, 0.0f);
              glVertex3fv((GLfloat *) v0); /* -Y face */
              glVertex3fv((GLfloat *) v1);
              glVertex3fv((GLfloat *) v5);
              glVertex3fv((GLfloat *) v4);

              glNormal3f(0.0f, -1.0f, 0.0f);
              glVertex3fv((GLfloat *) v2); /* +Y face */
              glVertex3fv((GLfloat *) v3);
              glVertex3fv((GLfloat *) v7);
              glVertex3fv((GLfloat *) v6);

              glNormal3f(1.0f, 0.0f, 0.0f);
              glVertex3fv((GLfloat *) v0); /* -X face */
              glVertex3fv((GLfloat *) v2);
              glVertex3fv((GLfloat *) v6);
              glVertex3fv((GLfloat *) v4);

              glNormal3f(1.0f, 0.0f, 0.0f);
              glVertex3fv((GLfloat *) v1); /* +X face */
              glVertex3fv((GLfloat *) v3);
              glVertex3fv((GLfloat *) v7);
              glVertex3fv((GLfloat *) v5);
            glEnd();
            glPopMatrix();
            ind += 3; // next sphere
          }
          OGLERR;
        }
        break;

      case DTRIANGLE: 
        {
        DispCmdTriangle *cmd = (DispCmdTriangle *)cmdptr;
        glBegin(GL_TRIANGLES);
          glNormal3fv(cmd->norm1);
          glVertex3fv(cmd->pos1);
          glNormal3fv(cmd->norm2);
          glVertex3fv(cmd->pos2);
          glNormal3fv(cmd->norm3);
          glVertex3fv(cmd->pos3);
        glEnd();
        }
        break;

      case DSQUARE:
        // draw a square, given the four points
        {
        DispCmdSquare *cmd = (DispCmdSquare *)cmdptr;
        glBegin(GL_QUADS);
          glNormal3fv((GLfloat *) cmd->norml);
          glVertex3fv((GLfloat *) cmd->pos1);
          glVertex3fv((GLfloat *) cmd->pos2);
          glVertex3fv((GLfloat *) cmd->pos3);
          glVertex3fv((GLfloat *) cmd->pos4);
        glEnd();
        }
        break;

#if 0
      case DSTRIPETEX:
        if (ext->hastex3d) {
#if defined(GL_VERSION_1_2)
#define STRIPEWIDTH 32
          GLubyte stripeImage[4 * STRIPEWIDTH];
          GLuint texName = 0;
          // glGenTextures(1, &texName);
          int i;
          for (i=0; i<STRIPEWIDTH; i++) {
            stripeImage[4*i    ] = (GLubyte) ((i>4) ? 255 : 0); // R
            stripeImage[4*i + 1] = (GLubyte) ((i>4) ? 255 : 0); // G
            stripeImage[4*i + 2] = (GLubyte) ((i>4) ? 255 : 0); // B
            stripeImage[4*i + 3] = (GLubyte) 255;               // W
          }

          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
          glBindTexture(GL_TEXTURE_1D, texName);
          glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
          glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_REPEAT);
          glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_R, GL_REPEAT);
          glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, STRIPEWIDTH, 
                          0, GL_RGBA, GL_UNSIGNED_BYTE, stripeImage);

          // XXX should use GL_MODULATE, but requires all polygons to be
          //     drawn "white", in order for shading to make it through the
          //     texturing process.  GL_REPLACE works well for situations
          //     where we want coloring to come entirely from texture.
          glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
          GLfloat xplaneeq[4] = {0.5, 0.0, 0.0, 0.0};
          glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
          glTexGenfv(GL_S, GL_EYE_PLANE, xplaneeq);
          glEnable(GL_TEXTURE_GEN_S);
          glEnable(GL_TEXTURE_1D);
#endif
        }
        break;

      case DSTRIPETEXOFF:
        if (ext->hastex3d) {
#if defined(GL_VERSION_1_2)
          glDisable(GL_TEXTURE_GEN_S);
          glDisable(GL_TEXTURE_1D);
#endif
        }
        break;
#endif

      case DVOLUMETEXTURE:
        if (ext->hastex3d)
#if defined(GL_VERSION_1_2)
        {
  
#if defined(GL_GENERATE_MIPMAP_HINT)
          // set MIP map generation hint for high quality
          glHint(GL_GENERATE_MIPMAP_HINT, GL_NICEST);
#endif
          glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

          DispCmdVolumeTexture *cmd = (DispCmdVolumeTexture *)cmdptr;
          require_volume_texture(cmd->ID, 
              cmd->xsize, cmd->ysize, cmd->zsize, 
              cmd->texmap);

          GLfloat xplaneeq[4]; 
          GLfloat yplaneeq[4]; 
          GLfloat zplaneeq[4]; 
          int i;

          glEnable(GL_TEXTURE_3D);
          glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

#if defined(VMDUSEOPENGLSHADER)
          // Update active GLSL texturing mode
          if (mainshader && ogl_useglslshader) {
            if (!ogl_lightingenabled)
              mainshader->UseShader(1); // enable shader so state updates 
            ogl_glsltexturemode = 1;
            GLint vmdtexturemode = 1;  // enable 3-D texturing->MODULATE
            GLint loc = GLGETUNIFORMLOCATIONARB(mainshader->ProgramObject, "vmdtexturemode");
            GLUNIFORM1IARB(loc, vmdtexturemode);
      
            // Set active texture map index
            loc = GLGETUNIFORMLOCATIONARB(mainshader->ProgramObject, "vmdtex0");
            GLUNIFORM1IARB(loc, 0); // using texture unit 0
            if (!ogl_lightingenabled)
              mainshader->UseShader(0); // disable shader after state updates
          }
#endif

          // automatically generate texture coordinates by translating from
          // model coordinate space to volume coordinates.  These aren't
          // going to be used by volume slices, but the performance hit
          // is expected to be insignificant.
          for (i=0; i<3; i++) {
            xplaneeq[i] = cmd->v1[i];
            yplaneeq[i] = cmd->v2[i];
            zplaneeq[i] = cmd->v3[i];
          }
          xplaneeq[3] = cmd->v0[0];
          yplaneeq[3] = cmd->v0[1];
          zplaneeq[3] = cmd->v0[2];

          glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR); 
          glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR); 
          glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR); 
          glTexGenfv(GL_S, GL_EYE_PLANE, xplaneeq);
          glTexGenfv(GL_T, GL_EYE_PLANE, yplaneeq);
          glTexGenfv(GL_R, GL_EYE_PLANE, zplaneeq);
          glEnable(GL_TEXTURE_GEN_S);
          glEnable(GL_TEXTURE_GEN_T);
          glEnable(GL_TEXTURE_GEN_R);
#endif
        }
        break;

      case DVOLTEXON:
        if (ext->hastex3d) {
#if defined(GL_VERSION_1_2)
          glEnable(GL_TEXTURE_3D);     // enable volume texturing
#if defined(VMDUSEOPENGLSHADER)
          // Update active GLSL texturing mode
          if (mainshader && ogl_useglslshader) {
            if (!ogl_lightingenabled)
              mainshader->UseShader(1); // enable shader so state updates 
            ogl_glsltexturemode = 1;
            GLint vmdtexturemode = 1;  // enable 3-D texturing->MODULATE
            GLint loc = GLGETUNIFORMLOCATIONARB(mainshader->ProgramObject, "vmdtexturemode");
            GLUNIFORM1IARB(loc, vmdtexturemode);
            if (!ogl_lightingenabled)
              mainshader->UseShader(0); // disable shader after state updates
          }
#endif
          glEnable(GL_TEXTURE_GEN_S);  // enable automatic texture 
          glEnable(GL_TEXTURE_GEN_T);  //   coordinate generation
          glEnable(GL_TEXTURE_GEN_R);
#endif
        }
        break;

      case DVOLTEXOFF:
        if (ext->hastex3d) {
#if defined(GL_VERSION_1_2)
          glDisable(GL_TEXTURE_3D);     // disable volume texturing
#if defined(VMDUSEOPENGLSHADER)
          // Update active GLSL texturing mode
          if (mainshader && ogl_useglslshader) {
            if (!ogl_lightingenabled)
              mainshader->UseShader(1); // enable shader so state updates 
            ogl_glsltexturemode = 0;
            GLint vmdtexturemode = 0;  // disable 3-D texturing
            GLint loc = GLGETUNIFORMLOCATIONARB(mainshader->ProgramObject, "vmdtexturemode");
            GLUNIFORM1IARB(loc, vmdtexturemode);
            if (!ogl_lightingenabled)
              mainshader->UseShader(0); // disable shader after state updates
          }
#endif

          glDisable(GL_TEXTURE_GEN_S);  // disable automatic texture 
          glDisable(GL_TEXTURE_GEN_T);  //   coordinate generation
          glDisable(GL_TEXTURE_GEN_R);
#endif
        }
        break;


      case DVOLSLICE:
        if (ext->hastex3d) {
          DispCmdVolSlice *cmd = (DispCmdVolSlice *)cmdptr;
#if defined(GL_VERSION_1_2)

          // DVOLUMETEXTURE does most of the work for us, but we override
          // a few of the texenv settings
          glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

          // enable or disable texture interpolation and filtering
          switch (cmd->texmode) {
            case 2:
              glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
              glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
              break;

            case 1:
              glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_DONT_CARE);
              glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
              break;
     
            case 0: 
            default:
              glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
              glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
              glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
              break;
          }
      
          // use the texture edge colors rather border color when wrapping
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

#if defined(VMDUSEOPENGLSHADER)
          // Update active GLSL texturing mode
          if (mainshader && ogl_useglslshader) {
            ogl_glsltexturemode = 2;
            GLint vmdtexturemode = 2;  // enable 3-D texturing->REPLACE
            GLint loc = GLGETUNIFORMLOCATIONARB(mainshader->ProgramObject, "vmdtexturemode");
            GLUNIFORM1IARB(loc, vmdtexturemode);
          }
#endif
          glBegin(GL_QUADS);        
          for (int i=0; i<4; i++) {
            glNormal3fv(cmd->normal);
            glVertex3fv(cmd->v + 3*i);
          }
          glEnd();        
#endif // GL_VERSION_1_2
        } 
        break;

      case DTRIMESH_C3F_N3F_V3F: 
        { 
        // draw a triangle mesh
        DispCmdTriMesh *cmd = (DispCmdTriMesh *) cmdptr;
        float *colors=NULL, *normals=NULL, *vertices=NULL;

        if (cmd->pervertexcolors)
          cmd->getpointers(colors, normals, vertices);
        else
          cmd->getpointers(normals, vertices);

#if 1
#if defined(GL_VERSION_1_1)
        if (!(simplegraphics || ogl_acrobat3dcapture)) {
          // Vertex array implementation
          if (cmd->pervertexcolors)
            glEnableClientState(GL_COLOR_ARRAY);
          else
            glDisableClientState(GL_COLOR_ARRAY);
          glEnableClientState(GL_NORMAL_ARRAY);
          glEnableClientState(GL_VERTEX_ARRAY);

          if (cmd->pervertexcolors)
            glColorPointer(3, GL_FLOAT, 0, (void *) colors);
          glNormalPointer(GL_FLOAT, 0, (void *) normals);
          glVertexPointer(3, GL_FLOAT, 0, (void *) vertices);

#if defined(GL_EXT_compiled_vertex_array)
          if (ext->hascompiledvertexarrayext) {
            GLLOCKARRAYSEXT(0, cmd->numverts);
          }
#endif

          glDrawArrays(GL_TRIANGLES, 0, cmd->numverts);

#if defined(GL_EXT_compiled_vertex_array)
          if (ext->hascompiledvertexarrayext) {
            GLUNLOCKARRAYSEXT();
          }
#endif
        } else {
#endif
          // Immediate mode implementation
          int i, ind;
          glBegin(GL_TRIANGLES);
          ind = 0;
          if (cmd->pervertexcolors) {
            for (i=0; i<cmd->numverts; i++) {
              glColor3fv(&colors[ind]);
              glNormal3fv(&normals[ind]);
              glVertex3fv(&vertices[ind]);
              ind += 3;
            }
          } else {
            for (i=0; i<cmd->numverts; i++) {
              glNormal3fv(&normals[ind]);
              glVertex3fv(&vertices[ind]);
              ind += 3;
            }
          }

          glEnd();
#if defined(GL_VERSION_1_1)
        }
#endif

#endif
        }
        break;

      case DTRIMESH_C4F_N3F_V3F: 
        {
        // draw a triangle mesh
        DispCmdTriMesh *cmd = (DispCmdTriMesh *) cmdptr;
        int ind = cmd->numfacets * 3;
        float *cnv;
        int *f;
        cmd->getpointers(cnv, f);

#if defined(GL_VERSION_1_1)
        // Vertex array implementation
        if (!(simplegraphics || ogl_acrobat3dcapture)) {
          // If OpenGL 1.1, then use vertex arrays 
          glInterleavedArrays(GL_C4F_N3F_V3F, 0, cnv);

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLLOCKARRAYSEXT(0, cmd->numverts);
          }
#endif

          glDrawElements(GL_TRIANGLES, ind, GL_UNSIGNED_INT, f);

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLUNLOCKARRAYSEXT();
          }
#endif
        } else {
#endif

          // simple graphics mode, but not Acrobat3D capture mode
          if (!ogl_acrobat3dcapture) {
            int i, ind2;
            glBegin(GL_TRIANGLES);
            for (i=0; i<ind; i++) {
              ind2 = f[i] * 10;
               glColor3fv(cnv + ind2    );
              glNormal3fv(cnv + ind2 + 4);
              glVertex3fv(cnv + ind2 + 7);
            }
            glEnd();
          } else { 
            // Version 7.0.9 of Acrobat3D can't capture multicolor
            // triangles, so we revert to averaged-single-color
            // triangles until they fix this capture bug.
            int i;
            for (i=0; i<cmd->numfacets; i++) {
              int ind = i * 3;
              float tmp[3], tmp2[3];

              int v0 = f[ind    ] * 10;
              int v1 = f[ind + 1] * 10;
              int v2 = f[ind + 2] * 10;

              vec_add(tmp, cnv + v0, cnv + v1);
              vec_add(tmp2, tmp, cnv + v2);
              vec_scale(tmp, 0.3333333f, tmp2);
              glBegin(GL_TRIANGLES);
              glColor3fv(tmp);
              glNormal3fv(cnv + v0 + 4);
              glVertex3fv(cnv + v0 + 7);
              glNormal3fv(cnv + v1 + 4);
              glVertex3fv(cnv + v1 + 7);
              glNormal3fv(cnv + v2 + 4);
              glVertex3fv(cnv + v2 + 7);
              glEnd();
            }
          }

#if defined(GL_VERSION_1_1)
        }
#endif
        }
        break;


      case DTRIMESH_C4U_N3F_V3F: 
        { 
        // draw a triangle mesh
        DispCmdTriMesh *cmd = (DispCmdTriMesh *) cmdptr;
        unsigned char *colors=NULL;
        float *normals=NULL, *vertices=NULL;

        if (cmd->pervertexcolors)
          cmd->getpointers(colors, normals, vertices);
        else
          cmd->getpointers(normals, vertices);

#if 1
#if defined(GL_VERSION_1_1)
        if (!(simplegraphics || ogl_acrobat3dcapture)) {
          // Vertex array implementation
          if (cmd->pervertexcolors)
            glEnableClientState(GL_COLOR_ARRAY);
          else
            glDisableClientState(GL_COLOR_ARRAY);
          glEnableClientState(GL_NORMAL_ARRAY);
          glEnableClientState(GL_VERTEX_ARRAY);

          if (cmd->pervertexcolors)
            glColorPointer(4, GL_UNSIGNED_BYTE, 0, (void *) colors);
          glNormalPointer(GL_FLOAT, 0, (void *) normals);
          glVertexPointer(3, GL_FLOAT, 0, (void *) vertices);

#if defined(GL_EXT_compiled_vertex_array)
          if (ext->hascompiledvertexarrayext) {
            GLLOCKARRAYSEXT(0, cmd->numverts);
          }
#endif

          glDrawArrays(GL_TRIANGLES, 0, cmd->numverts);

#if defined(GL_EXT_compiled_vertex_array)
          if (ext->hascompiledvertexarrayext) {
            GLUNLOCKARRAYSEXT();
          }
#endif
        } else {
#endif
          // Immediate mode implementation
          int i, ind;
          glBegin(GL_TRIANGLES);
          ind = 0;
          if (cmd->pervertexcolors) {
            for (i=0; i<cmd->numverts; i++) {
              glColor3ubv(&colors[ind]);
              glNormal3fv(&normals[ind]);
              glVertex3fv(&vertices[ind]);
              ind += 3;
            }
          } else {
            for (i=0; i<cmd->numverts; i++) {
              glNormal3fv(&normals[ind]);
              glVertex3fv(&vertices[ind]);
              ind += 3;
            }
          }

          glEnd();
#if defined(GL_VERSION_1_1)
        }
#endif

#endif
        }
        break;


      case DTRIMESH_C4U_N3B_V3F: 
        { 
        // draw a triangle mesh
        DispCmdTriMesh *cmd = (DispCmdTriMesh *) cmdptr;
        unsigned char *colors=NULL;
        char *normals=NULL;
        float *vertices=NULL;

        if (cmd->pervertexcolors)
          cmd->getpointers(colors, normals, vertices);
        else
          cmd->getpointers(normals, vertices);

#if 1
#if defined(GL_VERSION_1_1)
        if (!(simplegraphics || ogl_acrobat3dcapture)) {
          // Vertex array implementation
          if (cmd->pervertexcolors)
            glEnableClientState(GL_COLOR_ARRAY);
          else
            glDisableClientState(GL_COLOR_ARRAY);
          glEnableClientState(GL_NORMAL_ARRAY);
          glEnableClientState(GL_VERTEX_ARRAY);

          if (cmd->pervertexcolors)
            glColorPointer(4, GL_UNSIGNED_BYTE, 0, (void *) colors);
          glNormalPointer(GL_BYTE, 0, (void *) normals);
          glVertexPointer(3, GL_FLOAT, 0, (void *) vertices);

#if defined(GL_EXT_compiled_vertex_array)
          if (ext->hascompiledvertexarrayext) {
            GLLOCKARRAYSEXT(0, cmd->numverts);
          }
#endif

          glDrawArrays(GL_TRIANGLES, 0, cmd->numverts);

#if defined(GL_EXT_compiled_vertex_array)
          if (ext->hascompiledvertexarrayext) {
            GLUNLOCKARRAYSEXT();
          }
#endif
        } else {
#endif
          // Immediate mode implementation
          int i, ind;
          glBegin(GL_TRIANGLES);
          ind = 0;
          if (cmd->pervertexcolors) {
            for (i=0; i<cmd->numverts; i++) {
              glColor3ubv(&colors[ind]);
              glNormal3bv((GLbyte *) &normals[ind]);
              glVertex3fv(&vertices[ind]);
              ind += 3;
            }
          } else {
            for (i=0; i<cmd->numverts; i++) {
              glNormal3bv((GLbyte *) &normals[ind]);
              glVertex3fv(&vertices[ind]);
              ind += 3;
            }
          }

          glEnd();
#if defined(GL_VERSION_1_1)
        }
#endif

#endif
        }
        break;

        
      case DTRISTRIP: 
        {
        // draw triangle strips
        DispCmdTriStrips *cmd = (DispCmdTriStrips *) cmdptr;
        int numstrips = cmd->numstrips;
        int strip;

        float *cnv;
        int *f;
        int *vertsperstrip;

        cmd->getpointers(cnv, f, vertsperstrip);

        // use single-sided lighting when drawing possible, for
        // peak rendering speed.
        if (!cmd->doublesided) {
          glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
        }

#if defined(GL_VERSION_1_1)
        if (!(simplegraphics || ogl_acrobat3dcapture)) {
          // If OpenGL 1.1, then use vertex arrays
          glInterleavedArrays(GL_C4F_N3F_V3F, 0, cnv);

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLLOCKARRAYSEXT(0, cmd->numverts);
          }
#endif

#if defined(GL_EXT_multi_draw_arrays)
          // Try the Sun/ARB MultiDrawElements() extensions first.
          if (ext->hasmultidrawext) {
            int **indices = new int *[cmd->numstrips];

            // build array of facet list pointers to allow the renderer to
            // send everything in a single command/DMA when possible
            int qv=0;
            for (int i=0; i<numstrips; i++) {
              indices[i] = (int *) ((char *)f + qv * sizeof(int));
              qv += vertsperstrip[i]; // incr vertex index, next strip
            }

            GLMULTIDRAWELEMENTSEXT(GL_TRIANGLE_STRIP, 
                                   (GLsizei *) vertsperstrip, 
                                   GL_UNSIGNED_INT, 
                                   (const GLvoid **) indices, 
                                   numstrips);

            delete [] indices;
          }
          else  // if not MDE, then fall back to other techniques
#endif
          // Use the regular OpenGL 1.1 vertex array APIs, loop over all strips
          {
            int qv=0;
            for (strip=0; strip < numstrips; strip++) {
              glDrawElements(GL_TRIANGLE_STRIP, vertsperstrip[strip],
                             GL_UNSIGNED_INT, (int *) ((char *) f + qv * sizeof(int)));
              qv += vertsperstrip[strip];
            }
          }

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLUNLOCKARRAYSEXT();
          }
#endif
        } else {
#endif
          // simple graphics mode, but not Acrobat3D capture mode
          if (!ogl_acrobat3dcapture) {
            // No OpenGL 1.1? ouch, then we have to do this the slow way
            int t, ind;
            int v = 0; // current vertex index, initially 0
            // loop over all of the triangle strips
            for (strip=0; strip < numstrips; strip++) {         
              glBegin(GL_TRIANGLE_STRIP);
              // render all of the triangles in this strip
              for (t = 0; t < vertsperstrip[strip]; t++) {
                ind = f[v] * 10;
                 glColor3fv(cnv + ind    );
                glNormal3fv(cnv + ind + 4);
                glVertex3fv(cnv + ind + 7);
                v++; // increment vertex index, for the next triangle
              }
              glEnd();
            }
          } else {
            // Acrobat3D capture mode works around several bugs in the
            // capture utility provided with version 7.x.  Their capture
            // feature can't catch triangle strips, so we have to render
            // each of the triangles individually.

            // render triangle strips one triangle at a time
            // triangle winding order is:
            //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
            int strip, t, v = 0;
            int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

            // loop over all of the triangle strips
            for (strip=0; strip < numstrips; strip++) {
              // loop over all triangles in this triangle strip
              glBegin(GL_TRIANGLES);

              for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
                // render one triangle, using lookup table to fix winding order
                int v0 = f[v + (stripaddr[t & 0x01][0])] * 10;
                int v1 = f[v + (stripaddr[t & 0x01][1])] * 10;
                int v2 = f[v + (stripaddr[t & 0x01][2])] * 10;

#if 1
                // Version 7.0.9 of Acrobat3D can't capture multicolor
                // triangles, so we revert to averaged-single-color
                // triangles until they fix this capture bug.
                float tmp[3], tmp2[3];
                vec_add(tmp, cnv + v0, cnv + v1); 
                vec_add(tmp2, tmp, cnv + v2); 
                vec_scale(tmp, 0.3333333f, tmp2);
                glColor3fv(tmp);
                glNormal3fv(cnv + v0 + 4);
                glVertex3fv(cnv + v0 + 7);
                glNormal3fv(cnv + v1 + 4);
                glVertex3fv(cnv + v1 + 7);
                glNormal3fv(cnv + v2 + 4);
                glVertex3fv(cnv + v2 + 7);
#else
                 glColor3fv(cnv + v0    );
                glNormal3fv(cnv + v0 + 4);
                glVertex3fv(cnv + v0 + 7);
                 glColor3fv(cnv + v1    );
                glNormal3fv(cnv + v1 + 4);
                glVertex3fv(cnv + v1 + 7);
                 glColor3fv(cnv + v2    );
                glNormal3fv(cnv + v2 + 4);
                glVertex3fv(cnv + v2 + 7);
#endif

                v++; // move on to next vertex
              }
              glEnd();
              v+=2; // last two vertices are already used by last triangle
            }
          }

#if defined(GL_VERSION_1_1)
        }
#endif

        // return to double-sided lighting mode if we switched
        if (!cmd->doublesided) {
          glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
        }
        }
        break;

      case DWIREMESH: 
        {
        // draw a wire mesh
        DispCmdWireMesh *cmd = (DispCmdWireMesh *) cmdptr;
        int ind = cmd->numlines * 2;
        float *cnv;
        int *l;
        cmd->getpointers(cnv, l);
#if defined(GL_VERSION_1_1)
        if (!simplegraphics) {
          glInterleavedArrays(GL_C4F_N3F_V3F, 0, cnv);

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLLOCKARRAYSEXT(0, cmd->numverts);
          }
#endif

          glDrawElements(GL_LINES, ind, GL_UNSIGNED_INT, l);

#if defined(GL_EXT_compiled_vertex_array) 
          if (ext->hascompiledvertexarrayext) {
            GLUNLOCKARRAYSEXT();
          }
#endif
        } else {
#endif
          int i, ind2;
          glBegin(GL_LINES);
          for (i=0; i<ind; i++) {
            ind2 = l[i] * 10;
             glColor3fv(cnv + ind2    );
            glNormal3fv(cnv + ind2 + 4);
            glVertex3fv(cnv + ind2 + 7);
          }
          glEnd();
#if defined(GL_VERSION_1_1)
        }
#endif
        }
        break;

      case DCYLINDER:
        {
        // draw a cylinder of given radius and resolution
        float *cmd = (float *)cmdptr; 
        cylinder_full((int)(cmd[7]), cmd+9, (int)(cmd[8]));
        } 
        break;

      case DCONE:
        {
        DispCmdCone *cmd = (DispCmdCone *)cmdptr;
        // draw a cone of given radius and resolution
        cylinder(cmd->pos2, cmd->pos1, cmd->res, cmd->radius, cmd->radius2);
        }
        break;

      case DTEXTSIZE:
        textsize = ((DispCmdTextSize *)cmdptr)->size;
        break;

      case DTEXTOFFSET:
        textoffset_x = ((DispCmdTextOffset *)cmdptr)->x;
        textoffset_y = ((DispCmdTextOffset *)cmdptr)->y;
        break;

      case DTEXT:
        {
        float *pos = (float *)cmdptr;        
        float thickness = pos[3];   // thickness is stored in 4th element
        char *txt = (char *)(pos+4);
        float wp[4];
        float mp[4] = { 0, 0, 0, 1};

#ifdef VMDWIREGL
        // WireGL doesn't suppor the glPushAttrib() function, so these are
        // variables used to save current OpenGL state prior to 
        // clobbering it with new state, so we can return properly.
        GLfloat   tmppointSize;
        GLfloat   tmplineWidth;
        GLboolean tmplineStipple;
        GLint     tmplineSRepeat;
        GLint     tmplineSPattern;
#endif

        mp[0] = pos[0]; mp[1] = pos[1]; mp[2] = pos[2];
        textMat.multpoint4d(mp,wp);

        glPushMatrix();
          glLoadIdentity();
          glMatrixMode(GL_PROJECTION);
          glPushMatrix();
            glLoadIdentity();
            glTranslatef((wp[0]+textoffset_x)/wp[3], 
                         (wp[1]+textoffset_y)/wp[3], 
                          wp[2]/wp[3]);

            glScalef(textsize/Aspect,textsize,textsize);

#ifdef VMDWIREGL
              glGetFloatv(GL_POINT_SIZE,          &tmppointSize   );
              glGetFloatv(GL_LINE_WIDTH,          &tmplineWidth   );
            glGetIntegerv(GL_LINE_STIPPLE_REPEAT, &tmplineSRepeat );
            glGetIntegerv(GL_LINE_STIPPLE_PATTERN,&tmplineSPattern);
            tmplineStipple = glIsEnabled(GL_LINE_STIPPLE);
#else
            glPushAttrib(GL_LINE_BIT | GL_POINT_BIT);
#endif

            // enable line antialiasing, looks much nicer, may run slower
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_LINE_SMOOTH);

// #define VMDMSAAFONTTOGGLE 1

            // MSAA lines with widths > 1.0 can look bad at low sample counts
            // so we either toggle MSAA off/on, or we have to stick to lines
            // of 1.0 pixels in width.
#if defined(VMDMSAAFONTTOGGLE)
#if defined(GL_ARB_multisample)
            // Toggle MSAA off/on on-the-fly
            if (aaEnabled) {
              glDisable(GL_MULTISAMPLE_ARB);
            }
            glLineWidth(thickness);
            glPointSize(thickness * 0.95f); // scale down point size by a hair
#endif 
#else 
            glLineWidth(thickness);
            glPointSize(thickness * 0.95f); // scale down point size by a hair
#endif

            glDisable(GL_LINE_STIPPLE);
            if (thickness > 2.0f)
              glListBase(fontNpxListBase); // font stroke vectors only
            else
              glListBase(font1pxListBase); // font stroke vectors+points

            glCallLists(strlen(txt), GL_UNSIGNED_BYTE, (GLubyte *)txt);

#if defined(VMDMSAAFONTTOGGLE)
#if defined(GL_ARB_multisample)
            // Toggle MSAA off/on on-the-fly
            if (aaEnabled) {
              glEnable(GL_MULTISAMPLE_ARB);
            }
#endif 
#endif

            // disable line antialiasing, return to normal mode 
            glDisable(GL_BLEND);
            glDisable(GL_LINE_SMOOTH);

#ifdef VMDWIREGL
            glLineWidth(tmplineWidth);
            glPointSize(tmppointSize);
            glLineStipple(tmplineSRepeat, (GLushort) tmplineSPattern);
            if (tmplineStipple == GL_TRUE)
               glEnable(GL_LINE_STIPPLE);
            else
               glDisable(GL_LINE_STIPPLE);
#else
            glPopAttrib();
#endif


          glPopMatrix();
          glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        }
        break;

      case DCOLORINDEX:
        // set the current color to the given color index ... assumes the
        // color has already been defined
        glColor3fv((GLfloat *)(colorData+3*(((DispCmdColorIndex *)cmdptr)->color)));
        break;

      case DMATERIALON:
        glEnable(GL_LIGHTING);
        ogl_lightingenabled=1;
#if defined(VMDUSEOPENGLSHADER)
        if (mainshader && ogl_useglslshader) {
          mainshader->UseShader(1); // use glsl mainshader when shading is on
        }
#endif
        break;

      case DMATERIALOFF:
        glDisable(GL_LIGHTING);
        ogl_lightingenabled=0;
#if defined(VMDUSEOPENGLSHADER)
        if (mainshader && ogl_useglslshader) {
          mainshader->UseShader(0); // use fixed-func pipeline when shading is off
        }
#endif
        break;

      case DSPHERERES:
        // set the current sphere resolution
        set_sphere_res(((DispCmdSphereRes *)cmdptr)->res);
        break;

      case DSPHERETYPE:
        // set the current sphere type
        set_sphere_mode(((DispCmdSphereType *)cmdptr)->type);
        break;

      case DLINESTYLE: 
        // set the current line style
        set_line_style(((DispCmdLineType *)cmdptr)->type);
        break;

      case DLINEWIDTH: 
        // set the current line width
        set_line_width(((DispCmdLineWidth *)cmdptr)->width);
        break;

      case DPICKPOINT:
      case DPICKPOINT_ARRAY:
      default:
        // msgErr << "OpenGLRenderer: Unknown drawing token " << tok
        //        << " encountered ... Skipping this command." << sendmsg;
        break;

    } 
  }
 
  } // XXX code to run render loop or not

  // Tail end of display list caching code
  if (ogl_cacheenabled && (!ogl_cacheskip)) { 
    if (ogl_cachecreated) {
      glEndList();              // finish off display list we're creating
    } else {
      if (ogl_cachedebug) {
        msgInfo << "Calling cached geometry: id=" << (int)ogl_cachedid << sendmsg;
      }
      glCallList(ogl_cachedid); // call the display list we previously cached
    }
  }


  glPopMatrix();
} // end loop over instance images

  glPopMatrix();
} // end loop over periodic images

  // restore transformation matrix
  glPopMatrix();
}

void OpenGLRenderer::render_done() {
  ogl_glsltoggle = 1; // force GLSL update next time through

  GLuint tag;
  // delete all unused display lists
  while ((tag = displaylistcache.deleteUnused()) != GLCACHE_FAIL) {
    glDeleteLists(tag, 1);
  }

  // delete all unused textures
  while ((tag = texturecache.deleteUnused()) != GLCACHE_FAIL) {
    glDeleteTextures(1, &tag);
  }
}

void OpenGLRenderer::require_volume_texture(unsigned long ID,
    unsigned xsize, unsigned ysize, unsigned zsize,
    unsigned char *texmap) {

  if (!ext->hastex3d) return;
  GLuint texName;
  if ((texName = texturecache.markUsed(ID)) == 0) {
    glGenTextures(1, &texName);
    texturecache.encache(ID, texName); // cache this texture ID
    glBindTexture(GL_TEXTURE_3D, texName);

    // set texture border color to black
    GLfloat texborder[4] = {0.0, 0.0, 0.0, 1.0};
    glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, texborder);

    // use the border color when wrapping at the edge
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // enable texture interpolation and filtering
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    if (build3Dmipmaps(xsize, ysize, zsize, texmap)) {
      msgErr << "OpenGLRenderer failed to download 3-D texture map!" 
             << sendmsg; 
    }
  } else { // already cached, so just enable.
    glBindTexture(GL_TEXTURE_3D, texName);
  }
}


int OpenGLRenderer::build3Dmipmaps(int width, int height, int depth, unsigned char *tx) {
#if defined(GL_VERSION_1_2)
  if (ext->hastex3d) {
    int xsize=width;
    int ysize=height;
    int zsize=depth;
    int xstep=1;
    int ystep=1;
    int zstep=1;
    int x,y,z;

    if (tx == NULL) {
      msgErr << "Skipping MIP map generation for NULL 3-D texture map" 
             << sendmsg;
      return 1;
    } 

    // build Mipmaps if the card can't handle the full resolution texture map
    if (xsize > max3DtexX || ysize > max3DtexY || zsize > max3DtexZ) { 
      unsigned char *texmap;

      while (xsize > max3DtexX) {
        xsize >>= 1;
        xstep <<= 1;
      }
  
      while (ysize > max3DtexY) {
        ysize >>= 1;
        ystep <<= 1;
      }

      while (zsize > max3DtexZ) {
        zsize >>= 1; 
        zstep <<= 1;
      }

      if (xsize == 0 || ysize == 0 || zsize == 0)
        return 1; // error, can't subsample the image down to required res

      texmap = (unsigned char *) malloc(xsize*ysize*zsize*3);
      if (texmap == NULL) {
        msgErr << "Failed to allocate MIP map for downsampled texture" 
               << sendmsg;
        return 1; // failed allocation
      }

#if 0
      // XXX draw a checkerboard texture until the MIPmap code is finished
      msgError << "3-D texture map can't fit into accelerator memory, aborted."
               << sendmsg;

      for (z=0; z<zsize; z++) {
        for (y=0; y<ysize; y++) {
          int addr = z*xsize*ysize + y*xsize;
          for (x=0; x<xsize; x++) {
            if ((x + y + z) % 2) {
              texmap[(addr + x)*3    ] = 0;
              texmap[(addr + x)*3 + 1] = 0;
              texmap[(addr + x)*3 + 2] = 0;
            } else {
              texmap[(addr + x)*3    ] = 255;
              texmap[(addr + x)*3 + 1] = 255;
              texmap[(addr + x)*3 + 2] = 255;
            }
          }
        }
      }

#else
      msgInfo << "Downsampling 3-D texture map from " 
              << width << "x" << height << "x" << depth << " to " 
              << xsize << "x" << ysize << "x" << zsize << sendmsg;
               
      for (z=0; z<zsize; z++) {
        for (y=0; y<ysize; y++) {
          int addr = z*xsize*ysize + y*xsize;
          for (x=0; x<xsize; x++) {
            int sumR=0, sumG=0, sumB=0;
            int texelcount = 0;
            int ox, oxs, oxe;
            int oy, oys, oye;
            int oz, ozs, oze;

            oxs = x * xstep;
            oys = y * ystep;
            ozs = z * zstep;

            oxe = oxs + xstep;
            oye = oys + ystep;
            oze = ozs + zstep;
            if (oxe > width) oxe=width;
            if (oye > height) oye=height;
            if (oze > depth) oze=depth;

            for (oz=ozs; oz<oze; oz++) {
              for (oy=oys; oy<oye; oy++) {
                int oaddr = oz*width*height + oy*width;
                for (ox=oxs; ox<oxe; ox++) {
                  int oadx = (oaddr + ox)*3;
                  sumR += tx[oadx    ];
                  sumG += tx[oadx + 1];
                  sumB += tx[oadx + 2];
                  texelcount++;
                }
              }
            }

            int adx = (addr + x)*3;
            texmap[adx    ] = (unsigned char) (sumR / ((float) texelcount));
            texmap[adx + 1] = (unsigned char) (sumG / ((float) texelcount));
            texmap[adx + 2] = (unsigned char) (sumB / ((float) texelcount));
          }
        }
      }
#endif

      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      GLTEXIMAGE3D(GL_TEXTURE_3D, 0, GL_RGB8, xsize, ysize, zsize,
                   0, GL_RGB, GL_UNSIGNED_BYTE, texmap);

      free(texmap); // free the generated texture map for now
    } else {
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      GLTEXIMAGE3D(GL_TEXTURE_3D, 0, GL_RGB8, width, height, depth,
                   0, GL_RGB, GL_UNSIGNED_BYTE, tx);
    }
  
    return 0;
  }
#endif

  return 1; // failed to render 3-D texture map
}

void OpenGLRenderer::update_shader_uniforms(void * voidshader, int forceupdate) {
#if defined(VMDUSEOPENGLSHADER)
  OpenGLShader *sh = (OpenGLShader *) voidshader; 
  GLint loc;

  // Update GLSL projection mode (used to control normal flipping code)
  GLint vmdprojectionmode = (ogl_glslprojectionmode == DisplayDevice::PERSPECTIVE) ? 1 : 0;
  loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdprojectionmode");
  GLUNIFORM1IARB(loc, vmdprojectionmode);

  // Update active GLSL texturing mode from cached state just in case
  GLint vmdtexturemode = ogl_glsltexturemode;
  loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdtexturemode");
  GLUNIFORM1IARB(loc, vmdtexturemode);

  // Update material parameters for OpenGL shader.
  // XXX unnecessary once references to gl_FrontMaterial.xxx work
  GLfloat matparms[4];
  matparms[0] = oglambient;
  matparms[1] = ogldiffuse;
  matparms[2] = oglspecular;
  matparms[3] = oglshininess;
  loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdmaterial");
  GLUNIFORM4FVARB(loc, 1, matparms);

  // Set vmdopacity uniform used for alpha-blended transparency in GLSL  
  GLfloat vmdopacity[1];
  vmdopacity[0] = oglopacity;
  loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdopacity");
  GLUNIFORM1FVARB(loc, 1, vmdopacity);

  // Set GLSL outline magnitude and width
  GLfloat vmdoutline[1];
  vmdoutline[0] = ogloutline;
  loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdoutline");
  GLUNIFORM1FVARB(loc, 1, vmdoutline);

  GLfloat vmdoutlinewidth[1];
  vmdoutlinewidth[0] = ogloutlinewidth;
  loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdoutlinewidth");
  GLUNIFORM1FVARB(loc, 1, vmdoutlinewidth);

  // Set GLSL transparency rendering mode for active material
  loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdtransmode");
  GLUNIFORM1IARB(loc, ogltransmode);

  // Set fog mode for shader using vmdfogmode uniform
  loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdfogmode");
  GLUNIFORM1IARB(loc, ogl_fogmode);

  // Set active texture map index
  loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdtex0");
  GLUNIFORM1IARB(loc, 0); // using texture unit 0

  // Update the main lighting state used by GLSL if it isn't the same
  // as what is currently set in the fixed-function pipeline.
  // XXX this code will not be necessary once vendors correctly implement
  //     references to gl_LightSource[n].position in GLSL shader
  if (forceupdate || (ogl_glslserial != ogl_rendstateserial)) {
    int i;

    if (!forceupdate) {
      // Once updated, no need to do it again
      ogl_glslserial = ogl_rendstateserial;
    }

    // Set light positions and pre-calculating Blinn halfway
    // vectors for use by the shaders
    for (i=0; i<DISP_LIGHTS; i++) {
      char varbuf[32];
      sprintf(varbuf, "vmdlight%d", i);
      GLint loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, varbuf);
      GLUNIFORM3FVARB(loc, 1, &ogl_lightpos[i][0]);

      // calculate Blinn's halfway vector 
      // L = direction to light
      // V = direction to camera
      // H=normalize(L + V)
      float L[3], V[3];
      GLfloat Hvec[3];
      (transMat.top()).multpoint3d(&ogl_lightpos[i][0], L);
      vec_scale(V, -1.0, eyeDir);
      vec_normalize(V);
      Hvec[0] = L[0] + V[0];
      Hvec[1] = L[1] + V[1];
      Hvec[2] = L[2] + V[2];
      vec_normalize(Hvec);
      sprintf(varbuf, "vmdlight%dH", i);
      loc = GLGETUNIFORMLOCATIONARB(mainshader->ProgramObject, varbuf);
      GLUNIFORM3FVARB(loc, 1, Hvec);
    } 

    // Set light on/off state for shader as well, using pre-known uniforms
    // XXX this code assumes a max of 4 lights, due to the use of a 
    //     vec4 for storing the values, despite DISP_LIGHTS sizing 
    //     the array of light scales.
    loc = GLGETUNIFORMLOCATIONARB(sh->ProgramObject, "vmdlightscale");
    GLfloat vmdlightscale[DISP_LIGHTS];
    for (i=0; i<DISP_LIGHTS; i++) {
      vmdlightscale[i] = (float) ogl_lightstate[i];
    }
    GLUNIFORM4FVARB(loc, 1, vmdlightscale);
  }
#endif
}


