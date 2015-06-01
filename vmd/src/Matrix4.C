/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: Matrix4.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.54 $	$Date: 2012/10/18 15:58:29 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * 4 x 4 Matrix, used for a transformation matrix.
 *
 ***************************************************************************/

#include <math.h>
#include <string.h>
#include <stdio.h>
#include "Matrix4.h"
#include "utilities.h"

// constructor, for the case when an array of floating-point numbers is given
Matrix4::Matrix4(const float *m) {
  memcpy((void *)mat, (const void *)m, 16*sizeof(float));
}

// multiplies a 3D point (first arg) by the Matrix, returns in second arg
void Matrix4::multpoint3d(const float opoint[3], float npoint[3]) const {
#if 0
    // should try re-testing this formulation to see if it outperforms
    // the old one, without introducing floating point imprecision
    float tmp[3];
    float itmp3 = 1.0f / (opoint[0]*mat[3] + opoint[1]*mat[7] + 
                          opoint[2]*mat[11] + mat[15]);
    npoint[0]=itmp3 * (opoint[0]*mat[0] + opoint[1]*mat[4] + opoint[2]*mat[ 8] + mat[12]);
    npoint[1]=itmp3 * (opoint[0]*mat[1] + opoint[1]*mat[5] + opoint[2]*mat[ 9] + mat[13]);
    npoint[2]=itmp3 * (opoint[0]*mat[2] + opoint[1]*mat[6] + opoint[2]*mat[10] + mat[14]);
#else
    float tmp[3];
    float itmp3 = 1.0f / (opoint[0]*mat[3] + opoint[1]*mat[7] +
                          opoint[2]*mat[11] + mat[15]);
    tmp[0] = itmp3*opoint[0];
    tmp[1] = itmp3*opoint[1];
    tmp[2] = itmp3*opoint[2];
    npoint[0]=tmp[0]*mat[0] + tmp[1]*mat[4] + tmp[2]*mat[ 8] + itmp3*mat[12];
    npoint[1]=tmp[0]*mat[1] + tmp[1]*mat[5] + tmp[2]*mat[ 9] + itmp3*mat[13];
    npoint[2]=tmp[0]*mat[2] + tmp[1]*mat[6] + tmp[2]*mat[10] + itmp3*mat[14];
#endif
}


// multiplies a 3D point array (2nd arg) by the Matrix, returns in 3rd arg
void Matrix4::multpointarray_3d(int numpts, const float *opoints, float *npoints) const {
  int i, numpts3;
  numpts3=numpts*3;
  for (i=0; i<numpts3; i+=3) {
    multpoint3d(&opoints[i], &npoints[i]);
  }
}


// multiplies a 3D norm (first arg) by the Matrix, returns in second arg
// This differs from point multiplication in that the translatation operations
// are ignored.
void Matrix4::multnorm3d(const float onorm[3], float nnorm[3]) const {
  float tmp[4];

  tmp[0]=onorm[0]*mat[0] + onorm[1]*mat[4] + onorm[2]*mat[8];
  tmp[1]=onorm[0]*mat[1] + onorm[1]*mat[5] + onorm[2]*mat[9];
  tmp[2]=onorm[0]*mat[2] + onorm[1]*mat[6] + onorm[2]*mat[10];
  tmp[3]=onorm[0]*mat[3] + onorm[1]*mat[7] + onorm[2]*mat[11];
  float itmp = 1.0f / sqrtf(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]);
  nnorm[0]=tmp[0]*itmp;
  nnorm[1]=tmp[1]*itmp;
  nnorm[2]=tmp[2]*itmp;
}


// multiplies a 3D texture plane equation by the Matrix
// This differs from point multiplication in that the translatation operations
// are ignored.
void Matrix4::multplaneeq3d(const float onorm[3], float nnorm[3]) const {
  float xvec[3] = {1.0f, 0.0f, 0.0f};
  float yvec[3] = {0.0f, 1.0f, 0.0f};
  float zvec[3] = {0.0f, 0.0f, 1.0f};
  float xvnew[3];
  float yvnew[3];
  float zvnew[3];

  multnorm3d(xvec, xvnew);
  multnorm3d(yvec, yvnew);
  multnorm3d(zvec, zvnew);

  int i;
  for (i=0; i<3; i++) {
    nnorm[i] = onorm[0] * xvnew[i] + onorm[1] * yvnew[i] + onorm[2] * zvnew[i];
  }
}


// multiplies a 4D point (first arg) by the Matrix, returns in second arg
void Matrix4::multpoint4d(const float opoint[4], float npoint[4]) const {
  npoint[0]=opoint[0]*mat[0]+opoint[1]*mat[4]+opoint[2]*mat[8]+opoint[3]*mat[12];
  npoint[1]=opoint[0]*mat[1]+opoint[1]*mat[5]+opoint[2]*mat[9]+opoint[3]*mat[13];
  npoint[2]=opoint[0]*mat[2]+opoint[1]*mat[6]+opoint[2]*mat[10]+opoint[3]*mat[14];
  npoint[3]=opoint[0]*mat[3]+opoint[1]*mat[7]+opoint[2]*mat[11]+opoint[3]*mat[15];
}


// clears the matrix (resets it to identity)
void Matrix4::identity(void) {
  memset((void *)mat, 0, 16*sizeof(float));
  mat[0]=1.0f;
  mat[5]=1.0f;
  mat[10]=1.0f;
  mat[15]=1.0f;
}


// sets the matrix so all items are the given constant value
void Matrix4::constant(float f) {
  for (int i=0; i<16; mat[i++] = f); 
}

// return the inverse of this matrix, that is, 
// the inverse of the rotation, the inverse of the scaling, and 
// the opposite of the translation vector.
#define MATSWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}
int Matrix4::inverse(void) {

  float matr[4][4], ident[4][4];
  int i, j, k, l, ll;
  int icol=0, irow=0;
  int indxc[4], indxr[4], ipiv[4];
  float big, dum, pivinv, temp;
 
  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      matr[i][j] = mat[4*i+j];
      ident[i][j] = 0.0f;
    }
    ident[i][i]=1.0f;
  } 
  // Gauss-jordan elimination with full pivoting.  Yes, folks, a 
  // GL Matrix4 is inverted like any other, since the identity is 
  // still the identity.
  
  // from numerical recipies in C second edition, pg 39

  for(j=0;j<=3;j++) ipiv[j] = 0;
  for(i=0;i<=3;i++) {
    big=0.0;
    for (j=0;j<=3;j++) {
      if(ipiv[j] != 1) {
        for (k=0;k<=3;k++) {
          if(ipiv[k] == 0) {
            if(fabs(matr[j][k]) >= big) {
              big = (float) fabs(matr[j][k]);
              irow=j;
              icol=k;
            }
          } else if (ipiv[k] > 1) return 1;
        } 
      }
    }
    ++(ipiv[icol]);
    if (irow != icol) {
      for (l=0;l<=3;l++) MATSWAP(matr[irow][l],matr[icol][l]);
      for (l=0;l<=3;l++) MATSWAP(ident[irow][l],ident[icol][l]);
    }
    indxr[i]=irow;
    indxc[i]=icol;
    if(matr[icol][icol] == 0.0f) return 1; 
    pivinv = 1.0f / matr[icol][icol];
    matr[icol][icol]=1.0f;
    for (l=0;l<=3;l++) matr[icol][l] *= pivinv;
    for (l=0;l<=3;l++) ident[icol][l] *= pivinv;
    for (ll=0;ll<=3;ll++) {
      if (ll != icol) {
        dum=matr[ll][icol];
        matr[ll][icol]=0.0f;
        for (l=0;l<=3;l++) matr[ll][l] -= matr[icol][l]*dum;
        for (l=0;l<=3;l++) ident[ll][l] -= ident[icol][l]*dum;
      }
    }
  }
  for (l=3;l>=0;l--) {
    if (indxr[l] != indxc[l]) {
      for (k=0;k<=3;k++) {
        MATSWAP(matr[k][indxr[l]],matr[k][indxc[l]]);
      }
    }
  }
  for (i=0; i<4; i++) 
    for (j=0; j<4; j++)
      mat[4*i+j] = matr[i][j];
  return 0;
}
      
void Matrix4::transpose() {
  float tmp[16];
  int i,j;
  for(i=0;i<4;i++) {
    for(j=0;j<4;j++) {
      tmp[4*i+j] = mat[i+4*j];
    }
  }
  for(i=0;i<16;i++)
    mat[i] = tmp[i];
}

// replaces this matrix with the given one
void Matrix4::loadmatrix(const Matrix4& m) {
  memcpy((void *)mat, (const void *)m.mat, 16*sizeof(float));
}

// premultiply the matrix by the given matrix
void Matrix4::multmatrix(const Matrix4& m) {
  float tmp[4];
  for (int j=0; j<4; j++) {
    tmp[0] = mat[j];
    tmp[1] = mat[4+j];
    tmp[2] = mat[8+j]; 
    tmp[3] = mat[12+j];
    for (int i=0; i<4; i++) {
      mat[4*i+j] = m.mat[4*i  ]*tmp[0] + m.mat[4*i+1]*tmp[1] +
                   m.mat[4*i+2]*tmp[2] + m.mat[4*i+3]*tmp[3]; 
    }
  } 
}

// performs a rotation around an axis (char == 'x', 'y', or 'z')
// angle is in degrees
void Matrix4::rot(float a, char axis) {
  Matrix4 m;			// create identity matrix
  double angle;

  angle = (double)DEGTORAD(a);

  if (axis == 'x') {
    m.mat[ 0] = 1.0;
    m.mat[ 5] = cosf(angle);
    m.mat[10] = m.mat[5];
    m.mat[ 6] = sinf(angle);
    m.mat[ 9] = -m.mat[6];
  } else if (axis == 'y') {
    m.mat[ 0] = cosf(angle);
    m.mat[ 5] = 1.0;
    m.mat[10] = m.mat[0];
    m.mat[ 2] = -sinf(angle);
    m.mat[ 8] = -m.mat[2];
  } else if (axis == 'z') {
    m.mat[ 0] = cosf(angle);
    m.mat[ 5] = m.mat[0];
    m.mat[10] =  1.0;
    m.mat[ 1] = sinf(angle);
    m.mat[ 4] = -m.mat[1];
  }

  // If there was an error, m is identity so we can multiply anyway.
  multmatrix(m);
}

// performs rotation around a given vector
void Matrix4::rotate_axis(const float axis[3], float angle) {
  transvec(axis[0], axis[1], axis[2]);
  rot((float) (RADTODEG(angle)), 'x');
  transvecinv(axis[0], axis[1], axis[2]);
}

// applies the transformation needed to bring the x axis along the given vector. 
void Matrix4::transvec(float x, float y, float z) {
  double theta = atan2(y,x);
  double length = sqrt(y*y + x*x);
  double phi = atan2((double) z, length);
  rot((float) RADTODEG(theta), 'z');
  rot((float) RADTODEG(-phi), 'y');
}

// applies the transformation needed to bring the given vector to the x axis.
void Matrix4::transvecinv(float x, float y, float z) {
  double theta = atan2(y,x);
  double length = sqrt(y*y + x*x);
  double phi = atan2((double) z, length);
  rot((float) RADTODEG(phi), 'y');
  rot((float) RADTODEG(-theta), 'z');
}

// performs a translation
void Matrix4::translate(float x, float y, float z) {
  Matrix4 m;		// create identity matrix
  m.mat[12] = x;
  m.mat[13] = y;
  m.mat[14] = z;
  multmatrix(m);
}

// performs scaling
void Matrix4::scale(float x, float y, float z) {
  Matrix4 m;		// create identity matrix
  m.mat[0] = x;
  m.mat[5] = y;
  m.mat[10] = z;
  multmatrix(m);
}

// sets this matrix to represent a window perspective
void Matrix4::window(float left, float right, float bottom, 
			 float top, float nearval, float farval) {

  constant(0.0);		// initialize this matrix to 0
  mat[0] = (2.0f*nearval) / (right-left);
  mat[5] = (2.0f*nearval) / (top-bottom);
  mat[8] = (right+left) / (right-left);
  mat[9] = (top+bottom) / (top-bottom);
  mat[10] = -(farval+nearval) / (farval-nearval);
  mat[11] = -1.0f;
  mat[14] = -(2.0f*farval*nearval) / (farval-nearval);
}


// sets this matrix to a 3D orthographic matrix
void Matrix4::ortho(float left, float right, float bottom,
			float top, float nearval, float farval) {

  constant(0.0);		// initialize this matrix to 0
  mat[0] =  2.0f / (right-left);
  mat[5] =  2.0f / (top-bottom);
  mat[10] = -2.0f / (farval-nearval);
  mat[12] = -(right+left) / (right-left);
  mat[13] = -(top+bottom) / (top-bottom);
  mat[14] = -(farval+nearval) / (farval-nearval);
  mat[15] = 1.0;
}


// sets this matrix to a 2D orthographic matrix
void Matrix4::ortho2(float left, float right, float bottom, float top) {

  constant(0.0);		// initialize this matrix to 0
  mat[0] =  2.0f / (right-left);
  mat[5] =  2.0f / (top-bottom);
  mat[10] = -1.0f;
  mat[12] = -(right+left) / (right-left);
  mat[13] = -(top+bottom) / (top-bottom);
  mat[15] =  1.0f;
}

/* This subroutine defines a viewing transformation with the eye at the point
 * (vx,vy,vz) looking at the point (px,py,pz).  Twist is the right-hand
 * rotation about this line.  The resultant matrix is multiplied with
 * the top of the transformation stack and then replaces it.  Precisely,
 * lookat does:
 * lookat = trans(-vx,-vy,-vz)*rotate(theta,y)*rotate(phi,x)*rotate(-twist,z)
 */
 void Matrix4::lookat(float vx, float vy, float vz, float px, float py,
			 float pz, short twist) {
  Matrix4 m(0.0);
  float tmp;

  /* pre multiply stack by rotate(-twist,z) */
  rot(-twist / 10.0f,'z');

  tmp = sqrtf((px-vx)*(px-vx) + (py-vy)*(py-vy) + (pz-vz)*(pz-vz));
  m.mat[0] = 1.0;
  m.mat[5] = sqrtf((px-vx)*(px-vx) + (pz-vz)*(pz-vz)) / tmp;
  m.mat[6] = (vy-py) / tmp;
  m.mat[9] = -m.mat[6];
  m.mat[10] = m.mat[5];
  m.mat[15] = 1.0;
  multmatrix(m);

  /* premultiply by rotate(theta,y) */
  m.constant(0.0);
  tmp = sqrtf((px-vx)*(px-vx) + (pz-vz)*(pz-vz));
  m.mat[0] = (vz-pz) / tmp;
  m.mat[5] = 1.0;
  m.mat[10] = m.mat[0];
  m.mat[15] = 1.0;
  m.mat[2] = -(px-vx) / tmp;
  m.mat[8] = -m.mat[2];
  multmatrix(m);

  /* premultiply by trans(-vx,-vy,-vz) */
  translate(-vx,-vy,-vz);
}

 // Transform 3x3 into 4x4 matrix:
 void trans_from_rotate(const float mat3[9], Matrix4 *mat4) {
  int i;
  for (i=0; i<3; i++) {
    mat4->mat[0+i] = mat3[3*i];
    mat4->mat[4+i] = mat3[3*i+1];
    mat4->mat[8+i] = mat3[3*i+2];
  }
}

// Print a matrix for debugging purpose
void print_Matrix4(const Matrix4 *mat4) {
  int i, j;
  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      printf("%f ", mat4->mat[4*j+i]);
    }
    printf("\n");
  }
  printf("\n");
}

