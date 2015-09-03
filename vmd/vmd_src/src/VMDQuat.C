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
 *      $RCSfile: VMDQuat.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.11 $      $Date: 2010/12/16 04:08:46 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Quaternion class used by tracker/tool code 
 ***************************************************************************/

#include <math.h>
#include "VMDQuat.h"
#include "utilities.h"

void Quat::identity() {
  qx = qy = qz = 0;
  qw = 1;
}

Quat::Quat(double x, double y, double z, double w) {
  qx = x;
  qy = y;
  qz = z;
  qw = w;
}

void Quat::rotate(const float *u, float angle) {
  Quat q;
  double theta = DEGTORAD(angle);
  q.qw = cos(0.5*theta);
  double sintheta = sin(0.5*theta);
  q.qx = u[0]*sintheta;
  q.qy = u[1]*sintheta;
  q.qz = u[2]*sintheta;
  mult(q);
}

void Quat::rotate(char axis, float angle) {
  Quat q;
  double theta = DEGTORAD(angle);
  q.qw = cos(0.5*theta);
  double sintheta = sin(0.5*theta);
  switch(axis) {
    case 'x': q.qx = sintheta; q.qy = q.qz = 0; break;
    case 'y': q.qy = sintheta; q.qz = q.qx = 0; break;
    case 'z': q.qz = sintheta; q.qx = q.qy = 0; break;
  }
  mult(q); 
}

void Quat::invert() {
  qx = -qx;
  qy = -qy;
  qz = -qz;
}

void Quat::mult(const Quat &q) {
  double x,y,z,w;
  w = q.qw*qw - q.qx*qx - q.qy*qy - q.qz*qz;
  x = q.qw*qx + q.qx*qw + q.qy*qz - q.qz*qy;
  y = q.qw*qy - q.qx*qz + q.qy*qw + q.qz*qx;
  z = q.qw*qz + q.qx*qy - q.qy*qx + q.qz*qw;
  qw = w;
  qx = x;
  qy = y;
  qz = z;
}

void Quat::multpoint3(const float *p, float *out) const {
  Quat pquat(p[0], p[1], p[2], 0);
  Quat inv(-qx, -qy, -qz, qw);
  inv.mult(pquat);
  inv.mult(*this);
  out[0] = (float) inv.qx;
  out[1] = (float) inv.qy;
  out[2] = (float) inv.qz;
}

void Quat::printQuat(float *q) {
  q[0] = (float) qx;
  q[1] = (float) qy;
  q[2] = (float) qz;
  q[3] = (float) qw;
}

void Quat::printMatrix(float *m) {
  m[0] = (float) (qw*qw + qx*qx - qy*qy -qz*qz);
  m[1] = (float) (2*(qx*qy + qw*qz));
  m[2] = (float) (2*(qx*qz - qw*qy));

  m[4] = (float) (2*(qx*qy - qw*qz));
  m[5] = (float) (qw*qw - qx*qx + qy*qy - qz*qz);
  m[6] = (float) (2*(qy*qz + qw*qx));
  
  m[8] = (float) (2*(qx*qz + qw*qy));
  m[9] = (float) (2*(qy*qz - qw*qx));
  m[10]= (float) (qw*qw - qx*qx - qy*qy + qz*qz);
  
  m[15]= (float) (qw*qw + qx*qx + qy*qy + qz*qz);
  
  m[3] = m[7] = m[11] = m[12] = m[13] = m[14] = 0;
}

static int perm[3] = {1,2,0};
#define mat(a,b) (m[4*a+b])

void Quat::fromMatrix(const float *m) {
  float T = mat(0,0) + mat(1,1) + mat(2,2);
  if (T > 0) { // w is the largest element in the quat
    double iw = sqrt(T+1.0);
    qw = iw * 0.5; 
    iw = 0.5/iw;
    qx = (mat(1,2) - mat(2,1)) * iw;
    qy = (mat(2,0) - mat(0,2)) * iw;
    qz = (mat(0,1) - mat(1,0)) * iw;
  } else {     // Find the largest diagonal element
    int i,j,k;
    double &qi = qx, &qj = qy, &qk = qz;
    i=0;
    if (mat(1,1) > mat(0,0)) {i=1; qi = qy; qj = qz; qk = qx; }
    if (mat(2,2) > mat(i,i)) {i=2; qi = qz; qj = qx; qk = qy; }
    j=perm[i];
    k=perm[j];
 
    double iqi = sqrt( (mat(i,i) - (mat(j,j) + mat(k,k))) + 1.0);
    qi = iqi * 0.5;
    iqi = 0.5/iqi;
   
    qw = (mat(j,k) - mat(k,j))*iqi;
    qj = (mat(i,j) + mat(j,i))*iqi;
    qk = (mat(i,k) + mat(k,i))*iqi;
  }
}
 
