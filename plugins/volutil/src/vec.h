/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: vec.h,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.2 $	$Date: 2009/08/19 04:11:17 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *
 ***************************************************************************/

#ifndef _VEC_H
#define _VEC_H

// Vector utilities

inline void vset(double *vout, double x, double y, double z) {
  vout[0] = x;
  vout[1] = y;
  vout[2] = z;
}

inline void vset(float *vout, float x, float y, float z) {
  vout[0] = x;
  vout[1] = y;
  vout[2] = z;
}

void vcopy(double *vout, const double *v1);
void  vcopy(float *vout, const float *v1);

double vnorm(const double *vec);
float  vnorm(const float *vec);

inline void vscale(double *vout, double ff) {
  vout[0] *= ff;
  vout[1] *= ff;
  vout[2] *= ff;
}

inline void vscale(float *vout, float ff) {
  vout[0] *= ff;
  vout[1] *= ff;
  vout[2] *= ff;
}

inline void vadd(double *vout, const double* v1, const double *v2) {
  vout[0] = v1[0] + v2[0];
  vout[1] = v1[1] + v2[1];
  vout[2] = v1[2] + v2[2];
}

inline void vadd(double *vout, const double* v1, const double *v2, const double *v3) {
  vout[0] = v1[0] + v2[0] + v3[0];
  vout[1] = v1[1] + v2[1] + v3[1];
  vout[2] = v1[2] + v2[2] + v3[2];
}


inline void vsub(double *vout, const double* v1, const double *v2) {
  vout[0] = v1[0] - v2[0];
  vout[1] = v1[1] - v2[1];
  vout[2] = v1[2] - v2[2];
}

inline void vsub(float *vout, const float* v1, const float *v2) {
  vout[0] = v1[0] - v2[0];
  vout[1] = v1[1] - v2[1];
  vout[2] = v1[2] - v2[2];
}


inline void vcross(double *vout, const double* v1, const double *v2) {
  vout[0] = v1[1]*v2[2] - v1[2]*v2[1];
  vout[1] = v1[0]*v2[2] - v1[2]*v2[0];
  vout[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


inline void vcross(float *vout, const float* v1, const float *v2) {
  vout[0] = v1[1]*v2[2] - v1[2]*v2[1];
  vout[1] = v1[0]*v2[2] - v1[2]*v2[0];
  vout[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


inline double vdot(double* v1, double *v2) {
  return (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]);
}

inline float vdot(float* v1, float *v2) {
  return (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]);
}


inline void vaddscaledto(double *vout, double ff, double *v1) {
  vout[0] += ff*v1[0];
  vout[1] += ff*v1[1];
  vout[2] += ff*v1[2];
}

inline void vaddscaledto(float *vout, float ff, float *v1) {
  vout[0] += ff*v1[0];
  vout[1] += ff*v1[1];
  vout[2] += ff*v1[2];
}

#endif
