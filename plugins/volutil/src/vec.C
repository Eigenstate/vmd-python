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
 *	$RCSfile: vec.C,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.1 $	$Date: 2009/08/06 20:58:47 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *
 ***************************************************************************/

#include <math.h>
#include <string.h>

#include "vec.h"

void vcopy(double *vout, const double *v1) {
  memcpy(vout, v1, 3*sizeof(double));
}

void  vcopy(float *vout, const float *v1) {
  memcpy(vout, v1, 3*sizeof(float));
}


double vnorm(const double *vec) {
  return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

float vnorm(const float *vec) {
  return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}


