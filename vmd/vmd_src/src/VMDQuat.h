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
 *      $RCSfile: VMDQuat.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.12 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Quaternion class 
 ***************************************************************************/

#ifndef QUAT_H__
#define QUAT_H__

/// Quaternion class
class Quat {
public:
  Quat()  {identity(); }               ///< initialize to identity
  Quat(double x, double y, double z, double w);

  // for the next two, angle is measured in degrees  
  void rotate(const float *u, float angle);  // assume u normalized!
  void rotate(char axis, float angle); ///< axis = 'x', 'y', or 'z'
  void invert();
  void identity();
 
  void mult(const Quat &);
  void multpoint3(const float *, float *) const;

  void fromMatrix(const float *);      ///< convert from a row-major matrix
  void printQuat(float *);             ///< print it
  void printMatrix(float *);           ///< print it as a matrix

private:
  double qx, qy, qz, qw;
};

#endif	
