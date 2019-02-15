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
 *	$RCSfile: Matrix4.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.42 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * 4 x 4 Matrix, used for a transformation matrix.
 *
 ***************************************************************************/
#ifndef MATRIX_FOUR_H
#define MATRIX_FOUR_H

/// 4x4 matrix class with numerous operators, conversions, etc.
class Matrix4 {
public:
  Matrix4(void) { identity(); }                ///< identity constructor
  Matrix4(float f) { constant(f); }            ///< const elements constructor
  Matrix4(const float *m);                     ///< construct from float array
  Matrix4(const Matrix4& m) { loadmatrix(m); } ///< copy constructor 
  ~Matrix4(void) {}                            ///< destructor
  float mat[16];                               ///< the matrix itself

  /// multiplies a 3D point (first arg) by the Matrix, returns in second arg
  void multpoint3d(const float[3], float[3]) const;

  /// multiplies a 3D point array (2nd arg) by the Matrix, returns in 3rd arg
  void multpointarray_3d(int numpts, const float *, float *) const;

  /// multiplies a 3D norm (first arg) by the Matrix, returns in second arg
  void multnorm3d(const float[3], float[3]) const;

  /// multiplies a 3D texture plane equation by the Matrix
  void multplaneeq3d(const float[3], float[3]) const;

  /// multiplies a 4D point (first arg) by the Matrix, returns in second arg
  void multpoint4d(const float[4], float[4]) const;

  /// clears the matrix (resets it to identity)
  void identity(void);
  
  /// sets the matrix so all items are the given constant value
  void constant(float);
  
  /// inverts the matrix, that is, 
  /// the inverse of the rotation, the inverse of the scaling, and 
  /// the opposite of the translation vector.
  /// returns 0 if there were no problems, -1 if the matrix is singular
  int inverse(void);
  
  /// transposes the matrix
  void transpose(void);
  
  /// replaces this matrix with the given one
  void loadmatrix(const Matrix4 &m);
  Matrix4& operator=(const Matrix4& m) {loadmatrix(m); return *this;}

  /// premultiply the matrix by the given matrix, this->other * this
  void multmatrix(const Matrix4 &);

  /// performs a left-handed rotation around an axis (char == 'x', 'y', or 'z')
  void rot(float, char); // angle in degrees

  /// apply a rotation around the given vector; angle in radians.
  void rotate_axis(const float axis[3], float angle);
  
  /// apply a rotation such that 'x' is brought along the given vector.
  void transvec(float x, float y, float z);
 
  /// apply a rotation such that the given vector is brought along 'x'.
  void transvecinv(float x, float y, float z);

  /// performs a translation
  void translate(float, float, float);
  void translate(float d[3]) { translate(d[0], d[1], d[2]); }

  /// performs scaling
  void scale(float, float, float);
  void scale(float f) { scale(f, f, f); }

  /// sets this matrix to represent a window perspective
  void window(float, float, float, float, float, float);

  /// sets this matrix to a 3D orthographic matrix
  void ortho(float, float, float, float, float, float);

  /// sets this matrix to a 2D orthographic matrix
  void ortho2(float, float, float, float);

  /// This subroutine defines a viewing transformation with the eye at point
  /// (vx,vy,vz) looking at the point (px,py,pz).  Twist is the right-hand
  /// rotation about this line.  The resultant matrix is multiplied with
  /// the top of the transformation stack and then replaces it.  Precisely,
  /// lookat does:
  /// lookat=trans(-vx,-vy,-vz)*rotate(theta,y)*rotate(phi,x)*rotate(-twist,z)
  void lookat(float, float, float, float, float, float, short);
};

/// Transform 3x3 into 4x4 matrix:
void trans_from_rotate(const float mat3[9], Matrix4 *mat4);

/// Print formatted matrix
void print_Matrix4(const Matrix4 *mat4);

#endif

