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
 *	$RCSfile: R3dDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.35 $	$Date: 2011/02/23 05:36:54 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * The R3dDisplayDevice implements routines needed to render to a file 
 * in raster3d format
 *
 ***************************************************************************/
#ifndef R3DDISPLAYDEVICE
#define R3DDISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass exports VMD scene to a file in Raster3D format
class R3dDisplayDevice : public FileRenderer {
private:
  /// Keep track of when an object would be legal (also where
  /// a comment would be legal)
  int objLegal;

  /// Keep track of the last-accessed material definitions; this avoids
  /// the need to rewrite the same material properties over and over for
  /// each object.
  float old_mat_shininess;
  float old_mat_opacity;
  float old_mat_specular;
  int mat_on;

  void reset_vars(void); ///< reset internal state variables
  void write_materials(void);
  void close_materials(void);

protected:
  // assorted graphics functions
  void comment(const char *);
  void cylinder(float *, float *, float, int);
  void line(float *, float *);
  void point(float *);
  void sphere(float *);
  void text(float *pos, float size, float thickness, const char *str);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float *c1,    const float *c2,    const float *c3);
  
public: 
  R3dDisplayDevice(void);
  virtual ~R3dDisplayDevice(void);
  void write_header(void);
  void write_trailer(void);
}; 

#endif

