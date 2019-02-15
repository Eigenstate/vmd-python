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
 *	$RCSfile: POV3DisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.48 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Version 3 of the Persistence Of Vision raytracer
 *
 ***************************************************************************/

#ifndef POV3DISPLAYDEVICE
#define POV3DISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"

// #define POV_FILTERED_TRANSPARENCY
//
// To use POV-Ray's filtered transparency (where rays passing through
// transparent objects are 'tinted' the color of the transparent
// object) uncomment the above line.
//
// Otherwise, unfiltered transparency is used (no tinting). NOTE
// THAT UNFILTERED TRANSPARENCY IS ONLY AVAILABLE IN POV-RAY v. 3
// AND ABOVE

/// FileRenderer subclass exports scenes to POV-Ray 3.x ray tracer scene format
class POV3DisplayDevice : public FileRenderer {
private:
  int old_materialIndex;    ///< previous active material index
  int clip_on[3];           ///< clipping plane states
  int degenerate_cones;     ///< count of dropped primitives
  int degenerate_cylinders; ///< count of dropped primitives
  int degenerate_triangles; ///< count of dropped primitives

  void reset_vars(void);    ///< reset internal state variables
  void write_materials(void);

protected:
  // assorted graphics functions
  void comment(const char *);
  void cone(float * a, float * b, float rad, int /* resolution */); 
  void cylinder(float *, float *, float rad, int filled);
  void line(float *xyz1, float *xyz2);
  void point(float *xyz);
  void sphere(float *xyzr);
  void text(float *pos, float size, float thickness, const char *str);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float *c1,    const float *c2,    const float *c3);
  virtual void trimesh_c4n3v3(int numverts, float *cnv, 
                              int numfacets, int *facets);
  virtual void trimesh_c4u_n3b_v3f(unsigned char *c, char *n,
                                   float *v, int numfacets);
  void tristrip(int numverts, const float *cnv, int numstrips, 
                const int *vertsperstrip, const int *facets);

  void set_line_width(int new_width);
  void start_clipgroup(void);
  void end_clipgroup(void);

public: 
  POV3DisplayDevice(void);
  virtual ~POV3DisplayDevice(void);
  void write_header(void); 
  void write_trailer(void);
}; 

#endif

