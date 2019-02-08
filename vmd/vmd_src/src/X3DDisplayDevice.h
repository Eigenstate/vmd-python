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
 *	$RCSfile: X3DDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.14 $	$Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   X3D scene export code
 ***************************************************************************/

#ifndef X3DDISPLAYDEVICE_H
#define X3DDISPLAYDEVICE_H

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to X3D scene format
class X3DDisplayDevice : public FileRenderer {
private:
  virtual void cylinder_noxfrm(float *a, float *b, float rad, int filled);

protected:
  // XXX would-be private routines also needed by the X3DOM subclass
  virtual void write_cindexmaterial(int, int); // write colors, materials etc.
  virtual void write_colormaterial(float *, int); // write colors, materials etc.

  // assorted graphics functions
  virtual void comment(const char *);
  virtual void cone(float *a, float *b, float rad, int /* resolution */);
  virtual void cylinder(float *a, float *b, float rad, int filled);
  virtual void line(float *xyz1, float *xyz2);
  virtual void line_array(int num, float thickness, float *points);
  virtual void point(float *xyz);
  virtual void point_array(int num, float size, float *xyz, float *colors);
  virtual void polyline_array(int num, float thickness, float *points);
  virtual void sphere(float *xyzr);
  virtual void text(float *pos, float size, float thickness, const char *str);
  virtual void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  virtual void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float *c1,    const float *c2,    const float *c3);
  virtual void trimesh_c4n3v3(int numverts, float * cnv, 
                              int numfacets, int * facets);
  virtual void trimesh_c4u_n3b_v3f(unsigned char *c, char *n,
                                   float *v, int numfacets);
  virtual void tristrip(int numverts, const float * cnv,
                        int numstrips, const int *vertsperstrip,
                        const int *facets);

  virtual void load(const Matrix4& mat);       ///< load transofrmation matrix
  virtual void multmatrix(const Matrix4& mat); ///< concatenate transformation
  virtual void set_color(int color_index);     ///< set the colorID

public:
  /// construct the renderer; set the 'visible' name for the renderer list
  X3DDisplayDevice(const char *public_name,
                   const char *public_pretty_name,
                   const char *default_file_name,
                   const char *default_command_line);

  X3DDisplayDevice(void);
  virtual void write_header(void);
  virtual void write_trailer(void);
}; 


class X3DOMDisplayDevice : public X3DDisplayDevice {
protected:
  // assorted graphics functions
  virtual void line_array(int num, float thickness, float *points);
  virtual void polyline_array(int num, float thickness, float *points);
  virtual void text(float *pos, float size, float thickness, const char *str);
  virtual void tristrip(int numverts, const float * cnv,
                        int numstrips, const int *vertsperstrip,
                        const int *facets);

public:
  X3DOMDisplayDevice(void);               // constructor
};



#endif



