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
 *	$RCSfile: VrmlDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.28 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   FileRenderer subclass to export VMD scenes to VRML 1.0 scene format
 *
 ***************************************************************************/

#ifndef VRMLDISPLAYDEVICE_H
#define VRMLDISPLAYDEVICE_H

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to VRML 1.0 scene format
class VrmlDisplayDevice : public FileRenderer {
private:
  struct triList {
    triList *next;
    int ptz[3];
  };
  triList *tList;
  void write_cindexmaterial(int, int); // write colors, materials etc.
  void write_colormaterial(float *, int); // write colors, materials etc.

protected:
  void comment(const char *);
  void cone(float *a, float *b, float rad, int /* resolution */);
  void cylinder(float *a, float *b, float rad, int filled);
  void line(float *xyz1, float *xyz2);
  void sphere(float *xyzr);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);

  /// transformation functions
  void push(void);
  void pop(void);
  void load(const Matrix4& mat);
  void multmatrix(const Matrix4& mat);
  void set_color(int color_index); ///< set the colorID

public:
  VrmlDisplayDevice(void);
  void write_header(void);
  void write_trailer(void);
}; 

#endif



