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
 *	$RCSfile: TachyonDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.47 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Tachyon Parallel / Multiprocessor Ray Tracer
 *
 ***************************************************************************/

#ifndef TACHYONDISPLAYDEVICE
#define TACHYONDISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to the Tachyon scene format
class TachyonDisplayDevice : public FileRenderer {
private:
  int inclipgroup;            ///< whether a clipping group is currently active
  int involtex;               ///< volume texturing is enabled
  int voltexID;               ///< current volume texturing ID
  float xplaneeq[4];          ///< volumetric texture plane equations
  float yplaneeq[4];
  float zplaneeq[4];

  void reset_vars(void);      ///< reset internal state between renders
  void write_camera(void);    ///< write out camera specification
  void write_lights(void);    ///< write out the active lights
  void write_materials(void); ///< write out colors, textures, materials etc.
  void write_cindexmaterial(int, int);    ///< write colors, materials etc.
  void write_colormaterial(float *, int); ///< write colors, materials etc.

protected:
  // assorted graphics functions
  void comment(const char *);
  void cylinder(float *, float *, float rad, int filled);
  void line(float *xyz1, float *xyz2);
  void point(float *xyz);
  void sphere(float *xyzr);
  virtual void sphere_array(int num, int res, float *centers, float *radii, float *colors);
  void text(float *pos, float size, float thickness, const char *str);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float * c1,   const float * c2,   const float * c3);
  virtual void trimesh_c4n3v3(int numverts, float * cnv, 
                              int numfacets, int * facets);
  virtual void trimesh_c4u_n3b_v3f(unsigned char *c, char *n, 
                                   float *v, int numfacets);
  virtual void tristrip(int numverts, const float * cnv,
                        int numstrips, const int *vertsperstrip,
                        const int *facets);

  // define a volumetric texture map
  virtual void define_volume_texture(int ID, int xs, int ys, int zs,
                                     const float *xplaneeq,
                                     const float *yplaneeq,
                                     const float *zplaneeq,
                                     unsigned char *texmap);

  // enable volumetric texturing, either in "replace" or "modulate" mode
  virtual void volume_texture_on(int texmode);

  // disable volumetric texturing
  virtual void volume_texture_off(void);

  // begin a group of objects to be clipped by the same set of
  // clipping planes
  void start_clipgroup(void);
  void end_clipgroup(void);

  void update_exec_cmd();

public: 
  TachyonDisplayDevice(void);           // constructor
  virtual ~TachyonDisplayDevice(void);  // destructor
  void write_header(void); 
  void write_trailer(void);
}; 

#endif

