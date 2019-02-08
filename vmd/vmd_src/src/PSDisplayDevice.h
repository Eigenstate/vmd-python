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
 *      $RCSfile: PSDisplayDevice.h,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.42 $       $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *   Save the image to a Postscript file.
 *
 ***************************************************************************/

#ifndef PSDISPLAYDEVICE_H
#define PSDISPLAYDEVICE_H

#include <stdio.h>
#include "DepthSortObj.h"
#include "DispCmds.h"
#include "FileRenderer.h"
#include "SortableArray.h"

/// FileRenderer subclass exports VMD scenes to PostScript files
class PSDisplayDevice : public FileRenderer {
private:
   SortableArray <DepthSortObject> depth_list;
   float x_scale, y_scale;
   float x_offset, y_offset;

   /// Flag for malloc errors
   int memerror;

   /// Draws the depth-sorted objects.
   void process_depth_list(void);

   /// Change the number of polygons used to represent a sphere
   void set_sphere_res(int res);

   //@{
   /// Push the polygonal representation of an object onto the depth-sorted
   /// list. 
   void sphere_approx(float *c, float r);
   void cylinder_approx(float *a, float *b, float r, int res, int filled);
   void cone_approx(float *a, float *b, float r);
   void decompose_mesh(DispCmdTriMesh *mesh);
   void decompose_tristrip(DispCmdTriStrips *strip);
   //@}

   inline float compute_dist(float *c);
   float compute_light(float *a, float *b, float *c);
   float norm_light[3];

   //@{
   /// Variables used to cache the triangle mesh approximation of a unit
   /// sphere.
   int sph_iter;
   int sph_desired_iter;
   int sph_nverts;
   float *sph_verts;
   //@}

   //@{
   ///Rendering statistics.
   long memusage;
   long points;
   long objects;
   //@}

protected:
   virtual void comment(const char *s);

public:
   PSDisplayDevice(void);
   ~PSDisplayDevice(void);
   virtual void write_header(void);
   virtual void write_trailer(void);

   /// Process the display list, adding primitives to the depth-sorted
   /// list for final output. Higher-level geometry (e.g., spheres and 
   /// cylinders) are decomposed into simpler primitives (e.g., triangles 
   /// and squares).
   virtual void render(const VMDDisplayList *display_list);
   virtual void render_done(void);
};

#endif

