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
 *      $RCSfile: DepthSortObj.h,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************/
#ifndef DEPTHSORTOBJECT_H
#define DEPTHSORTOBJECT_H

#include "Matrix4.h"

// We only need to have support for the following objects, which are the
// PostScript-level objects that get written:
//   triangles
//   quadrilaterals
//   points
//   lines
//   text
// We can do this by just having the number of coordinates (this
// uniquely defines the object), and the color/material information.

/// Class for storage and re-ordering of depth-sorted geometry 
/// as needed for PSDisplayDevice
class DepthSortObject {
public:
   float *points;     ///< vertices used to draw points, lines, quads, etc
   float light_scale; ///< shading factor, also used to store text scale factor
   int npoints;       ///< number of points (1, 2, 3, or 4 for quads)
   int color;         ///< color index
   float dist;        ///< distance from eye
   char* text;        ///< text to print, if a text object.

   DepthSortObject() : light_scale(0.0f), npoints(0), color(0), text(NULL) {}

   int operator>(const DepthSortObject& cmp) {
      return dist > cmp.dist;
   }

   int operator<(const DepthSortObject& cmp) {
      return dist < cmp.dist;
   }
};

#endif
