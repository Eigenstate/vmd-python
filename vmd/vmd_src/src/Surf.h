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
 *	$RCSfile: Surf.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.24 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * Interface to the SURF executable written by Amitabh Varshney at UNC.  
 * SURF Downloadable from ftp://ftp.cs.unc.edu/pub/projects/GRIP/SURF/
 *
 ***************************************************************************/
#ifndef SURF_H
#define SURF_H

#include "ResizeArray.h"

/// Interface to the SURF solvent accessible surface package
class Surf {
public:
  int numtriangles;      ///< number of triangles in the facet list.
  ResizeArray<float>  v; ///< vertices
  ResizeArray<float>  n; ///< normals
  ResizeArray<int>    f; ///< facets
  ResizeArray<int>  ind; ///< facet-to-atom index map
   
public:
  Surf();

  /// return 1 on success, 0 on fail
  /// takes the probe radius and the array of x,y,z,r values
  int compute(float probe_r, int num_points, float *r, 
               float *x, float *y, float *z);

  void clear();          ///< free up triangle mesh memory 
};
#endif

