#ifndef NANOSHAPERINTERFACE_H
#define NANOSHAPERINTERFACE_H

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
 *	$RCSfile: NanoShaperInterface.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.7 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Communicate with the NanoShaper surface generation program.  For more
 *   information about NanoShaper, please see:
 *
 ***************************************************************************/

// NanoShaper surface generation process:
//   Send coords and get back surface information
//   Pass this data to NanoShaper server:
//     index, x, y, z, radius
//
//   NanoShaper server returns this data:
//     face list containing 3 vertex points each
//     atomid, as mapped to input values
//     position list containing x, y, z
//     norm list containing normx, normy, normz

#include "ResizeArray.h"

/// structure containing NanoShaper vertex coordinates
struct NanoShaperCoord {
  float x[3];       ///< floating point xyz coordinates 
  int operator==(const NanoShaperCoord& c) {
    return !memcmp(x, c.x, 3L*sizeof(float));
  }
};

/// structure containing NanoShaper facet information
struct NanoShaperFace {
  int vertex[3];    ///< 1-based vertex indices

  int surface_type; ///< 1 - triangle in a toric reentrant face
                    ///< 2 - triangle in a sphereic reentrant face
                    ///< 3 - triangle in a contact face   

  int anaface;      ///< 1-based face number in the analytical description
                    ///< of the solvent excluded surface

  int component;    ///< which surface is it in?

  int operator==(const NanoShaperFace &f) {
    return (!memcmp(vertex, f.vertex, 3L*sizeof(float)) &&
                    surface_type==f.surface_type && anaface==f.anaface &&
                    component==f.component);
  }
};

/// Manages communication with the NanoShaper surface generation program
/// Can only use this class once!
class NanoShaperInterface {
public:
  /// return 1 on success
  enum {BAD_RANGE = -2, NO_PORTS = -3, NO_CONNECTION = -4,
	NO_INITIALIZATION = -5, NANOSHAPER_DIED = -6, COMPUTED = 1};

  enum {NS_SURF_SES = 0,
        NS_SURF_SKIN = 1, 
        NS_SURF_BLOBBY = 2,
        NS_SURF_POCKETS = 3};

  /// free memory in the ResizeArrays.
  void clear();

  // use file interface instead of sockets
  int compute_from_file(int surftype, float gspacing,
                        float probe_radius, float skin_parm, float blob_parm,
                        int n, int *ids, float *xyzr, int *flgs);

  int err;                               ///< was there an error?
  NanoShaperInterface(void) { err = 0; }
  ResizeArray<int>             atomids;  ///< map vertices to atoms
  ResizeArray<NanoShaperFace>  faces;    ///< surface facet list
  ResizeArray<NanoShaperCoord> coords;   ///< vertex list referenced by facets
  ResizeArray<NanoShaperCoord> norms;    ///< normal list referenced by facets
};
  
#endif

