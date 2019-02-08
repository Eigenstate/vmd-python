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
 *      $RCSfile: GeometryAngle.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.25 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Measures the angle between three atoms, and draws a marker for the angle
 * into the display list for a given Displayable.
 *
 ***************************************************************************/
#ifndef GEOMETRYANGLE_H
#define GEOMETRYANGLE_H

#include "GeometryMol.h"

/// GeometryMol subclass to measure and display the angle between three atoms
class GeometryAngle : public GeometryMol {
public:
  /// constructor: molecule id's, atom indices, molecule list
  GeometryAngle(int *, int *, const int *cell, MoleculeList *, CommandQueue *, Displayable *);
  
  // public virtual routines
  virtual float calculate(void);  ///< recalculate the angle and return it
  virtual void create_cmd_list(); ///< draw the geometry marker 
  virtual void set_pick(void);    ///< use the TCL variables
};

#endif

