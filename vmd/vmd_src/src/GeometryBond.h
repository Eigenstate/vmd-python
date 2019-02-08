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
 *      $RCSfile: GeometryBond.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.25 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Measures the distance between two atoms, and draws a marker for the bond
 * into the display list for a given Displayable.
 *
 ***************************************************************************/
#ifndef GEOMETRYBOND_H
#define GEOMETRYBOND_H

#include "GeometryMol.h"

/// GeometryMol subclass to measure and display bond lengths and distances
class GeometryBond : public GeometryMol {
public:
  /// constructor: molecule id's, atom indices, molecule list
  GeometryBond(int *, int *, const int *cell, MoleculeList *, CommandQueue *, Displayable *);
  
  // public virtual routines
  virtual float calculate(void);  ///< recalc bond length/distance and return it
  virtual void create_cmd_list(); ///< draw the geometry marker
  virtual void set_pick(void);    ///< use the TCL variables
};

#endif

