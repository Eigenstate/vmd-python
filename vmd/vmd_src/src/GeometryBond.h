/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: GeometryBond.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.23 $      $Date: 2010/12/16 04:08:17 $
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

