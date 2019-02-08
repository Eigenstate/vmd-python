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
 *      $RCSfile: GeometryDihedral.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.25 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Measures the angle between four atoms, and draws a marker for the dihedral
 * into the display list for a given Displayable.
 *
 ***************************************************************************/
#ifndef GEOMETRYDIHE_H
#define GEOMETRYDIHE_H

#include "GeometryMol.h"

/// GeometryMol subclass to measure and display dihedral angle between 4 atoms
class GeometryDihedral : public GeometryMol {

public:
  /// constructor: molecule id's, atom indices, molecule list
  GeometryDihedral(int *, int *, const int *cell, MoleculeList *, CommandQueue *, Displayable *);
  
  // public virtual routines
  virtual float calculate(void);  ///< recalculate dihedral angle and return it
  virtual void create_cmd_list(); ///< draw the geometry marker
  virtual void set_pick(void);    ///< use the TCL variables
};

#endif

