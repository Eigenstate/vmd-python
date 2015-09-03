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
 *      $RCSfile: GeometrySpring.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.13 $      $Date: 2010/12/16 04:08:18 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Simulates and draws a spring between two atoms in IMD.
 *
 ***************************************************************************/
#ifndef GEOMETRYSPRING_H
#define GEOMETRYSPRING_H

#include "GeometryMol.h"

/// GeometryMol subclass to simulate and draw a spring between two atoms in IMD
class GeometrySpring : public GeometryMol {
private:
  float k;
  float rvec[3];

public:
  /// constructor: molecule id's, atom indices, molecule list, spring constant
  GeometrySpring(int *, int *, MoleculeList *, CommandQueue *, float thek,
      Displayable *);
  
  //
  // public virtual routines
  //
  virtual float calculate(void);  /// recalculate spring value, and return it
  virtual void create_cmd_list(); ///< draw the geometry marker
  virtual void set_pick(void);    ///< use the TCL variables

  ~GeometrySpring();
  void prepare();
};

#endif

