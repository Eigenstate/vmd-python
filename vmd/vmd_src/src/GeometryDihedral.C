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
 *      $RCSfile: GeometryDihedral.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.31 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Measures the angle between four atoms, and draws a marker for the dihedral
 * into the display list for a given Displayable.
 *
 ***************************************************************************/

#include <stdio.h>

#include "GeometryDihedral.h"
#include "MoleculeList.h"
#include "Molecule.h"
#include "utilities.h"


////////////////////////  constructor  /////////////////////////
GeometryDihedral::GeometryDihedral(int *m, int *a, const int *cell, 
    MoleculeList *mlist, 
    CommandQueue *cq, Displayable *d)
	: GeometryMol(4, m, a, cell, mlist, cq, d) {

}



////////////////////  public virtual routines  //////////////////////

// recalculate the value of this geometry, and return it
float GeometryDihedral::calculate(void) {

  // get coords to calculate distance 
  float pos1[3], pos2[3], pos3[3], pos4[3]; 
  if(!normal_atom_coord(0, pos1))
    return 0.0;
  if(!normal_atom_coord(1, pos2))
    return 0.0;
  if(!normal_atom_coord(2, pos3))
    return 0.0;
  if(!normal_atom_coord(3, pos4))
    return 0.0;

  return (geomValue = dihedral(pos1, pos2, pos3, pos4));
}


// draw the geometry marker in the given Displayable's drawing list
void GeometryDihedral::create_cmd_list() {
  char valbuf[32];

  // get the transformed positions, and draw a line between them
  reset_disp_list();
  float pos1[3], pos2[3], pos3[3], pos4[3];
  if(!transformed_atom_coord(0, pos1))
    return;
  if(!transformed_atom_coord(1, pos2))
    return;
  if(!transformed_atom_coord(2, pos3))
    return;
  if(!transformed_atom_coord(3, pos4))
    return;

  DispCmdColorIndex cmdColor;
  cmdColor.putdata(my_color, cmdList);

  DispCmdLineType cmdLineType;
  DispCmdLineWidth cmdLineWidth;
  cmdLineType.putdata(DASHEDLINE, cmdList);
  cmdLineWidth.putdata(4, cmdList);

  // draw a line into the given Displayable
  display_line(pos1, pos2, cmdList);
  display_line(pos2, pos3, cmdList);
  display_line(pos3, pos4, cmdList);
  
  // print value of distance at midpoint
  midpoint(valuePos, pos2, pos3);
  // left-align the value so that it doesn't appear to shift its position
  // when the label text size changes.  Shift it to the right by a constant
  // amount so that it doesn't intersect the line.
  valuePos[0] += 0.05f;
  sprintf(valbuf, "%-7.2f", geomValue);
  display_string(valbuf, cmdList);
}

void GeometryDihedral::set_pick(void) {
  // set the Tcl values
  if (objIndex[0] == objIndex[1] && objIndex[1] == objIndex[2] &&
      objIndex[2] == objIndex[3]) {
    set_pick_selection(objIndex[0], 4, comIndex);
  } else {
    set_pick_selection();
  }
  set_pick_value(geomValue);
}

