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
 *      $RCSfile: GeometryBond.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.28 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Measures the angle between two atoms, and draws a marker for the bond
 * into the display list for a given Displayable.
 *
 ***************************************************************************/

#include <stdio.h>

#include "GeometryBond.h"
#include "MoleculeList.h"
#include "Molecule.h"
#include "utilities.h"


////////////////////////  constructor  /////////////////////////
GeometryBond::GeometryBond(int *m, int *a, const int *cell, MoleculeList *mlist,
    CommandQueue *cq, Displayable *d)
	: GeometryMol(2, m, a, cell, mlist, cq, d) { }

// recalculate the value of this geometry, and return it
float GeometryBond::calculate(void) {

  // get coords to calculate distance 
  float pos1[3], pos2[3];
  if(!normal_atom_coord(0, pos1))
    return 0.0;
  if(!normal_atom_coord(1, pos2))
    return 0.0;

  vec_sub(pos2, pos2, pos1);
  return(geomValue = norm(pos2));
}


// draw the geometry marker in the given Displayable's drawing list
void GeometryBond::create_cmd_list() {
  char valbuf[32];

  // get the transformed positions, and draw a line between them
  float pos1[3], pos2[3];

  reset_disp_list();
  if(!transformed_atom_coord(0, pos1))
    return;
  if(!transformed_atom_coord(1, pos2))
    return;

  append(DMATERIALOFF);
  DispCmdColorIndex cmdColor;
  cmdColor.putdata(my_color, cmdList);

  DispCmdLineType cmdLineType;
  DispCmdLineWidth cmdLineWidth;
  cmdLineType.putdata(DASHEDLINE, cmdList);
  cmdLineWidth.putdata(4, cmdList);

  // print value of distance at midpoint
  midpoint(valuePos, pos1, pos2);
  // left-align the value so that it doesn't appear to shift its position
  // when the label text size changes.  Shift it to the right by a constant
  // amount so that it doesn't intersect the line.
  valuePos[0] += 0.05f;
  sprintf(valbuf, "%-7.2f", geomValue);
  display_string(valbuf, cmdList);

  // draw a line into the given Displayable
  display_line(pos1, pos2, cmdList);
  
}

void GeometryBond::set_pick(void) {

  // set the Tcl values
  if(objIndex[0] == objIndex[1]) {  // selections must be from the same molid
    set_pick_selection(objIndex[0], 2, comIndex);
  } else {
    set_pick_selection();  // if bad, set to the "nothing" molecule
  }
  set_pick_value(geomValue);
}

