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
 *      $RCSfile: GeometryAngle.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.29 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Measures the angle between three atoms, and draws a marker for the angle
 * into the display list for a given Displayable.
 *
 ***************************************************************************/

#include <stdio.h>

#include "GeometryAngle.h"
#include "MoleculeList.h"
#include "Molecule.h"
#include "utilities.h"
#include "DispCmds.h"


////////////////////////  constructor  /////////////////////////
GeometryAngle::GeometryAngle(int *m, int *a, const int *cell, MoleculeList *mlist, 
    CommandQueue *cq, Displayable *d)
	: GeometryMol(3, m, a, cell, mlist, cq, d) {

}


////////////////////  public virtual routines  //////////////////////

// recalculate the value of this geometry, and return it
float GeometryAngle::calculate(void) {

  // get coords to calculate distance 
  float pos1[3], pos2[3], pos3[3], r1[3], r2[3];
  if(!normal_atom_coord(0, pos1))
    return 0.0;
  if(!normal_atom_coord(1, pos2))
    return 0.0;
  if(!normal_atom_coord(2, pos3))
    return 0.0;

  vec_sub(r1, pos1, pos2);
  vec_sub(r2, pos3, pos2);
  return(geomValue = angle(r1, r2));
}


// draw the geometry marker in the given Displayable's drawing list
void GeometryAngle::create_cmd_list() {
  char valbuf[32];

  // get the transformed positions, and draw a line between them
  reset_disp_list();
  float pos1[3], pos2[3], pos3[3];
  if(!transformed_atom_coord(0, pos1))
    return;
  if(!transformed_atom_coord(1, pos2))
    return;
  if(!transformed_atom_coord(2, pos3))
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
  
  // print value of distance at midpoint
  midpoint(valuePos, pos1, pos3);
  midpoint(valuePos, valuePos, pos2);
  // left-align the value so that it doesn't appear to shift its position
  // when the label text size changes.  Shift it to the right by a constant
  // amount so that it doesn't intersect the line.
  valuePos[0] += 0.05f;
  sprintf(valbuf, "%-7.2f", geomValue);
  display_string(valbuf, cmdList);

}

void GeometryAngle::set_pick(void) {
  // set the Tcl values
  if (objIndex[0] == objIndex[1] && objIndex[1] == objIndex[2]) {
    set_pick_selection(objIndex[0], 3, comIndex);
  } else {
    set_pick_selection();
  }
  set_pick_value(geomValue);
}

