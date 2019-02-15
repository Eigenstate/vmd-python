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
 *      $RCSfile: GeometrySpring.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.19 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Simulates and draws a spring between two atoms in IMD.
 *
 ***************************************************************************/

#include <stdio.h>

#include "GeometrySpring.h"
#include "MoleculeList.h"
#include "Molecule.h"
#include "utilities.h"


////////////////////////  constructor  /////////////////////////
GeometrySpring::GeometrySpring(int *m, int *a, MoleculeList
			       *mlist,CommandQueue *cq, float thek, Displayable *d)
	: GeometryMol(2, m, a, NULL, mlist, cq, d) {
  k = thek;
  rvec[0]=0;
  rvec[1]=0;
  rvec[2]=0;
}



////////////////////  public virtual routines  //////////////////////

// recalculate the value of this geometry, and return it
float GeometrySpring::calculate(void) {

  // get coords to calculate distance 
  float pos1[3], pos2[3];
  if(!normal_atom_coord(0, pos1))
    return 0.0;
  if(!normal_atom_coord(1, pos2))
    return 0.0;

  vec_sub(pos2, pos2, pos1);

  // XXX this is a hack
  // decaying average for smooth springs
  vec_scale(rvec,0.95f,rvec);
  vec_scale(pos2,0.05f,pos2);
  vec_add(rvec,rvec,pos2);
  geomValue = norm(rvec);

  return(geomValue);
}

void GeometrySpring::prepare() {
  // get the molecules and atoms
  int mol1 = objIndex[0];
  int mol2 = objIndex[1];
  int atom1 = comIndex[0];
  int atom2 = comIndex[1];
  Molecule *m1 = molList->mol_from_id(mol1);
  Molecule *m2 = molList->mol_from_id(mol2);

  // find the appropriate force between these two
  float force1[3],force2[3];
  vec_scale(force1,k,rvec);
  vec_scale(force2,-k,rvec);
  
  // apply it to the atoms
  m1->addForce(atom1,force1);
  m2->addForce(atom2,force2);
}

// draw the geometry marker in the given Displayable's drawing list
void GeometrySpring::create_cmd_list() {
  char valbuf[32];

  // get the transformed positions, and draw a line between them
  reset_disp_list();
  float pos1[3], pos2[3];
  if(!transformed_atom_coord(0, pos1))
    return;
  if(!transformed_atom_coord(1, pos2))
    return;

  // draw a line into the given Displayable
  display_line(pos1, pos2, cmdList);
  
  // print value of distance at midpoint
  midpoint(valuePos, pos1, pos2);
  // left-align the value so that it doesn't appear to shift its position
  // when the label text size changes.  Shift it to the right by a constant
  // amount so that it doesn't intersect the line.
  valuePos[0] += 0.05f;
  sprintf(valbuf, "-%7.2f", geomValue);
  display_string(valbuf, cmdList);

}

void GeometrySpring::set_pick(void) {

  // set the Tcl values
  if(objIndex[0] == objIndex[1]) {  // selections must be from the same molid
    set_pick_selection(objIndex[0], 2, comIndex);
  } else {
    set_pick_selection();  // if bad, set to the "nothing" molecule
  }
  set_pick_value(geomValue);
}

GeometrySpring::~GeometrySpring() {
}
