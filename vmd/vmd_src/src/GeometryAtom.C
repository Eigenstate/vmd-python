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
 *      $RCSfile: GeometryAtom.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.29 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Draws a marker for a specified atom into the display list for a Displayable
 *
 ***************************************************************************/

#include "GeometryAtom.h"
#include "MoleculeList.h"
#include "Molecule.h"


////////////////////////  constructor  /////////////////////////
GeometryAtom::GeometryAtom(int m, int a, const int *cell, MoleculeList *mlist, 
    CommandQueue *cq, Displayable *d)
	: GeometryMol(1, &m, &a, cell, mlist, cq, d) {

  // indicate this object does not have a value
  hasValue = FALSE;
}



////////////////////  public virtual routines  //////////////////////

// draw the geometry marker in the given Displayable's drawing list
void GeometryAtom::create_cmd_list() {

  reset_disp_list();
  // get the molecule pointer and atom position
  Molecule *mol = transformed_atom_coord(0, valuePos);
  
  // do not draw if illegal molecule, or atom is not on
  if(!mol)
    return;

  append(DMATERIALOFF);
  DispCmdColorIndex cmdColor;
  cmdColor.putdata(my_color,cmdList);
  // everything is OK, draw text at atom position
  JString str;
  atom_formatted_name(str, mol, comIndex[0]);
  display_string((const char *)str, cmdList);
}

void GeometryAtom::set_pick(void) { 
  // and set the Tcl variable "pick_selection" to the selection
  set_pick_selection(objIndex[0], 1, comIndex);
}


