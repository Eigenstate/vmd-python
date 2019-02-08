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
 *      $RCSfile: PickModeList.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.18 $       $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  List of all pick modes.
 ***************************************************************************/

#include "PickModeList.h"
#include "PickModeAddBond.h"
#include "PickModeCenter.h"
#include "PickModeForce.h"
#include "PickModeMolLabel.h"
#include "PickModeMove.h"
#include "PickModeUser.h"

#include "VMDApp.h"

PickModeList::PickModeList(VMDApp *app) {

  // IMPORTANT NOTE: The order in which the modes are loaded corresponds
  // to the number identifier for the picking modes (harcoded in Tcl scripts 
  // and in the Fltk interface). Until this, this is implemented more 
  // robustly, one must be careful _not_ to change the order of the picking
  // modes, and only add new modes at the end of the list, etc.
  
  pickmodelist.add_name("Query", new PickMode(app)); 
  pickmodelist.add_name("Center", new PickModeCenter(app));

  pickmodelist.add_name("Atoms", new PickModeAtoms(app));
  pickmodelist.add_name("Bonds", new PickModeBonds(app));
  pickmodelist.add_name("Angles", new PickModeAngles(app));
  pickmodelist.add_name("Dihedrals", new PickModeDihedrals(app));

  pickmodelist.add_name("MoveAtom", new PickModeMoveAtom);
  pickmodelist.add_name("MoveResidue", new PickModeMoveResidue);
  pickmodelist.add_name("MoveFragment", new PickModeMoveFragment);
  pickmodelist.add_name("MoveMolecule", new PickModeMoveMolecule);
 
  pickmodelist.add_name("ForceAtom", new PickModeForceAtom); 
  pickmodelist.add_name("ForceResidue", new PickModeForceResidue); 
  pickmodelist.add_name("ForceFragment", new PickModeForceFragment); 

  pickmodelist.add_name("MoveRep", new PickModeMoveRep);

  pickmodelist.add_name("AddBond", new PickModeAddBond(app));
  pickmodelist.add_name("Pick", new PickModeUser(app));

  curpickmode = pickmodelist.data("Query");
}

PickModeList::~PickModeList() {
  for (int i=0; i<pickmodelist.num(); i++)
    delete pickmodelist.data(i);
}


