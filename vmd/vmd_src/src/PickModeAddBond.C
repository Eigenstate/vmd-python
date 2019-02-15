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
 *      $RCSfile: PickModeAddBond.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.19 $      $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The PickMode object which allows a pointer to be used to create new 
 * geometrical monitoring labels.  This particular version is used only for
 * adding molecular labels, i.e. Atoms, Bonds, Angles, and Dihedrals.  As
 * more atoms are selected, they are remembered until enough have been picked
 * to create the relevant label (i.e. 3 atoms to create an Angle label).
 *
 ***************************************************************************/

#include <math.h>
#include "PickModeAddBond.h"
#include "Pickable.h"
#include "DisplayDevice.h"
#include "Inform.h"
#include "DrawMolecule.h"
#include "CommandQueue.h"
#include "TextEvent.h"
#include "VMDApp.h"
#include "utilities.h"

PickModeAddBond::PickModeAddBond(VMDApp *vmdapp) 
: app(vmdapp) {

  needName = FALSE;

  // indicate we're still at the starting trying to find something
  haveItems = 0;
}

void PickModeAddBond::pick_molecule_start(DrawMolecule *mol, DisplayDevice *d,
			int, int tag, const int * /* cell */ , int dim, const float *pos) {
  // ignore the cell argument; we don't create bonds between images!
  atom = tag;
  memcpy(pPos, pos, dim*sizeof(float));
  needName = TRUE;
  
  int shift_pressed = d->shift_state() & DisplayDevice::SHIFT;
  app->commandQueue->runcommand(new PickAtomEvent(mol->id(), tag, 
    shift_pressed));
}

void PickModeAddBond::pick_molecule_end(DrawMolecule *m, DisplayDevice *) {

  if(needName) {
    // the selection was successful; first save the info for the object

    int id = m->id();
    molids[haveItems] = id; 
    atmids[haveItems] = atom;

    // every time an atom is selected, add an Atoms label
    app->label_add("Atoms", 1, &id, &atom, NULL, 0, 1);

    // indicate we have one more items in the ones we need
    haveItems++;

    // now check if we have enough items for the object we're out for
    if(haveItems >= 2) {
      if (molids[0] == molids[1]) {
        int id1 = atmids[0], id2 = atmids[1];
        MolAtom *atom1 = m->atom(id1), *atom2 = m->atom(id2);
        if (atom1->bonded(id2)) {
          int i;

          // Remove the bond
          for (i=0; i<atom1->bonds; i++) {
            if (atom1->bondTo[i] == id2) {
              for (int j=i; j<MAXATOMBONDS-1; j++)
                atom1->bondTo[j] = atom1->bondTo[j+1];
              atom1->bonds--;
              break;
            }
          }

          for (i=0; i<atom2->bonds; i++) {
            if (atom2->bondTo[i] == id1) {
              for (int j=i; j<MAXATOMBONDS-1; j++)
                atom2->bondTo[j] = atom2->bondTo[j+1];
              atom2->bonds--;
              break;
            }
          }
        } else {
          // Add the bond
          if (atom1->bonds >= MAXATOMBONDS ||
              atom2->bonds >= MAXATOMBONDS) {
            msgErr << "Unable to add bond: one or both atoms already has the maximum number." << sendmsg;
          } else {
            m->add_bond(id1, id2, 1, ATOMNORMAL);
          } 
        } 
        m->force_recalc(DrawMolItem::MOL_REGEN); // XXX many reps ignore bonds
      } else {
        msgErr << "Cannot add bond between two molecules." << sendmsg;
      } 
      // indicate we're done with this selection
      haveItems = 0;

      // when the user manipulates the bond info, we consider the bond
      // data to have become "valid", such that it will be saved out if
      // they write out files that can contain bond info.
      m->set_dataset_flag(BaseMolecule::BONDS);
    } 
  }
  needName = FALSE;
}

