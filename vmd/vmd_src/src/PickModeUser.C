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
 *      $RCSfile: PickModeUser.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.8 $      $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The PickMode object which allows a pointer to be used to make selections
 * that interact with user-defined scripts and plugins.  No labels are added.
 *
 ***************************************************************************/

#include <math.h>
#include "PickModeUser.h"
#include "Pickable.h"
#include "DisplayDevice.h"
#include "Inform.h"
#include "DrawMolecule.h"
#include "CommandQueue.h"
#include "TextEvent.h"
#include "VMDApp.h"
#include "utilities.h"

PickModeUser::PickModeUser(VMDApp *vmdapp) 
: app(vmdapp) {
  needName = FALSE;

  // indicate we're still at the starting trying to find something
  haveItems = 0;
}

void PickModeUser::pick_molecule_start(DrawMolecule *mol, DisplayDevice *d,
			int, int tag, const int * /* cell */ , int dim, const float *pos) {
  // ignore the cell argument; we don't create bonds between images!
  atom = tag;
  memcpy(pPos, pos, dim*sizeof(float));
  needName = TRUE;
  
  int shift_pressed = d->shift_state() & DisplayDevice::SHIFT;
  app->commandQueue->runcommand(new PickAtomEvent(mol->id(), tag, 
    shift_pressed, true));
}

void PickModeUser::pick_molecule_end(DrawMolecule *m, DisplayDevice *) {

  if(needName) {
    // the selection was successful; first save the info for the object

    int id = m->id();
    molid = id; 
    atmid = atom;

    // indicate we have one more items in the ones we need
    haveItems++;

    // now check if we have enough items for the object we're out for
    if(haveItems >= 1) {
      msgInfo << "User Pick: mol" << molid << " atom:" << atmid << sendmsg;
      // indicate we're done with this selection
      haveItems = 0;
    } 
  }
  needName = FALSE;
}

