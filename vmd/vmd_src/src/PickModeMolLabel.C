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
 *      $RCSfile: PickModeMolLabel.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.46 $      $Date: 2019/01/17 21:21:01 $
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
#include "PickModeMolLabel.h"
#include "Pickable.h"
#include "DisplayDevice.h"
#include "Inform.h"
#include "DrawMolecule.h"
#include "CommandQueue.h"
#include "TextEvent.h"
#include "VMDApp.h"
#include "utilities.h"

PickModeMolLabel::PickModeMolLabel(const char *nm, int size, VMDApp *vmdapp) 
: needItems(size), app(vmdapp) {

  modename = stringdup(nm);

  needName = FALSE;

  // save number of elements needed, and allocate storage
  molids = new int[size];
  atmids = new int[size];
  cells  = new int[3L*size];

  // indicate we're still at the starting trying to find something
  haveItems = 0;
}

PickModeMolLabel::~PickModeMolLabel(void) {

  delete [] molids;
  delete [] atmids;
  delete [] cells;
  delete [] modename;
}


void PickModeMolLabel::pick_molecule_start(DrawMolecule *mol, DisplayDevice *d,
			int, int tag, const int *cell, int dim, const float *pos) {
  atom = tag;
  // XXX shouldn't we be saving the molid as well???
  memcpy(pPos, pos, dim*sizeof(float));
  memcpy(lastCell, cell, 3*sizeof(int));
  needName = TRUE;
  
  int shift_pressed = d->shift_state() & DisplayDevice::SHIFT;
  app->commandQueue->runcommand(new PickAtomEvent(mol->id(), tag, 
    shift_pressed));
}

void PickModeMolLabel::pick_molecule_move(DrawMolecule *, DisplayDevice *,
			int, int dim, const float *pos) {

  // just keep track to see if the pointer moves any; if so, cancel action
  if(needName) {
    float mvdist = 0.0;
    for(int i=0; i < dim; i++)
      mvdist += (float) fabs(pPos[i] - pos[i]);

    if(mvdist > 0.01)
      needName = FALSE;
  }
}

void PickModeMolLabel::pick_molecule_end(DrawMolecule *m, DisplayDevice *) {

  if(needName) {
    // the selection was successful; first save the info for the object

    int id = m->id();
    molids[haveItems] = id; 
    atmids[haveItems] = atom;
    memcpy(cells+3*haveItems, lastCell, 3*sizeof(int));

    // every time an atom is selected, add an Atoms label
    app->label_add("Atoms", 1, &id, &atom, lastCell, 0.0f, 1);

    // indicate we have one more items in the ones we need
    haveItems++;

    // now check if we have enough items for the object we're out for
    if(haveItems >= needItems) {

      if(needItems > 1) {
        // for labels other than just atoms, add new monitor
        app->label_add(modename, needItems, molids, atmids, cells, 0.0f, 1);
      }

      // indicate we're done with this selection
      haveItems = 0;
    }
  }
  needName = FALSE;
}

void PickModeMolLabel::pick_graphics(int molid, int tag, int btn, DisplayDevice *d) {
  int shift_pressed = d->shift_state() & DisplayDevice::SHIFT;
  app->commandQueue->runcommand(new PickGraphicsEvent(molid, tag, btn, shift_pressed));
}

