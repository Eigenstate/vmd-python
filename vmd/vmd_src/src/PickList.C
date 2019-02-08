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
 *	$RCSfile: PickList.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.49 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The PickList object, which maintains a list of Pickable objects and
 * has the ability to find and deal with items picked by a pointer device.
 *
 * Only one picking operation may be in effect at a time.  A picking operation
 * consists of these steps:
 *	1. A pick is started by a pointer, by queueing a CmdPickStart command.
 *	2. pick_start is called, which determines if something is selected.
 *		If so, returns tag of item, and sets internal flags to
 *		indicate what is being picked, and how.
 *              The current PickMode and relevant picking parameters are
 *              passed to the picked object.
 *	3. pick_move is called whenever the pointer moves, by queueing the
 *		CmdPickMove command.  This continues until the picking is
 *		finished.  
 *	4. pick_end is called by queueing CmdPickEnd; this behaves similarly
 *		the pick_move.  When finished, the internal flags are reset
 *		to indicate that picking is finished, and a new one can begin.
 * NOTE: multiple concurrent picks could be implemented later by providing
 * an object which maintains separate sets of the internal flags and variables,
 * for each picking device.  This would also require another argument to the
 * pick_* routines, to indicate which picking device is being used.  Also,
 * PickMode and Pickable object involved in another picking operation would
 * have to be excluded from being informed of other picking operations.
 ***************************************************************************/

#include "PickList.h"
#include "DisplayDevice.h"
#include "Pickable.h"
#include "PickModeList.h"
#include "PickMode.h"
#include "ResizeArray.h"
#include "NameList.h"
#include "Inform.h"
#include "VMDApp.h"
#include "CommandQueue.h"
#include "TextEvent.h"
#include "Molecule.h"
#include "MoleculeList.h"

// forward declaration
static void print_atom_info(VMDApp *app, PickEvent *event);

//////////////////////////// constructor  ////////////////////////////
PickList::PickList(VMDApp *vmdapp) : pickableObjs(32), app(vmdapp) {

  // reset all current picking flags
  currPickDim = currPickTag = (-1);
  currPickable = NULL;
  app->pickModeList->set_pick_mode(2);
  total_callback_clients = 0; // no clients yet!
}

//////////////////////////// public routines  ////////////////////////////

// adds a Pickable to the current list, if it is not already in the list
void PickList::add_pickable(Pickable *p) {

  // find if this is in the list already.
  int indx = pickableObjs.find(p);
  
  // if it is not, append it
  if(indx < 0) {
    pickableObjs.append(p);
  }
}


// remove the given pickable from the list; return TRUE if it was in the list
void PickList::remove_pickable(Pickable *p) {
  int ind = pickableObjs.find(p);

  // check if we have a valid Pickable index
  if (ind >= 0) {
    // cancel any active picking state if we're deleting the
    // pickable that's currently active
    if (picking() && (currPickable == p)) {
      // reset the status variables to null out the active pick
      currPickDim = currPickTag = (-1);
      currPickable = NULL;
    }

    pickableObjs.remove(ind);
  }
}

/////////////////////////////////////////////////////////////////////
// routines to handle starting a pick, moving during a pick,
// ending of a pick
/////////////////////////////////////////////////////////////////////
void PickList::pick_callback_clear(char *callback_client){
  app->commandQueue->runcommand(new PickAtomCallbackEvent(-1,-1,callback_client));
}

Pickable *PickList::pick_check(int dim, const float
			       *pos, int &tag, int *cell, float window_size,
			       char *callback_client){
  
//  printf("doing a pickcheck\n");
  if(pos==NULL) {
    msgErr << "pick_check called with NULL pos" << sendmsg;
    return NULL;
  }

  Pickable *currObj, *retobj = NULL;
  float eyedist = (-1.0);
  int currtag, rettag = (-1);
  int i, np = num_pickable();

  if(!np)
    return NULL;

  // use left eye settings for picking; if not stereo, will just be normal
  app->display->left();
      
  // for all Pickable objects, check to see if they have a picked object
  for(i=0; i < np; i++) {
    currObj = pickable(i);
    if(currObj->pickable_on()) {
      currtag = app->display->pick(dim, pos, currObj->pick_cmd_list(), eyedist,
			cell, window_size);
      if(currtag != -1) {
        // a new object closer to the eye position was found.  Save it.
        retobj = currObj;
        rettag = currtag;
      }
    }
  }
  
  // clean up after setting stereo mode, but do not do buffer swap
  app->display->update(FALSE);
      
  // for now, only check left eye.  Can later see if checking right eye helps
  // as well.
  
  // finished; report results
  if(callback_client != NULL) {
    if(retobj) {
      int mol,atom;
      Molecule *m = app->moleculeList->check_pickable(retobj);
      if (m) {
	mol = m->id();
	atom = rettag;
	app->commandQueue->runcommand(new PickAtomCallbackEvent(mol,atom,callback_client));
      }
    }
    else { // we didn't find anyhing
      app->commandQueue->runcommand(new PickAtomCallbackEvent(-1,-1,callback_client));
    }
  }

  if(retobj) {
    tag = rettag;
    // send a normal pick event if this is not a check for callbacks
    if(callback_client==NULL) {
      PickEvent *event = new PickEvent(retobj, tag);
      print_atom_info(app, event);
      app->commandQueue->runcommand(event);
    }
  }
  return retobj;
}


// called when a pick is begun: display device, button, mode, dim, pos
// returns 'tag' of closest object, or (-1) if nothing is picked.
// When a pick is started, the internal flags for this object are set,
// and no other item can be picked until pick_end is called.
// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
// For 3D version: x,y,z are transformed position of pointer
int PickList::pick_start(int b, int dim,
			 const float *pos) {
  Pickable *closePickable;
  int tag = (-1); 
  int cell[3];
  float window_size = 0.01f;
  if (dim == 3) window_size *= 5;

  // make sure we're not already picking something
  if(picking())
    return (-1);

  cell[0] = cell[1] = cell[2] = 0;

  // check if something has been actually picked
  if((closePickable = pick_check(dim, pos, tag, cell, window_size)) != NULL) {
    // set all variables to show that we're picking something
    currPickDim = dim;
    currPickTag = tag;
    currPickable = closePickable;
    
    // use left eye settings for picking; if not stereo, will just be normal
    app->display->left();
  
//    printf("pick start got tag %d\n", tag);
    PickMode *pm = app->pickModeList->current_pick_mode(); 
    if (pm != NULL)
      closePickable->pick_start(pm, app->display, b, currPickTag, cell, dim, pos);

    // clean up after setting stereo mode, but do not do buffer swap
    app->display->update(FALSE);   
  }
  return tag;
}


// called when a pick moves: display device, button, mode, dim, pos
// Returns TRUE if a pick is currently active, FALSE otherwise.
// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
// For 3D version: x,y,z are transformed position of pointer
int PickList::pick_move(const float *pos) {

  // make sure we're already picking something
  if(!picking() )
    return FALSE;

  // use left eye settings for picking; if not stereo, will just be normal
  app->display->left();
  
  currPickable->pick_move(app->pickModeList->current_pick_mode(), app->display, currPickTag,
                          currPickDim, pos);
  // clean up after setting stereo mode, but do not do buffer swap
  app->display->update(FALSE);
      
  return TRUE;
}


// called when a pick ends: display device, button, mode, dim, pos
// Returns TRUE if a pick is currently active, FALSE otherwise.
// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
// For 3D version: x,y,z are transformed position of pointer
int PickList::pick_end() {
  // make sure we're already picking something
  if(!picking() )
    return FALSE;

  // use left eye settings for picking; if not stereo, will just be normal
  app->display->left();

  currPickable->pick_end(app->pickModeList->current_pick_mode(), app->display);
 
  // clean up after setting stereo mode, but do not do buffer swap
  app->display->update(FALSE);
      
  // reset all current status variables, and return
  currPickDim = currPickTag = (-1);
  currPickable = NULL;
  
  return TRUE;
}


// print atom info to the console
static void print_atom_info(VMDApp *app, PickEvent *event) {
  Molecule *mol = app->moleculeList->check_pickable(event->pickable);
  if (!mol) return;
  int atomindex = event->tag;
  if (atomindex >= mol->nAtoms) return;

  MolAtom *a = mol->atom(atomindex);

  msgInfo << "picked atom: \n------------\n"
          << "molecule id: " << mol->id()
          << "\ntrajectory frame: " << app->molecule_frame(mol->id())
          << "\nname: " << mol->atomNames.name(a->nameindex)
          << "\ntype: " << mol->atomTypes.name(a->typeindex)
          << "\nindex: " << atomindex
          << "\nresidue: " << a->uniq_resid
          << "\nresname: " << mol->resNames.name(a->resnameindex)
          << "\nresid: " << a->resid
          << "\nchain: " << mol->chainNames.name(a->chainindex)
          << "\nsegname: " << mol->segNames.name(a->segnameindex)
          << "\nx: " << mol->current()->pos[3L*atomindex]
          << "\ny: " << mol->current()->pos[3L*atomindex+1]
          << "\nz: " << mol->current()->pos[3L*atomindex+2] << "\n"
          << sendmsg;
}
