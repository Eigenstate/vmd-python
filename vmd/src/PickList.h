/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: PickList.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.34 $	$Date: 2010/12/16 04:08:34 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The PickList object, which maintains a list of Pickable objects and
 * has the ability to find and deal with items picked by a pointer device.
 *
 * Each Scene is derived from PickList.
 *
 * Each PickList has a list of possible picking modes; these are created by
 * the request of Pickable objects, which then get notified when an
 * object is picked.  Only Pickables which are interested in the specified
 * mode are told of the picking result.
 *
 * For each pick mode, there is a corresponding PickMode object; PickMode
 * is a base class for the particular type of picking mode requested.
 * Pickable objects may request to have a new pick mode added to this list,
 * and if they do so they must provide an instance of the PickMode object
 * they require.  If the mode exists, the objects may replace the mode or
 * may just use the current one.  When modes are removed, the instance is
 * deleted.
 *
 * Only one picking operation may be in effect at a time.  A picking operation
 * consists of these steps:
 *	1. A pick is started by a pointer, by queueing a CmdPickStart command.
 *	2. pick_start is called, which determines if something is selected.
 *		If so, returns tag of item, and sets internal flags to
 *		indicate what is being picked, and how.
 *		2.1 Each Pickable which is interested in the current mode is
 *			told when the pick starts, and all later steps.
 *		2.2 The object for the current pick mode is also told of
 *			the start of the pick.
 *	3. pick_move is called whenever the pointer moves, by queueing the
 *		CmdPickMove command.  This continues until the picking is
 *		finished.  The PickMode object and interested Pickables are
 *		also told of the move.
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
#ifndef PICKLIST_H
#define PICKLIST_H

#include "ResizeArray.h"
#include "NameList.h"
#include "Command.h"
class Pickable;
class VMDApp;

/// Command subclass for picking events, used to notify the GUI of updates
class PickEvent : public Command {
public:
  Pickable *pickable;
  int tag;

  PickEvent(Pickable *p, int t)
  : Command(PICK_EVENT), pickable(p), tag(t) {}
};

/// Maintains a list of Pickable objects, find and deal with items picked 
/// by a pointer device.
class PickList {
private:
  int currPickDim;        ///< current picking button
  int currPickTag;        ///< current picking mode
  Pickable *currPickable; ///< last position

  int num_pickable(void) { return pickableObjs.num(); }
  Pickable *pickable(int n) { return pickableObjs[n]; }
  int picking(void) { return currPickable != NULL; }

  /// list of Pickable objects which contain pickable objects 
  ResizeArray<Pickable *> pickableObjs;
 
  VMDApp *app;
  
  /// The number of register picking clients
  int total_callback_clients;
 
public:
  /// constructor: no information needed
  PickList(VMDApp *);
  
  //
  // routines for registering Pickable objects
  //
  
  /// adds a Pickable to the current list, if it is not already in the list
  void add_pickable(Pickable *);

  /// remove the given pickable from the list
  void remove_pickable(Pickable *);
  
  //
  // routines to handle starting a pick, moving during a pick,
  // ending of a pick
  //

  /// using the given display, this checks to see if any object is under
  /// the given pointer position.  This does not set any internal picking
  /// flags, it just checks to see if something has been picked.  Returns
  /// a pointer to the Pickable object selected if successful, or NULL if
  /// nothing is picked.  If successful, returns 'tag' of item in final 
  /// argument.
  /// arguments: display device, dim, position, returned tag
  /// window_size tells pick how close picked items must be
  /// The optional argument <callback_client> instructs pick to send a
  /// TCL callback for the client with that name.
  /// If no callback client is registered, this does a normal PickEvent
  /// instead.
  /// Pass in a non-NULL argument for cell to get the unit cell information
  /// about the picked point.
  Pickable *pick_check(int, const float *, int &, int * cell,
		       float window_size, char *callback_client=NULL);

  /// called when a pick is begun: button, dim, pos.
  /// returns 'tag' of closest object, or (-1) if nothing is picked.
  /// When a pick is started, the internal flags for this object are set,
  /// and no other item can be picked until pick_end is called.
  /// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
  /// For 3D version: x,y,z are transformed position of pointer
  int pick_start(int btn, int dim, const float *);
  
  /// called when a pick moves, passing current pointer position
  /// Returns TRUE if a pick is currently active, FALSE otherwise.
  /// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
  /// For 3D version: x,y,z are transformed position of pointer
  int pick_move(const float *);
  
  /// called when a pick ends. 
  /// Returns TRUE if a pick is currently active, FALSE otherwise.
  /// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
  /// For 3D version: x,y,z are transformed position of pointer
  int pick_end();

  /// this notifies TCL that the pick is over - it is equivalent to
  /// performing a pick_check on a region not near any atoms.
  void pick_callback_clear(char *callback_client);
};

#endif

