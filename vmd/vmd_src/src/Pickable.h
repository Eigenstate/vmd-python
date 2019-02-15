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
 *	$RCSfile: Pickable.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.34 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * A Pickable object is one which contains data which may be selected by
 * using a pointer device in an open DisplayDevice.  Each Pickable registers
 * itself with a PickList object which indicates it has items in its
 * drawing list which may be picked, and which indicates it should be told
 * when something has been successfully selected.  There are one or many
 * picking modes, which are represented by derivations of the PickMode class.
 * When picking is done, it is done while in a current pick mode, which
 * controls what to do with the information.
 *
 * A Pickable must provide versions of virtual routines which tell what
 * pick modes the object is interested in.
 ***************************************************************************/
#ifndef PICKABLE_H
#define PICKABLE_H

class PickMode;
class DisplayDevice;
class VMDDisplayList;

/// contains data which may be selected with a pointer in a DisplayDevice
class Pickable {
public:
  Pickable() {}
  virtual ~Pickable() {}

  /// return our list of draw commands with picking draw commands in them
  virtual VMDDisplayList *pick_cmd_list() { return 0; }
  
  /// return whether the pickable object is being displayed
  virtual int pickable_on() { return 1; }

  //
  // public virtual pick action routines
  //

  /// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
  /// For 3D version: x,y,z are transformed position of pointer
  virtual void pick_start(PickMode *, DisplayDevice *, 
                          int /* btn */, int /* tag */, 
                          const int *cell /* [3] */,
                          int /* dim */, const float *) {}
  virtual void pick_move (PickMode *, DisplayDevice *,
                          int /* tag */, int /* dim */, const float *) {}
  virtual void pick_end  (PickMode *, DisplayDevice *) {}
};

#endif

