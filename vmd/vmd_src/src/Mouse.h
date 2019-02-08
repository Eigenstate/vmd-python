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
 *	$RCSfile: Mouse.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.69 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Mouse UI object, which maintains the current state of the 
 * mouse, including what it is currently used for, and how much it has moved
 * from one measurement to the next.  This also deals with any pop-up or
 * pull-down menus available by using the mouse, as well as picking objects.
 *
 * A three-button mouse is assumed here, with the following usage:
 *	1) Buttons 1 and 2 : manipulation and picking.
 *	2) Button 3 (right): pop-up menu
 *
 * This is the general base class definition; specific versions for each
 * windowing/graphics system may be supplied.  This class can function as
 * it is, however, and will just be a 'zero' mouse - the buttons will never
 * be pressed, and the cursor will always be at 0,0.
 *
 ***************************************************************************/
#ifndef MOUSE_OBJ_H
#define MOUSE_OBJ_H

#include "UIObject.h"
#include "Command.h"
#include "NameList.h"

/// UIObject subclass implementing Mouse-based motion control, picking, etc
class Mouse : public UIObject {
public:
  /// different available mouse movement/picking modes
  enum MoveMode {ROTATION, TRANSLATION, SCALING, LIGHT, USERPOINT, 
    // All "picking" modes need to be listed after PICK, 
    // and non-picking modes need to be listed before
    // This is because later code does a check on 
    // (enum value) >= PICK to determine if we are in 
    // a "picking" mode
    PICK, QUERY, CENTER, \
    LABELATOM, LABELBOND, LABELANGLE, LABELDIHEDRAL, \
    MOVEATOM, MOVERES, MOVEFRAG, MOVEMOL, MOVEREP,\
    FORCEATOM, FORCERES, FORCEFRAG, \
    ADDBOND};

  enum MouseButton { B_NONE = 0, B_LEFT = 1, B_MIDDLE, B_RIGHT };

  /// gets a string representing a mode's name
  static const char *get_mode_str(MoveMode mode);

private:
  MoveMode moveMode;          ///< the current move mode
  MouseButton pickInProgress; ///< active-pick-in-progress flag.
      // If zero, no picking operation is currently in progress.
      // If something is currently being selected with the mouse
      // and the button is still down, this flag indicates which button is 
      // used.  In fact, pickInProgress = pick-button + 1. 
  int moveObj;                ///< object mouse is affecting (if mode == LIGHT)
  int currX, currY;           ///< current position, in pixels from lower-left
  int oldX, oldY;             ///< last position, in pixels from lower-left
  MouseButton buttonDown;     ///< ORed result of pressed mouse buttons
 
  /// increment for mouse translation, rotation, scaling
  float transInc, rotInc, scaleInc;

  /// rotational velocities, and scaling factor, and flag for whether rotating
  float xRotVel, yRotVel, zRotVel, scaling, RotVelScale;

  /// flag for enabling/disabling rocking with the mouse
  int rocking_enabled;

  /// check mouse and take appropriate actions for built-in VMD mouse modes
  int mouse_moved(void);

  /// check mouse and call callbacks for user-defined behaviors, flying, etc
  int mouse_userpoint(void);

  /// handle events from the display device
  void handle_winevent(long, long); 
  
public:
  Mouse(VMDApp *);
  virtual ~Mouse(void);
  
  /// reset the mouse interface (force update of all info displays)
  virtual void reset(void);
  
  /// set the current move mode
  int move_mode(MoveMode, int = 0);
 
  /// stop rotation of object
  void stop_rotation(void);

  /// check for events, queue any found, return TRUE if an event was generated
  virtual int check_event(void);

  /// turn on/off rocking
  void set_rocking(int on);
};


/// change the current mouse mode
/// This command doesn't generate an output text command, it is just
/// used to change the VMD internal state
class CmdMouseMode : public Command {
public:
  /// specify new mode and setting
  CmdMouseMode(int mm, int ms)
  : Command(MOUSE_MODE), mouseMode(mm), mouseSetting(ms) {}

  /// mode and setting for the mouse
  int mouseMode, mouseSetting;
};

#endif

