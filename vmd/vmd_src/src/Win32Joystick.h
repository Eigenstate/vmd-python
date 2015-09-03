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
 *	$RCSfile: Win32Joystick.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.16 $	$Date: 2010/12/16 04:08:52 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   The Win32Joystick UI object, which maintains the current state of the 
 *   joystick.  
 ***************************************************************************/
#ifndef WIN32JOYSTICK_OBJ_H
#define WIN32JOYSTICK_OBJ_H

#include "UIObject.h"
#include "Command.h"
#include "NameList.h"
#include "windows.h"  // Base Windows types/includes
#include "mmsystem.h" // Windows joystick API

/// store low-level Win32 joystick status and config info
typedef struct {
  int exists;        ///< seems to exist
  int avail;         ///< is on and available, and returned data
  JOYCAPS caps;      ///< capability structure 
  JOYINFOEX info;    ///< current position information
  int xrange;        ///< x range (max - min)
  int yrange;        ///< y range (max - min)
  int zrange;        ///< z range (max - min)
  int vx;            ///< current x value (processed)
  int vy;            ///< current y value (processed)
  int vz;            ///< current z value (processed)
  int moveMode;      ///< current move mode for this stick
  int buttons;       ///< which buttons are down
  int buttonchanged; ///< which buttons have changed state
} vmdwinjoystick;
  

/// UIObject subclass to allow joystick-based motion control on MS-Windows
class Win32Joystick : public UIObject {
public:
  /// different available movement modes
  enum MoveMode { OFF, ROTATION, TRANSLATION, SCALING, USER};

private:
  int maxjoys;            // max joysticks supported by windows driver
  vmdwinjoystick *sticks; // array of joysticks
  float transInc;         // increment for translation
  float rotInc;           // increment for rotation
  float scaleInc;         // increment for scaling    
  float scaling;          // scale factor

public:
  Win32Joystick(VMDApp *);
  virtual ~Win32Joystick(void);
  
  //
  // virtual routines for UI init/display
  //
   
  /// reset the user interface (force update of all info displays)
  virtual void reset(void);
  
  /// update the display due to a command being executed.  Return whether
  /// any action was taken on this command.
  /// Arguments are the command type, command object, and the 
  /// success of the command (T or F).
  virtual int act_on_command(int, Command *);
  
  /// check for event, queue if found, return TRUE if an event was generated
  virtual int check_event(void);
};

#endif

