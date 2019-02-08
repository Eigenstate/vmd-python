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
 *	$RCSfile: Spaceball.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.31 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Spaceball UI object, which maintains the current state of the 
 * spaceball.  This code uses John Stone's spaceball driver library.
 *
 ***************************************************************************/
#ifndef SPACEBALL_OBJ_H
#define SPACEBALL_OBJ_H

#include "UIObject.h"
#include "Command.h"
#include "NameList.h"

#if defined(VMDLIBSBALL) && !defined(VMDSPACEWARE)
#include "sball.h" // spaceball I/O library header
#endif

/// UIObject subclass for Spaceball-based motion control
class Spaceball : public UIObject {
public:
  /// enum for Spaceball movement modes
  enum MoveMode { NORMAL, MAXAXIS, SCALING, ANIMATE, TRACKER, USER };

  /// gets a string representing a mode's name
  static const char *get_mode_str(MoveMode mode);

private:
#if defined(VMDLIBSBALL) && !defined(VMDSPACEWARE)
  SBallHandle sball; ///< handle from spaceball I/O library
#endif

  MoveMode moveMode; ///< the current move mode
  float sensitivity; ///< overall sensitivity scaling factor
  int maxstride;     ///< maximum stride when in animate mode
  float transInc;    ///< increment for translation
  float rotInc;      ///< increment for rotation
  float scaleInc;    ///< increment for scaling
  float animInc;     ///< increment for animation
  int null_region;   ///< null region value 
  int buttonDown;    ///< which buttons are down 

  /// tracker data reported to SpaceballTracker
  float trtx;
  float trty;
  float trtz;
  float trrx;
  float trry;
  float trrz;
  int trbuttons;

public:
  Spaceball(VMDApp *);      ///< constructor
  virtual ~Spaceball(void); ///< destructor
  
  //
  // virtual routines for UI init/display
  //
   
  /// reset the user interface (force update of all info displays)
  virtual void reset(void);
  
  /// update the display due to a command being executed.  Return whether
  /// any action was taken on this command.
  /// Arguments are the command type, command object, and the 
  /// success of the command (T or F).
  virtual int act_on_command(int, Command *); ///< command execute update
  
  /// check for and event, queue and return TRUE if one is found.  
  virtual int check_event(void);

  /// set the current mode
  int move_mode(MoveMode);

  /// set the sensitivity scaling factor
  void set_sensitivity(float s) {
    sensitivity = s;
  }

  /// set the null region, torque/displacement values below 
  /// which no activity will occur
  void set_null_region(int nr) {
    null_region = nr;
  }

  /// set the maximum animation stride allowed
  void set_max_stride(int ms) {
    maxstride = ms;
  }

  /// return the current spaceball event data, 
  /// used by the UIVR SpaceballTracker interface
  void get_tracker_status(float &tx, float &ty, float &tz, 
                          float &rx, float &ry, float &rz, int &buttons);
};


/// change the current mouse mode
/// This command doesn't generate an output text command, it is just
/// used to change the VMD internal state
class CmdSpaceballMode : public Command {
public:
  /// specify new mode and setting
  CmdSpaceballMode(int mm)
  : Command(SPACEBALL_MODE), spaceballMode(mm) {}

  /// mode and setting for the mouse
  int spaceballMode;
};

#endif

