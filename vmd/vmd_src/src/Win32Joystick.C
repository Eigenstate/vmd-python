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
 *	$RCSfile: Win32Joystick.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.25 $	$Date: 2011/01/28 17:06:54 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   The Win32Joystick UI object, which maintains the current state of the 
 *   joystick.
 ***************************************************************************/

#include "Win32Joystick.h"
#include "DisplayDevice.h"
#include "TextEvent.h"
#include "CommandQueue.h"
#include "Inform.h"
#include "PickList.h"
#include "VMDApp.h"
#include "stdlib.h" // for getenv()

// constructor
Win32Joystick::Win32Joystick(VMDApp *vmdapp)
	: UIObject(vmdapp) {
  MMRESULT mres;
  int i, numsticks;
  sticks = NULL;

  // max joysticks the driver supports, returns 0 if no driver, or failure
  maxjoys = joyGetNumDevs(); 

  if (maxjoys > 0) {
    sticks = (vmdwinjoystick *) malloc(maxjoys * sizeof(vmdwinjoystick));

    if (sticks != NULL) {
      numsticks = 0; 
      for (i=0; i<maxjoys; i++) {
        memset(&(sticks[i]), 0, sizeof(vmdwinjoystick));
        mres = joyGetDevCaps(i, &sticks[i].caps, sizeof(JOYCAPS));
    
        if (mres == JOYERR_NOERROR) {
          msgInfo << "Joystick " << i << ", " << ((int) sticks[i].caps.wNumAxes) << " axes, " << ((int) sticks[i].caps.wNumButtons) << " buttons, ";

          if (sticks[i].caps.szPname == NULL) 
             msgInfo << "type unknown." << sendmsg;
          else 
             msgInfo << sticks[i].caps.szPname << sendmsg;

          sticks[i].xrange = sticks[i].caps.wXmax - sticks[i].caps.wXmin; 
          sticks[i].yrange = sticks[i].caps.wYmax - sticks[i].caps.wYmin; 
          sticks[i].zrange = sticks[i].caps.wZmax - sticks[i].caps.wZmin; 

          sticks[i].moveMode = OFF; // set joystick off by default, in case
                                    // its wacko, this prevents a bad joystick
                                    // from interfering with VMD unless the
                                    // user intentionally enables it.
          sticks[i].exists = TRUE;
          numsticks++;
        } else {
          sticks[i].exists = FALSE;
        }
      }
    }
  }

  if (numsticks < 1 || maxjoys < 1) {
    msgInfo << "No joysticks found.  Joystick interface disabled." << sendmsg; 
  }

  // set the default translation and rotation increments
  transInc = 1.0f / 4000.0f;
    rotInc = 1.0f / 500.0f;
  scaleInc = 1.0f / 4000.0f;
  reset();
}


// destructor
Win32Joystick::~Win32Joystick(void) {
  if (sticks != NULL)
    free(sticks);
}

/////////////////////// virtual routines for UI init/display  /////////////
   
// reset the joystick to original settings
void Win32Joystick::reset(void) {
  scaling = 1.0;
}

// update the display due to a command being executed.  Return whether
// any action was taken on this command.
// Arguments are the command type, command object, and the 
// success of the command (T or F).
int Win32Joystick::act_on_command(int type, Command *cmd) {
  return FALSE; // we don't take any commands presently
}


// do null region processing on raw joystick values
static int nullregion(int null, int val) {
  if (abs(val) > null) {
    return ((val > 0) ? (val - null) : (val + null));
  }
  return 0;
}


// check for an event, and queue it if found.  Return TRUE if an event
// was generated.
int Win32Joystick::check_event(void) {
  int retval = FALSE;
  int rx, ry, rz, tx, ty, tz;
  int i;
  MMRESULT mres;
  float scf;
  // for use in UserKeyEvent() calls
  DisplayDevice::EventCodes keydev=DisplayDevice::WIN_KBD;
  
  if (maxjoys < 1 || sticks == NULL)
    return FALSE; // joysticks disabled 

  rx = ry = rz = tx = ty = tz = 0;

  for (i=0; i<maxjoys; i++) {
    if (sticks[i].exists == FALSE) 
      continue; // skip processing joysticks that aren't there

    memset(&(sticks[i].info), 0, sizeof(JOYINFOEX));
    sticks[i].info.dwSize = sizeof(JOYINFOEX);
    sticks[i].info.dwFlags = JOY_RETURNALL;

    // query current joystick status
    mres = joyGetPosEx(i, &sticks[i].info);

    if (mres == JOYERR_NOERROR) {
      sticks[i].vx = (int) (10000.0f * ((((float) sticks[i].info.dwXpos - sticks[i].caps.wXmin) / ((float) sticks[i].xrange)) - 0.5f));
      sticks[i].vy = (int) (10000.0f * ((((float) sticks[i].info.dwYpos - sticks[i].caps.wYmin) / ((float) sticks[i].yrange)) - 0.5f));
      sticks[i].vz = (int) (10000.0f * ((((float) sticks[i].info.dwZpos - sticks[i].caps.wZmin) / ((float) sticks[i].zrange)) - 0.5f));
 
      sticks[i].vx = nullregion(800, sticks[i].vx);
      sticks[i].vy = nullregion(800, sticks[i].vy);
      sticks[i].vz = nullregion(800, sticks[i].vz);
      sticks[i].avail = TRUE; // joystick moved
      retval = TRUE; // at least one stick had data
    } else {
      sticks[i].avail = FALSE; // error of some kind, or not there
    }
  }

  // process what stick is actually doing
  for (i=0; i<maxjoys; i++) {
    if (sticks[i].avail != TRUE) 
      continue;  // skip processing that stick 
  
    sticks[i].buttonchanged = sticks[i].info.dwButtons ^ sticks[i].buttons;
   
    // if the user presses button 1, reset the view
    if ((sticks[i].buttonchanged & JOY_BUTTON1) && (sticks[i].info.dwButtons & JOY_BUTTON1)) {
      scaling = 1.0;
      app->scene_resetview();
      msgInfo << "Joystick " << i << " reset view orientation" << sendmsg;
    }

    // Toggle between the different modes
    if ((sticks[i].buttonchanged & JOY_BUTTON2) && (sticks[i].info.dwButtons & JOY_BUTTON2)) {
      switch (sticks[i].moveMode) {
        case ROTATION:
          sticks[i].moveMode = TRANSLATION;
          msgInfo << "Joystick " << i << " set to translation mode" << sendmsg;
          break;

        case TRANSLATION:
          sticks[i].moveMode = SCALING;
          msgInfo << "Joystick " << i << " set to scaling mode" << sendmsg;
          break;

        case SCALING:
          sticks[i].moveMode = OFF;
          msgInfo << "Joystick " << i << " axes disabled" << sendmsg;
          break;

        case OFF:
        default:
          sticks[i].moveMode = ROTATION;
          msgInfo << "Joystick " << i << " set to rotation mode" << sendmsg;
          break;
      }
    }

    if ((sticks[i].buttonchanged & JOY_BUTTON3) && (sticks[i].info.dwButtons & JOY_BUTTON3)) {
      runcommand(new UserKeyEvent(keydev, '3', (int) DisplayDevice::AUX));
    }
    if ((sticks[i].buttonchanged & JOY_BUTTON4) && (sticks[i].info.dwButtons & JOY_BUTTON4)) {
      runcommand(new UserKeyEvent(keydev, '4', (int) DisplayDevice::AUX));
    }
    if ((sticks[i].buttonchanged & JOY_BUTTON5) && (sticks[i].info.dwButtons & JOY_BUTTON5)) {
      runcommand(new UserKeyEvent(keydev, '5', (int) DisplayDevice::AUX));
    }
    if ((sticks[i].buttonchanged & JOY_BUTTON6) && (sticks[i].info.dwButtons & JOY_BUTTON6)) {
      runcommand(new UserKeyEvent(keydev, '6', (int) DisplayDevice::AUX));
    }
    if ((sticks[i].buttonchanged & JOY_BUTTON7) && (sticks[i].info.dwButtons & JOY_BUTTON7)) {
      runcommand(new UserKeyEvent(keydev, '7', (int) DisplayDevice::AUX));
    }
    if ((sticks[i].buttonchanged & JOY_BUTTON8) && (sticks[i].info.dwButtons & JOY_BUTTON8)) {
      runcommand(new UserKeyEvent(keydev, '8', (int) DisplayDevice::AUX));
    }
    if ((sticks[i].buttonchanged & JOY_BUTTON9) && (sticks[i].info.dwButtons & JOY_BUTTON9)) {
      runcommand(new UserKeyEvent(keydev, '9', (int) DisplayDevice::AUX));
    }
 
    switch(sticks[i].moveMode) {
      case ROTATION:
        rx = sticks[i].vy;
        ry = sticks[i].vx;

        if (sticks[i].caps.wCaps & JOYCAPS_HASZ)
          rz = sticks[i].vz;
        else 
          rz = 0;

        app->scene_rotate_by(((float) rx) * rotInc, 'x');
        app->scene_rotate_by(((float) ry) * rotInc, 'y');
        app->scene_rotate_by(((float) rz) * rotInc, 'z');
        break;
  
      case TRANSLATION:
        tx = sticks[i].vx;
        ty = sticks[i].vy;

        if (sticks[i].caps.wCaps & JOYCAPS_HASZ)
          tz = sticks[i].vz;
        else 
          tz = 0;

        app->scene_translate_by(tx * transInc, ty * transInc, -tz * transInc);
        break;

      case SCALING: 
        tx = sticks[i].vx;
        scf = scaling + scaleInc * (float) tx;
        if (scf < 0.0)
          scf = 0.0;
        app->scene_scale_by(scf);
        break;

      case OFF:
      default:
        // do nothing
        // The OFF mode is a safety feature so that VMD's stability 
        // isn't compromised by bad joysticks.  This provides the user
        // with a way to selectively enable joysticks, even though VMD
        // will find and attach them all, only the ones that the user has
        // enabled will actually affect the VMD view.
        break;
    }

    sticks[i].buttons = sticks[i].info.dwButtons;
  }

  return retval;
}



