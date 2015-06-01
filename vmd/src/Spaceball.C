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
 *	$RCSfile: Spaceball.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.61 $	$Date: 2011/02/18 20:28:32 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Spaceball UI object, which maintains the current state of the 
 * spaceball.  This code uses John Stone's spaceball library when
 * VMDLIBSBALL is defined, or the Spaceware library when VMDSPACEWARE 
 * is defined.
 *
 ***************************************************************************
 * TODO list:
 *   1) Mac code needs to either flush queued events, force-read them all
 *      at every screen redraw, or use an independent thread to process them
 *      so that the event queue never lags the screen draws.
 *   2) The 'animate' mode needs to use the wall-clock time since the last
 *      event to normalize the applied force to account for complex 
 *      reps slowing down the update rate.  With the exponential speed
 *      the animate step size can get out of hand far too easily.  The max
 *      step size allowed should be capped, preferably to a user-configurable
 *      limit.  A default step size limit of something like 50 would be a 
 *      good compromise. (Partially done)
 *   3) Orthographic mode confuses beginners because the scene doesn't
 *      scale when they pull objects towards the camera.  They expect 
 *      behavior more like what one gets with perspective.  We may want to
 *      provide a default behavior of scaling when orthographic projection
 *      is active and the user pulls towards them. (Partially done)
 *   4) The spaceball mode switching is too complex to be handled well
 *      by a two button device.  We need to map these functions to 
 *      keyboard keys to make it easier to use.  It would be even better if
 *      we briefly displayed status messages on the screen for a couple of
 *      seconds when modes are changed since one might not be able to see
 *      the text console.  This would be useful for any mode changes that
 *      affect the VMD control/input state that don't have some other
 *      visual/audio cue to observe. (e.g. the mouse pointer changing
 *      when mouse modes are altered).
 *
 ***************************************************************************/

#include "Spaceball.h"
#include "DisplayDevice.h"
#include "TextEvent.h"
#include "CommandQueue.h"
#include "Inform.h"
#include "PickList.h"
#include "Animation.h"
#include "VMDApp.h"
#include "math.h"
#include "stdlib.h" // for getenv(), abs() etc.

//
// 3DConnexion MacOS X driver API
//
#if defined(VMDTDCONNEXION) && defined(__APPLE__)
#include <unistd.h>
#include <Carbon/Carbon.h>
#include <stdio.h>
#include <stdlib.h>

// 3dxware API header
#include "3DconnexionClient/ConnexionClientAPI.h"

extern "C" {
extern OSErr InstallConnexionHandlers(ConnexionMessageHandlerProc messageHandler, ConnexionAddedHandlerProc addedHandler, ConnexionRemovedHandlerProc removedHandler) __attribute__((weak_import));
}

// Pascal string name of the application binary used for window 
// focus handling when we don't provide an application bundle key
// at callback registration time
#if defined(ARCH_MACOSX)
static UInt8 *executablename = (UInt8 *) "\pvmd_MACOSX";
#elif defined(ARCH_MACOSXX86)
static UInt8 *executablename = (UInt8 *) "\pvmd_MACOSXX86";
#elif defined(ARCH_MACOSXX86_64)
static UInt8 *executablename = (UInt8 *) "\pvmd_MACOSXX86_64";
#else
#error
#endif

typedef struct {
  int enabled;   // flag indicating whether we're live or not
  UInt16 client; // 3DConnexion API client used for focus handling
  int tx;
  int ty;
  int tz;
  int rx; 
  int ry;
  int rz;
  int buttons;
  int eventcount;
} tdx_data;

// global event data structure
static tdx_data tdxevent;

// event handler proc
// XXX note that the default single-threaded implementation
//     described in the 3DConnexion documentation leaves a lot
//     to be desired.  In reality, we realistically want a multithreaded
//     implementation, otherwise the events queue up and get way behind
//     when we draw a complex molecule.  In order to handle the device
//     events separately, we can install a special event handler in 
//     a second thread and have it store event data to a shared memory area.
static void tdx_msghandler(io_connect_t connection, 
                           natural_t msgtype, void *msgarg) {
  ConnexionDeviceState *state = NULL;
  switch (msgtype) {
    case kConnexionMsgDeviceState:
      state = (ConnexionDeviceState *) msgarg;
      if (state->client == tdxevent.client) {
        switch (state->command) {
          case kConnexionCmdHandleAxis:
            tdxevent.tx +=  state->axis[0];
            tdxevent.ty += -state->axis[2];
            tdxevent.tz += -state->axis[1];
            tdxevent.rx +=  state->axis[3];
            tdxevent.ry += -state->axis[5];
            tdxevent.rz += -state->axis[4];
            tdxevent.eventcount++;
            break;

          case kConnexionCmdHandleButtons:
            tdxevent.buttons = state->buttons;
            tdxevent.eventcount++;
            break;
        }
      }
      break;

    default:
//      printf("Unknown message type\n");
      break;
  }
}

static void tdx_clear() {
  memset(&tdxevent, 0, sizeof(tdxevent));
}

static int tdx_enable() {
  UInt16 clientID;

  if (InstallConnexionHandlers == NULL) {
    msgInfo << "No 3DConnexion driver on this system." << sendmsg;
    return -1;
  }
  OSErr result = InstallConnexionHandlers(tdx_msghandler, 0L, 0L);
  if (result != noErr) {
    msgInfo << "Unable to register with 3DConnexion driver." << sendmsg;
    return -1;
  }

#if 1
  // only respond to all events when we have focus
  clientID = RegisterConnexionClient(0, executablename,
               kConnexionClientModeTakeOver, kConnexionMaskAll);
#else
  // respond to all events whether we have focus or not
  clientID = RegisterConnexionClient(kConnexionClientWildcard, NULL,
               kConnexionClientModeTakeOver, kConnexionMaskAll);
#endif

  tdxevent.enabled = 1;
  tdxevent.client = clientID;

  return 0;
}

static int tdx_detach() {
  if (tdxevent.enabled) {
    UnregisterConnexionClient(tdxevent.client);
    CleanupConnexionHandlers();
  }
  tdx_clear();
}

int tdx_getstatus(int &tx, int &ty, int &tz, int &rx, int &ry, int &rz, int &buttons) {
  int eventcount = tdxevent.eventcount;

  tx = tdxevent.tx; 
  ty = tdxevent.ty; 
  tz = tdxevent.tz; 
  rx = tdxevent.rx; 
  ry = tdxevent.ry; 
  rz = tdxevent.rz; 
  buttons = tdxevent.buttons;
  
  tdxevent.tx = 0;
  tdxevent.ty = 0;
  tdxevent.tz = 0;
  tdxevent.rx = 0;
  tdxevent.ry = 0;
  tdxevent.rz = 0;
  tdxevent.eventcount = 0;
  
  return eventcount;
}

#endif



// constructor
Spaceball::Spaceball(VMDApp *vmdapp)
	: UIObject(vmdapp) {

#if defined(VMDTDCONNEXION) && defined(__APPLE__)
  // Enable input from MacOS X 3DConnexion API
  tdx_clear();
  if (tdx_enable() == 0)
    msgInfo << "3DConnexion SpaceNavigator enabled." << sendmsg;
#endif

#if defined(VMDLIBSBALL) && !defined(VMDSPACEWARE)
  sball=NULL; // zero it out to begin with
  if (getenv("VMDSPACEBALLPORT") != NULL) {
    msgInfo << "Opening Spaceball (direct I/O) on port: " 
            << getenv("VMDSPACEBALLPORT") << sendmsg;
    sball = sball_open(getenv("VMDSPACEBALLPORT"));
    if (sball == NULL) 
      msgErr << "Failed to open Spaceball direct I/O serial port, device disabled." 
             << sendmsg; 
  }
#endif

  buttonDown = 0;

  reset();
}


// destructor
Spaceball::~Spaceball(void) {
#if defined(VMDTDCONNEXION) && defined(__APPLE__)
  // Disable input from MacOS X 3DConnextion API
  tdx_detach();
#endif

#if defined(VMDLIBSBALL) && !defined(VMDSPACEWARE)
  if (sball != NULL)
    sball_close(sball);
#endif
}


/////////////////////// virtual routines for UI init/display  /////////////
   
// reset the spaceball to original settings
void Spaceball::reset(void) {
  // set the default motion mode and initialize button state
  move_mode(NORMAL);

  // set global spaceball sensitivity within VMD
  // (this has no effect on underlying driver sensitivity settings)
  set_sensitivity(1.0f);

  // set the null region to a small value initially
  set_null_region(16);

  // set the maximum animate stride allowed to 20 by default
  set_max_stride(20);

  // set the default translation and rotation increments
  // these really need to be made user modifiable at runtime
  transInc = 1.0f / 25000.0f;
    rotInc = 1.0f /   200.0f;
  scaleInc = 1.0f / 25000.0f;
   animInc = 1.0f /    75.0f;
}

// update the display due to a command being executed.  Return whether
// any action was taken on this command.
// Arguments are the command type, command object, and the 
// success of the command (T or F).
int Spaceball::act_on_command(int type, Command *cmd) {
  return FALSE; // we don't take any commands presently
}


// check for an event, and queue it if found.  Return TRUE if an event
// was generated.
int Spaceball::check_event(void) {
  int tx, ty, tz, rx, ry, rz, buttons;
  int buttonchanged;
  int win_event=FALSE;
  int direct_event=FALSE;
  // for use in UserKeyEvent() calls
  DisplayDevice::EventCodes keydev=DisplayDevice::WIN_KBD;

  // explicitly initialize event state variables
  rx=ry=rz=tx=ty=tz=buttons=0;

#if defined(VMDTDCONNEXION) && defined(__APPLE__)
  if (tdx_getstatus(tx, ty, tz, rx, ry, rz, buttons))
    win_event = TRUE;
#else
  if (app->display->spaceball(&rx, &ry, &rz, &tx, &ty, &tz, &buttons)) 
    win_event = TRUE;
#endif


#if defined(VMDLIBSBALL)
  // combine direct spaceball events together with window-system events
  if (sball != NULL) {
    int rx2, ry2, rz2, tx2, ty2, tz2, buttons2;
    if (sball_getstatus(sball, &tx2, &ty2, &tz2, &rx2, &ry2, &rz2, &buttons2)) {
      direct_event = TRUE;
      rx += rx2;
      ry += ry2;
      rz += rz2;
      tx += tx2; 
      ty += ty2; 
      tz += tz2; 
      buttons |= buttons2;
    }
  }
#endif

  if (!win_event && !direct_event)
    return FALSE; // no events to report

  // find which buttons changed state
  buttonchanged = buttons ^ buttonDown; 

  // if the user presses button 1, reset the view, a very very very
  // important feature to have implemented early on... ;-)
#if defined(VMDLIBSBALL)  && !defined(VMDSPACEWARE)
  if (((buttonchanged & SBALL_BUTTON_1) && (buttons & SBALL_BUTTON_1)) ||
      ((buttonchanged & SBALL_BUTTON_LEFT) && (buttons & SBALL_BUTTON_LEFT))){
#else 
// #elif!defined(VMDLIBSBALL) && defined(VMDSPACEWARE)
  if ((buttonchanged & 2) && (buttons & 2)) {
#endif

    app->scene_resetview();
    msgInfo << "Spaceball reset view orientation" << sendmsg;
  }

  // Toggle between the different modes
#if   defined(VMDLIBSBALL)  &&  !defined(VMDSPACEWARE)
  if (((buttonchanged & SBALL_BUTTON_2) && (buttons & SBALL_BUTTON_2)) ||
      ((buttonchanged & SBALL_BUTTON_RIGHT) && (buttons & SBALL_BUTTON_RIGHT))) {
#else
//#elif !defined(VMDLIBSBALL) &&  defined(VMDSPACEWARE)
  if ((buttonchanged & 4) && (buttons & 4)) {
#endif

    switch (moveMode) {
      case NORMAL:
        move_mode(MAXAXIS);
        msgInfo << "Spaceball set to dominant axis rotation/translation mode" << sendmsg;
        break;   

      case MAXAXIS:
        move_mode(SCALING);
        msgInfo << "Spaceball set to scaling mode" << sendmsg;
        break;   

      case SCALING:
        move_mode(ANIMATE);
        msgInfo << "Spaceball set to animate mode" << sendmsg;
        break;   

      case ANIMATE:
        move_mode(TRACKER);
        msgInfo << "Spaceball set to tracker mode" << sendmsg;
        break;   

      case TRACKER:
        move_mode(USER);
        msgInfo << "Spaceball set to user mode" << sendmsg;
        break;   

      default: 
        move_mode(NORMAL);
        msgInfo << "Spaceball set to rotation/translation mode" << sendmsg;
        break;
    }
  }

  // if the user presses button 3 through N, run a User command
#if defined(VMDLIBSBALL)  &&  !defined(VMDSPACEWARE)
  if ((buttonchanged & SBALL_BUTTON_3) && (buttons & SBALL_BUTTON_3)) {
    runcommand(new UserKeyEvent(keydev, '3', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & SBALL_BUTTON_4) && (buttons & SBALL_BUTTON_4)) {
    runcommand(new UserKeyEvent(keydev, '4', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & SBALL_BUTTON_5) && (buttons & SBALL_BUTTON_5)) {
    runcommand(new UserKeyEvent(keydev, '5', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & SBALL_BUTTON_6) && (buttons & SBALL_BUTTON_6)) {
    runcommand(new UserKeyEvent(keydev, '6', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & SBALL_BUTTON_7) && (buttons & SBALL_BUTTON_7)) {
    runcommand(new UserKeyEvent(keydev, '7', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & SBALL_BUTTON_8) && (buttons & SBALL_BUTTON_8)) {
    runcommand(new UserKeyEvent(keydev, '8', (int) DisplayDevice::AUX));
  }
//#elif !defined(VMDLIBSBALL) &&  defined(VMDSPACEWARE)
#else
  if ((buttonchanged & 8) && (buttons & 8)) {
    runcommand(new UserKeyEvent(keydev, '3', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & 16) && (buttons & 16)) {
    runcommand(new UserKeyEvent(keydev, '4', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & 32) && (buttons & 32)) {
    runcommand(new UserKeyEvent(keydev, '5', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & 64) && (buttons & 64)) {
    runcommand(new UserKeyEvent(keydev, '6', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & 128) && (buttons & 128)) {
    runcommand(new UserKeyEvent(keydev, '7', (int) DisplayDevice::AUX));
  }
  if ((buttonchanged & 256) && (buttons & 256)) {
    runcommand(new UserKeyEvent(keydev, '8', (int) DisplayDevice::AUX));
  }
#endif

  // get absolute values of axis forces for use in 
  // null region processing and min/max comparison tests
  int atx, aty, atz, arx, ary, arz;
  atx = abs(tx);
  aty = abs(ty);
  atz = abs(tz);
  arx = abs(rx);
  ary = abs(ry);
  arz = abs(rz);


  // perform null region processing
  if (atx > null_region) {
    tx = ((tx > 0) ? (tx - null_region) : (tx + null_region));
  } else {
    tx = 0;
  }
  if (aty > null_region) {
    ty = ((ty > 0) ? (ty - null_region) : (ty + null_region));
  } else {
    ty = 0;
  }
  if (atz > null_region) {
    tz = ((tz > 0) ? (tz - null_region) : (tz + null_region));
  } else {
    tz = 0;
  }
  if (arx > null_region) {
    rx = ((rx > 0) ? (rx - null_region) : (rx + null_region));
  } else {
    rx = 0;
  }
  if (ary > null_region) {
    ry = ((ry > 0) ? (ry - null_region) : (ry + null_region));
  } else {
    ry = 0;
  }
  if (arz > null_region) {
    rz = ((rz > 0) ? (rz - null_region) : (rz + null_region));
  } else {
    rz = 0;
  }


  // Ignore null motion events since some versions of the Windows 
  // Spaceball driver emit a constant stream of null motion event
  // packets which would otherwise cause continuous redraws, pegging the 
  // CPU and GPU at maximum load.
  if ((arx+ary+arz+atx+aty+atz) > 0) {
    float ftx = tx * sensitivity;
    float fty = ty * sensitivity;
    float ftz = tz * sensitivity;
    float frx = rx * sensitivity;
    float fry = ry * sensitivity;
    float frz = rz * sensitivity;
    char rmaxaxis = 'x';
    float rmaxval = 0.0f;
    float tmaxval = 0.0f;
    float tmaxvec[3] = { 0.0f, 0.0f, 0.0f };
    tmaxvec[0] = tmaxvec[1] = tmaxvec[2] = 0.0f;

    switch(moveMode) {
      case NORMAL:
        // Z-axis rotation/trans have to be negated in order to please VMD...
        app->scene_rotate_by(frx * rotInc, 'x');
        app->scene_rotate_by(fry * rotInc, 'y');
        app->scene_rotate_by(-frz * rotInc, 'z');
        if (app->display_projection_is_perspective()) {
          app->scene_translate_by(ftx * transInc, fty * transInc, -ftz * transInc);
        } else {
          app->scene_scale_by((1.0f + scaleInc * -ftz > 0.0f) ? 
                               1.0f + scaleInc * -ftz : 0.0f);
          app->scene_translate_by(ftx * transInc, fty * transInc, 0);
        }
 
        break;

      case MAXAXIS:
        // Z-axis rotation/trans have to be negated in order to please VMD...
        // find dominant rotation axis
        if (arx > ary) {
          if (arx > arz) {
            rmaxaxis = 'x';
            rmaxval = frx; 
          } else {
            rmaxaxis = 'z';
            rmaxval = -frz; 
          }
        } else {     
          if (ary > arz) {
            rmaxaxis = 'y';
            rmaxval = fry; 
          } else {
            rmaxaxis = 'z';
            rmaxval = -frz; 
          }
        }

        // find dominant translation axis
        if (atx > aty) {
          if (atx > atz) {
            tmaxval = ftx;
            tmaxvec[0] = ftx; 
          } else {
            tmaxval = ftz;
            tmaxvec[2] = ftz; 
          }
        } else {     
          if (aty > atz) {
            tmaxval = fty;
            tmaxvec[1] = fty; 
          } else {
            tmaxval = ftz;
            tmaxvec[2] = ftz; 
          }
       }

       // determine whether to rotate or translate
       if (fabs(rmaxval) > fabs(tmaxval)) {
         app->scene_rotate_by(rmaxval * rotInc, rmaxaxis);
       } else {
         app->scene_translate_by(tmaxvec[0] * transInc, 
                                 tmaxvec[1] * transInc, 
                                -tmaxvec[2] * transInc);
       }
       break;

      case SCALING:
        app->scene_scale_by((1.0f + scaleInc * ftz > 0.0f) ? 
                             1.0f + scaleInc * ftz : 0.0f);
        break;

      case ANIMATE:
        // if we got a non-zero input, update the VMD animation state
        if (abs(ry) > 0) {
#if 1
          // exponential input scaling
          float speed = fabsf(expf(fabsf((fabsf(fry) * animInc) / 1.7f))) - 1.0f;
#else
          // linear input scaling
          float speed = fabsf(fry) * animInc;
#endif

          if (speed > 0) {
            if (speed < 1.0) 
              app->animation_set_speed(speed);
            else
              app->animation_set_speed(1.0f);
 
            int stride = 1;
            if (fabs(speed - 1.0) > (double) maxstride)
              stride = maxstride;
            else
              stride = 1 + (int) fabs(speed-1.0);
            if (stride < 1)
              stride = 1; 
            app->animation_set_stride(stride);
 
            // -ry is turned to the right, like a typical shuttle/jog control
            if (fry < 0) 
              app->animation_set_dir(Animation::ANIM_FORWARD1);
            else
              app->animation_set_dir(Animation::ANIM_REVERSE1);
          } else {
            app->animation_set_dir(Animation::ANIM_PAUSE);
            app->animation_set_speed(1.0f);
          }
        } else {
          app->animation_set_dir(Animation::ANIM_PAUSE);
          app->animation_set_speed(1.0f);
        }
        break;

      case TRACKER:
        trtx = ftx;
        trty = fty;
        trtz = ftz;
        trrx = frx;
        trry = fry;
        trrz = frz;
        trbuttons = buttons;
        break;

      case USER:
        // inform TCL
        app->commandQueue->runcommand(new SpaceballEvent(ftx, fty, ftz, 
                                                         frx, fry, frz, 
                                                         buttons));
        break;
    }
  }

  // update button status for next time through
  buttonDown = buttons;

  return TRUE;
}


///////////// public routines for use by text commands etc

const char* Spaceball::get_mode_str(MoveMode mm) {
  const char* modestr;

  switch (mm) {
    default:
    case NORMAL:      modestr = "rotate";     break;
    case MAXAXIS:     modestr = "maxaxis";    break;
    case SCALING:     modestr = "scale";      break;
    case ANIMATE:     modestr = "animate";    break;
    case TRACKER:     modestr = "tracker";    break;
    case USER:        modestr = "user";       break;
  }

  return modestr;
}


void Spaceball::get_tracker_status(float &tx, float &ty, float &tz,
                                   float &rx, float &ry, float &rz, 
                                   int &buttons) {
  tx =  trtx * transInc;
  ty =  trty * transInc;
  tz = -trtz * transInc;
  rx =  trrx * rotInc;
  ry =  trry * rotInc;
  rz = -trrz * rotInc;
  buttons = trbuttons;
}


// set the Spaceball move mode to the given state; return success
int Spaceball::move_mode(MoveMode mm) {
  // change the mode now
  moveMode = mm;

  /// clear out any remaining tracker event data if we're not in that mode
  if (moveMode != TRACKER) {
    trtx=trty=trtz=trrx=trry=trrz=0.0f; 
    trbuttons=0;
  }

  return TRUE; // report success
}



