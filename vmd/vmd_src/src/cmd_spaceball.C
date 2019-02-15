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
 *      $RCSfile: cmd_spaceball.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.10 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Text commands for control of Spaceball/SpaceNavigator/Magellan
 *   and similar 6DOF input devices.
 ***************************************************************************/

#include <stdlib.h>
#include <tcl.h>
#include "config.h"
#include "VMDApp.h"
#include "Spaceball.h"
#include "utilities.h"

// print usage message
static void spaceball_usage(Tcl_Interp *interp) {
  Tcl_AppendResult(interp, "spaceball usage:\n",
      "spaceball mode <mode>\n",
      "   modes: normal, maxaxis, scale, animate, tracker, user\n",
      "spaceball sensitivity <sensitivity>\n",
      "spaceball nullregion <nullregion>\n",
      NULL);
}


int text_cmd_spaceball(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if (argc < 3 || argc > 4) {
    // if here, something went wrong, so return an error message
    spaceball_usage(interp);
    return TCL_ERROR;
  }

  if(!strupncmp(argv[1], "mode", CMDLEN)) {
    int m1 = Spaceball::NORMAL;
    // see if these are string values
    if (!strupncmp(argv[2], "normal", CMDLEN))       m1 = Spaceball::NORMAL;
    else if (!strupncmp(argv[2], "maxaxis", CMDLEN)) m1 = Spaceball::MAXAXIS;
    else if (!strupncmp(argv[2], "scale", CMDLEN))   m1 = Spaceball::SCALING;
    else if (!strupncmp(argv[2], "animate", CMDLEN)) m1 = Spaceball::ANIMATE;
    else if (!strupncmp(argv[2], "tracker", CMDLEN)) m1 = Spaceball::TRACKER;
    else if (!strupncmp(argv[2], "user", CMDLEN))    m1 = Spaceball::USER;

    if (!app->spaceball_set_mode(m1)) {
      Tcl_AppendResult(interp, "Unable to set Spaceball mode to ",
          argv[2], argc > 3 ? argv[3] : NULL, NULL);

      // if here, something went wrong, so return an error message
      spaceball_usage(interp);
      return TCL_ERROR;
    }
  } else if(!strupncmp(argv[1], "sensitivity", CMDLEN)) {
    float s;
    if (sscanf(argv[2], "%f", &s) == 1) {
      if (!app->spaceball_set_sensitivity(s)) {
        // if here, something went wrong, so return an error message
        spaceball_usage(interp);
        return TCL_ERROR;
      }
    } else {
      // if here, something went wrong, so return an error message
      spaceball_usage(interp);
      return TCL_ERROR;
    }
  } else if(!strupncmp(argv[1], "nullregion", CMDLEN)) {
    int nr;
    if (sscanf(argv[2], "%d", &nr) == 1) {
      if (!app->spaceball_set_null_region(nr)) {
        // if here, something went wrong, so return an error message
        spaceball_usage(interp);
        return TCL_ERROR;
      }
    } else {
      // if here, something went wrong, so return an error message
      spaceball_usage(interp);
      return TCL_ERROR;
    }
  } else {
    // if here, something went wrong, so return an error message
    spaceball_usage(interp);
    return TCL_ERROR;
  }
  
  // if here, everything worked out ok
  return TCL_OK;
}

