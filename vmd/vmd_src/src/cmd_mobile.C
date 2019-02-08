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
 *      $RCSfile: cmd_mobile.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.10 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Text commands for control of Mobile/SpaceNavigator/Magellan
 *   and similar 6DOF input devices.
 ***************************************************************************/

#include <stdlib.h>
#include <tcl.h>
#include "config.h"
#include "VMDApp.h"
#include "MobileInterface.h"
#include "utilities.h"

// print usage message
static void mobile_usage(Tcl_Interp *interp) {
  Tcl_AppendResult(interp, "mobile usage:\n",
      "mobile mode <mode>\n",
      "   modes: off, move, animate, tracker, user\n",
      "mobile port <incoming network port number>\n",
      "mobile get <mode/port/clientList/APIsupported>\n",
      "mobile set activeClient {name} {ip}\n",
      "mobile sendMsg {name} {ip} msgType {msg}\n",
//      "mobile sensitivity <sensitivity>\n",
//      "mobile nullregion <nullregion>\n",
      NULL);
}

int text_cmd_mobile(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if (argc < 3 || argc > 6) {
    // if here, something went wrong, so return an error message
    mobile_usage(interp);
    return TCL_ERROR;
  }

  if (!strupncmp(argv[1], "mode", CMDLEN)) {
    int m1 = Mobile::OFF;
    // see if these are string values
    if (!strupncmp(argv[2], "off", CMDLEN))          m1 = Mobile::OFF;
    else if (!strupncmp(argv[2], "move", CMDLEN))    m1 = Mobile::MOVE;
    else if (!strupncmp(argv[2], "animate", CMDLEN)) m1 = Mobile::ANIMATE;
    else if (!strupncmp(argv[2], "tracker", CMDLEN)) m1 = Mobile::TRACKER;
    else if (!strupncmp(argv[2], "user", CMDLEN))    m1 = Mobile::USER;
    else mobile_usage(interp); // error 

    if (!app->mobile_set_mode(m1)) {
      Tcl_AppendResult(interp, "Unable to set Mobile mode to ",
          argv[2], argc > 3 ? argv[3] : NULL, NULL);

      // if here, something went wrong, so return an error message
      mobile_usage(interp);
      return TCL_ERROR;
    }
  } else if(!strupncmp(argv[1], "port", CMDLEN)) {
    int port;
    if (sscanf(argv[2], "%d", &port) == 1) {
      if (!app->mobile_network_port(port)) {
        // if here, something went wrong, so return an error message
        mobile_usage(interp);
        return TCL_ERROR;
      }
    } else {
      // if here, something went wrong, so return an error message
      mobile_usage(interp);
      return TCL_ERROR;
    }
  } else if(!strupncmp(argv[1], "get", CMDLEN)) {
    if (!strupncmp(argv[2], "mode", CMDLEN))          
    {
       Tcl_AppendResult(interp, Mobile::get_mode_str((Mobile::MoveMode)app->mobile_get_mode()), NULL);
    } else if (!strupncmp(argv[2], "clientList", CMDLEN)) {
       ResizeArray <JString *>* ip;
       ResizeArray <JString *>* nick;
       ResizeArray <bool>* active;
       app->mobile_get_client_list( nick, ip, active);
       for (int i=0; i<nick->num(); i++)
       {
          Tcl_AppendResult(interp, " {", NULL);

          Tcl_AppendResult(interp, " {", NULL);
          // here's what we've got..
          //   (const char *)(*(*nick)[i])
          //      now to explain it....
          //   (const char *)               Tcl_AppendResult needs a char*
          //                 (*          )  We have a JString* in ResizeArray that we want to be a JString   
          //                   (*nick)      we have a ResizeArray ptr that we want to be a ResizeArray
          //                          [i]   specific array elem
          Tcl_AppendResult(interp, (const char *)(*(*nick)[i]), NULL);
          Tcl_AppendResult(interp, "}", NULL);

          Tcl_AppendResult(interp, " {", NULL);
          Tcl_AppendResult(interp, (const char *)(*(*ip)[i]), NULL);
          Tcl_AppendResult(interp, "}", NULL);

          Tcl_AppendResult(interp, " {", NULL);
          char tmp[10]; sprintf(tmp, "%d", (*active)[i]);
          Tcl_AppendResult(interp, tmp, NULL);
          Tcl_AppendResult(interp, "}", NULL);

          Tcl_AppendResult(interp, " }", NULL);
       }

    } else if (!strupncmp(argv[2], "port", CMDLEN)) {
       char tmpstr[20];
       sprintf(tmpstr, "%d", app->mobile_get_network_port());
       Tcl_AppendResult(interp, tmpstr, NULL);
    } else if (!strupncmp(argv[2], "APIsupported", CMDLEN)) {
       char tmpstr[20];
       sprintf(tmpstr, "%d", app->mobile_get_APIsupported());
       Tcl_AppendResult(interp, tmpstr, NULL);
    } else mobile_usage(interp); // error 

  } else if(!strupncmp(argv[1], "sendMsg", CMDLEN)) {
    if (argc >= 6)
    {
       if (!app->mobile_sendMsg(argv[2], argv[3], argv[4], argv[5])) {

            // if here, something went wrong, so return an error message
         mobile_usage(interp);
         return TCL_ERROR;
       }
    } else mobile_usage(interp); // error 

  } else if(!strupncmp(argv[1], "set", CMDLEN)) {
    if (!strupncmp(argv[2], "activeClient", CMDLEN))          
    {
       if (argc >= 5)
       {
          if (!app->mobile_set_activeClient(argv[3], argv[4])) {
//         Tcl_AppendResult(interp, "Unable to set activeClient to ",
//             argv[2], argc > 3 ? argv[3] : NULL, NULL);

            // if here, something went wrong, so return an error message
            mobile_usage(interp);
            return TCL_ERROR;
          }
       } else mobile_usage(interp); // error 

    } else mobile_usage(interp); // error 

  } else {
    // if here, something went wrong, so return an error message
    mobile_usage(interp);
    return TCL_ERROR;
  }
  
  // if here, everything worked out ok
  return TCL_OK;
}

