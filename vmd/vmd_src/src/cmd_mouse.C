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
 *      $RCSfile: cmd_mouse.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.30 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for mouse input control
 ***************************************************************************/

#include <stdlib.h>
#include <tcl.h>
#include "config.h"
#include "VMDApp.h"
#include "Mouse.h"
#include "PickModeList.h"
#include "utilities.h"


// print usage message
static void mouse_usage(Tcl_Interp *interp) {
  Tcl_AppendResult(interp, "mouse usage:\n",
      "mouse callback [on|off]\n",
      "mouse rocking [on|off]\n",
      "mouse mode <mode> <submode>\n",
      "   modes: rotate, translate, scale, light, userpoint\n",
      "   pick, center, query, labelatom\n",
      "   labelbond, labelangle, labeldihedral\n",
      "   moveatom, moveres, movefrag, movemol, moverep\n",
      "   forceatom, forceres, forcefrag, addbond\n", 
      NULL);
}


// The following use MOUSE_MODE
int text_cmd_mouse(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if (argc == 2 && !strupncmp(argv[1], "stoprotation", CMDLEN)) {
    app->scene_stoprotation();
    return TCL_OK;
  } else if(argc < 3 || argc > 4) {
    // if here, something went wrong, so return an error message
    mouse_usage(interp);
    return TCL_ERROR;
  }

  // get the mouse submode, if there is one
  int m2 = (argc > 3 ? atoi(argv[3]) : (-1));
    
  // the modes are:
  //  ROTATION == 0, ... USER == 6 ; as defined in Mouse.h
  if(!strupncmp(argv[1], "mode", CMDLEN)) {
    int m1;
    // see if these are string values
    if (!strupncmp(argv[2], "rotate", CMDLEN))          m1 = Mouse::ROTATION;  
    else if (!strupncmp(argv[2], "translate", CMDLEN))  m1 = Mouse::TRANSLATION;  
    else if (!strupncmp(argv[2], "scale", CMDLEN))      m1 = Mouse::SCALING;  
    else if (!strupncmp(argv[2], "light", CMDLEN))      m1 = Mouse::LIGHT;  
    else if (!strupncmp(argv[2], "userpoint", CMDLEN))  m1 = Mouse::USERPOINT;
    // picking modes...
    else if (!strupncmp(argv[2], "query", CMDLEN))      m1 = Mouse::QUERY;
    else if (!strupncmp(argv[2], "center", CMDLEN))     m1 = Mouse::CENTER;
    else if (!strupncmp(argv[2], "labelatom", CMDLEN))  m1 = Mouse::LABELATOM;
    else if (!strupncmp(argv[2], "labelbond", CMDLEN))  m1 = Mouse::LABELBOND;
    else if (!strupncmp(argv[2], "labelangle", CMDLEN)) m1 = Mouse::LABELANGLE;
    else if (!strupncmp(argv[2], "labeldihedral", CMDLEN)) m1 = Mouse::LABELDIHEDRAL;
    else if (!strupncmp(argv[2], "moveatom", CMDLEN))   m1 = Mouse::MOVEATOM;
    else if (!strupncmp(argv[2], "moveres", CMDLEN))    m1 = Mouse::MOVERES;
    else if (!strupncmp(argv[2], "movefrag", CMDLEN))   m1 = Mouse::MOVEFRAG;
    else if (!strupncmp(argv[2], "movemol", CMDLEN))    m1 = Mouse::MOVEMOL;
    else if (!strupncmp(argv[2], "forceatom", CMDLEN))  m1 = Mouse::FORCEATOM;
    else if (!strupncmp(argv[2], "forceres", CMDLEN))   m1 = Mouse::FORCERES;
    else if (!strupncmp(argv[2], "forcefrag", CMDLEN))  m1 = Mouse::FORCEFRAG;
    else if (!strupncmp(argv[2], "moverep", CMDLEN))    m1 = Mouse::MOVEREP;
    else if (!strupncmp(argv[2], "addbond", CMDLEN))    m1 = Mouse::ADDBOND;
    else if (!strupncmp(argv[2], "pick", CMDLEN)) { 
      if (argc == 3 || m2 == -1)
        m1 = Mouse::PICK;  
      else {
        // if pick is called with a submode, we process it the "old" way
        // This is only for backward-compatibility
        switch (m2) {
          case PickModeList::QUERY:          m1 = Mouse::QUERY; break;
          case PickModeList::CENTER:         m1 = Mouse::CENTER; break;
          case PickModeList::LABELATOM:      m1 = Mouse::LABELATOM; break;
          case PickModeList::LABELBOND:      m1 = Mouse::LABELBOND; break;
          case PickModeList::LABELANGLE:     m1 = Mouse::LABELANGLE; break;
          case PickModeList::LABELDIHEDRAL:  m1 = Mouse::LABELDIHEDRAL; break;
          case PickModeList::MOVEATOM:       m1 = Mouse::MOVEATOM; break;
          case PickModeList::MOVERES:        m1 = Mouse::MOVERES; break;
          case PickModeList::MOVEFRAG:       m1 = Mouse::MOVEFRAG; break;
          case PickModeList::MOVEMOL:        m1 = Mouse::MOVEMOL; break;
          case PickModeList::FORCEATOM:      m1 = Mouse::FORCEATOM; break;
          case PickModeList::FORCERES:       m1 = Mouse::FORCERES; break;
          case PickModeList::FORCEFRAG:      m1 = Mouse::FORCEFRAG; break;
          case PickModeList::MOVEREP:        m1 = Mouse::MOVEREP; break;
          case PickModeList::ADDBOND:        m1 = Mouse::ADDBOND; break;
          case PickModeList::PICK:           m1 = Mouse::PICK; break;
          default:                  
            m1 = Mouse::QUERY; // shouldn't happen...
        }
      }
    } else if (!strupncmp(argv[2], "user", CMDLEN)) { 
      // this is obsolete (left in for backwards-compatibility)
      m1 = Mouse::PICK;  
    }
    else {
      // not a string so convert to a number
      // XXX need to check if atoi fails..
      m1 = atoi(argv[2]);
    }

    if (!app->mouse_set_mode(m1, m2)) {
      Tcl_AppendResult(interp, "Unable to set mouse mode to ",
          argv[2], argc > 3 ? argv[3] : NULL, NULL);

      // if here, something went wrong, so return an error message
      mouse_usage(interp);

      return TCL_ERROR;
    }
    
  } else if(!strupncmp(argv[1], "callback", CMDLEN)) {
    int on=-1;
    if (Tcl_GetBoolean(interp, argv[2], &on) != TCL_OK) return TCL_ERROR;
    if(on!=-1) {
      app->set_mouse_callbacks(on);
    }
  } else if (argc == 3 && !strupncmp(argv[1], "rocking", CMDLEN)) {
    int on = 0;
    if (Tcl_GetBoolean(interp, argv[2], &on) != TCL_OK) return TCL_ERROR;
    app->set_mouse_rocking(on);
  } else {
    // if here, something went wrong, so return an error message
    mouse_usage(interp);
    return TCL_ERROR;
  }
  
  // if here, everything worked out ok
  return TCL_OK;
}

