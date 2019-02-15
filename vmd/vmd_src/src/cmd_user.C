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
 *	$RCSfile: cmd_user.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.28 $	$Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Commands to allow the user to customize the hotkeys and pop-up menus.
 ***************************************************************************/

#include <string.h>
#include <ctype.h>  // for toupper/tolower
#include <tcl.h>
#include "config.h"
#include "Mouse.h"
#include "utilities.h"
#include "DisplayDevice.h"
#include "JString.h"
#include "VMDApp.h"

static int check_canonical_form(const char *s) {
  // Handle single-word key names
  if (!strcmp(s, "Escape")) return TRUE;
  if (!strcmp(s, "Up")) return TRUE;
  if (!strcmp(s, "Down")) return TRUE;
  if (!strcmp(s, "Left")) return TRUE;
  if (!strcmp(s, "Right")) return TRUE;
  if (!strcmp(s, "Page_Up")) return TRUE;
  if (!strcmp(s, "Page_Down")) return TRUE;
  if (!strcmp(s, "Home")) return TRUE;
  if (!strcmp(s, "End")) return TRUE;
  if (!strcmp(s, "Insert")) return TRUE;
  if (!strcmp(s, "Delete")) return TRUE;

  // Handle key code sequences involving combinations
  if (s[0] == 'A') {
    // could be a letter, 'Alt-', or 'Aux-'
    s++;
    if (*s == 0) {
      return TRUE;
    }

    // must be an Alt- or Aux-
    if (*s != 'l' && *s != 'u') {
      return FALSE;
    }

    if (*s == 'l') {
      s++;
      if (*s++ != 't') return FALSE;
      if (*s++ != '-') return FALSE;
    } else if (*s == 'u') {
      s++;
      if (*s++ != 'x') return FALSE;
      if (*s++ != '-') return FALSE;
    }
  }

  if (s[0] == 'C') {
    // could be a letter, or 'Control-'
    s++;
    if (*s == 0) {
      return TRUE;
    }
    // must be a Control
    if (*s++ != 'o') return FALSE;
    if (*s++ != 'n') return FALSE;
    if (*s++ != 't') return FALSE;
    if (*s++ != 'r') return FALSE;
    if (*s++ != 'o') return FALSE;
    if (*s++ != 'l') return FALSE;
    if (*s++ != '-') return FALSE;
  }

  if (s[0] == 'F') {
    // could be a letter, or 'Fn' (F1-F12)
    s++;
    if (*s == 0) {
      return TRUE;
    }
    // must be an Fn
    if ((*s) >= '0' && (*s) <= '9') {
      s++;
      if (*s == 0) {
        return TRUE;
      }
      if (*s >= '0' && *s <= '2') {
        s++;
        if (*s == 0) {
          return TRUE;
        }
      }
      return FALSE;
    }
  }

  // must be a single character
  if (s[0] == 0) return FALSE; // NULL, was invalid

  if (s[1] == 0) return TRUE; // was a valid single character

  return FALSE; // was otherwise invalid
}

int text_cmd_user(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if(argc < 3) {
    Tcl_SetResult(interp, 
      (char *)
      "user list keys\n"
      "user print keys\n"
      "user add key <character> <command>",
      TCL_STATIC);
    return TCL_ERROR;
  }

  if(!strupncmp(argv[1], "add", CMDLEN)) {
    if (!strupncmp(argv[2], "key", CMDLEN)) {
      if(argc < 5)
        return TCL_ERROR;
      // check this is a valid string
      if (check_canonical_form(argv[3])) {
        const char *combstr = argv[4];
        const char *desc = NULL;
        if (argc > 5) desc = argv[5];
        int indx = app->userKeys.typecode(argv[3]);
        if (indx < 0) {
          app->userKeys.add_name(argv[3], stringdup(combstr));
        } else {
          delete [] app->userKeys.data(indx);
          app->userKeys.set_data(indx, stringdup(combstr));
        }
        if (desc) {
          indx = app->userKeyDesc.typecode(argv[3]);
          if (indx < 0) {
            app->userKeyDesc.add_name(argv[3], stringdup(desc));
          } else {
            delete [] app->userKeyDesc.data(indx);
            app->userKeys.set_data(indx, stringdup(desc));
          }
        }
      } else {
	Tcl_AppendResult(interp, "user key ", argv[3], " is not valid",
			 NULL);
	return TCL_ERROR;
      }
    }  else
      return TCL_ERROR;

  } else if(!strupncmp(argv[1], "list", CMDLEN)) {
    // return definitions of current items
    if (argc != 3) {
      return TCL_ERROR;
    }
    if (!strcmp(argv[2], "keys")) {
      int num = app->userKeys.num();
      for (int i=0; i<num; i++) {
	// return tuples of {keystroke command description}
	Tcl_AppendResult(interp, i==0?"":" ", "{", NULL);
	Tcl_AppendElement(interp,  app->userKeys.name(i));
	Tcl_AppendElement(interp, 
           (const char *) app->userKeys.data(i));
        int desc_typecode = app->userKeyDesc.typecode(app->userKeys.name(i));
        if (desc_typecode >= 0) {
          Tcl_AppendElement(interp,
             (const char *) app->userKeyDesc.data(i));
        } else {
          Tcl_AppendElement(interp, "");
        } 
	Tcl_AppendResult(interp, "}", NULL);
      }
      return TCL_OK;
    } else {
      return TCL_ERROR;
    }
    // will never get here

  } else if(!strupncmp(argv[1], "print", CMDLEN)) {
    // print out definitions of current items
    Tcl_AppendResult(interp, 
        "Keyboard shortcuts:\n",
        "-------------------\n", NULL);
    for (int i=0; i<app->userKeys.num(); i++) {
      const char *key = app->userKeys.name(i);
      Tcl_AppendResult(interp, "'", key, "' : ", app->userKeys.data(i), "\n",
          NULL);
      if (app->userKeyDesc.typecode(key) >= 0) {
        Tcl_AppendResult(interp, "     Description: ", 
            app->userKeyDesc.data(key), NULL);
      }
    }
  } else
    return TCL_ERROR;
    
  // if here, everything worked out ok
  return TCL_OK;
}

