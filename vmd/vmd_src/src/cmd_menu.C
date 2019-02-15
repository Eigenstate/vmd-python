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
 *      $RCSfile: cmd_menu.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.38 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for window/menu control
 ***************************************************************************/

#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <tcl.h>
#include "config.h"
#include "utilities.h"
#include "VMDApp.h"

#ifdef VMDTK
#include "VMDTkMenu.h"
#endif

#define TCL_HELP 99 //internal code to generate error AND display menu command help

int text_cmd_menu(ClientData cd, Tcl_Interp *interp, int argc,
                     const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;
  int retval = TCL_OK;

  // make sure a menu was named
  if (argc < 2) retval = TCL_HELP;

  if (argc == 2 && !strupncmp(argv[1], "list", CMDLEN)) {
      // return a list of the available menus
      for (int i=0; i<app->num_menus(); i++) 
        Tcl_AppendElement(interp,  app->menu_name(i));
  }
#ifdef VMDTK
  else if (argc > 1 && !strupncmp(argv[1], "tk", CMDLEN)) {
#ifndef MACVMD
    if ((argc == 5 || argc == 6 ) && !strupncmp(argv[2], "add", CMDLEN)) {
      VMDMenu *menu = new VMDTkMenu(argv[3], argv[4], app, interp);
      if (!app->add_menu(menu)) {
        delete menu;
        char buf[50];
        sprintf(buf, "Unable to add menu %s.\n", argv[3]);
        Tcl_SetResult(interp, buf, TCL_VOLATILE);
        retval = TCL_ERROR;
      } else {
        // tell VMD that this is a menu extension
        if (argc==6) app->menu_add_extension(argv[3],argv[5]);
        else app->menu_add_extension(argv[3],argv[3]);
  
        // tell Tcl that a new menu extension has been added
        Tcl_SetVar(interp, "vmd_menu_extension", argv[3], TCL_GLOBAL_ONLY);
      }
    } else if ((argc == 5 || argc == 6 ) && !strupncmp(argv[2], "register", CMDLEN)) {
      VMDTkMenu *menu = new VMDTkMenu(argv[3], NULL, app, interp);
      menu->register_proc(argv[4]);
      if (!app->add_menu(menu)) {
        delete menu;
        char buf[50];
        sprintf(buf, "Unable to add menu %s\n", argv[3]);
        Tcl_SetResult(interp, buf, TCL_VOLATILE);
        retval = TCL_ERROR;
      } else {
        // tell VMD that this is a menu extension
        if (argc==6) app->menu_add_extension(argv[3],argv[5]);
        else app->menu_add_extension(argv[3],argv[3]);
  
        // tell Tcl that a new menu extension has been added
        Tcl_SetVar(interp, "vmd_menu_extension", argv[3], TCL_GLOBAL_ONLY);
      }
    } else if (argc == 4 && !strupncmp(argv[2], "remove", CMDLEN)) {
      if (!app->remove_menu(argv[3])) {
        char buf[50];
        sprintf(buf, "Unable to remove menu %s\n", argv[3]);
        Tcl_SetResult(interp, buf, TCL_VOLATILE);
        retval = TCL_ERROR;
      }
      else app->menu_remove_extension(argv[3]);
    }
    else retval = TCL_HELP;
#else
    /* MACVMD just eats it, and does nothing presently */
#endif
  }
#endif
  else if (argc == 4 && !strupncmp(argv[2], "selectmol", CMDLEN)) {
    // undocumented command for internal use only!
    int ind;
    if (Tcl_GetInt(interp, argv[3], &ind) != TCL_OK) retval = TCL_HELP;
    else app->menu_select_mol(argv[1], ind);
  }
  else if(argc == 3) {
    if(!strupncmp(argv[2],"on",CMDLEN))
      app->menu_show(argv[1], 1);
    else if (!strupncmp(argv[2],"off",CMDLEN))
      app->menu_show(argv[1], 0);
    else if (!strupncmp(argv[2],"loc",CMDLEN)) {
      int x, y;
      if (app->menu_location(argv[1], x, y)) {
        char buf[20];
        sprintf(buf, "%d %d", x, y);
        Tcl_SetResult(interp, buf, TCL_VOLATILE);
      } 
      else {
        Tcl_AppendResult(interp, "menu loc: menu '", argv[1], 
          "' does not exist.", NULL);
        retval = TCL_ERROR;
      }
    } 
    else if (!strupncmp(argv[2], "status", CMDLEN))
      Tcl_AppendResult(interp, app->menu_status(argv[1]) ? "on" : "off", NULL);
    else  retval = TCL_HELP;
  }
  else if (argc == 5 && !strupncmp(argv[2],"move",CMDLEN))
    app->menu_move(argv[1], atoi(argv[3]), atoi(argv[4]));
  else
    retval = TCL_HELP;
  
  if (retval == TCL_HELP) {
    Tcl_SetResult(interp, 
      (char *) "Usage:\n"
      "\tmenu list                 -- returns list of available menus\n"
      "\tmenu <name> on            -- turn menu with given name on\n"
      "\tmenu <name> off           -- turn menu with given name off\n"
      "\tmenu <name> status         -- returns 'on' or 'off'\n"
      "\tmenu <name> loc           -- returns current position of menu\n"
      "\tmenu <name> move x y      -- move menu to given position\n"
      "\tmenu tk add <name> <tk window> [<menu path>]\n"
      "\t      -- add Tk menu to Extensions\n"
      "\tmenu tk register <name> <procname> [<menu path>]\n" 
      "\t      -- same as 'add', but <procname> returns tk window handle.\n"
      "\tmenu tk remove <name> -- remove menu from Extensions\n"
      "The Tk 'menu' command is also available.",
      TCL_STATIC);
    retval = TCL_ERROR;
  }
 
  return retval;
}

