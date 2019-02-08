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
 *      $RCSfile: cmd_util.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.36 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Command objects for doing various utilities, such as executing a shell,
 * displaying help, or quitting
 *
 ***************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <tcl.h>

#include "config.h"
#include "CommandQueue.h"
#include "utilities.h"
#include "VMDApp.h"
#include "TextEvent.h"

int text_cmd_sleep(ClientData, Tcl_Interp *, int argc, const char *argv[]) {

  if(argc == 2) {
    vmd_sleep((int) atof(argv[1]));
  }
  else {
    return TCL_ERROR;
  }
  // if here, no problem with command
  return TCL_OK;
}

int text_cmd_gopython(ClientData cd, Tcl_Interp *interp, 
                      int argc, const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;
  CommandQueue *cmdQueue = app->commandQueue;

  if (argc == 1) {
    if (!app->textinterp_change("python")) {
      Tcl_AppendResult(interp, "Unable to change to Python interpreter.", 
          NULL);
      return TCL_ERROR;
    }
  } else if (argc == 2) {
    // change to python, run the script, then change back to Tcl
    if (app->textinterp_change("python")) {
      app->logfile_read(argv[1]);
      app->textinterp_change("tcl");
    } else {
      Tcl_AppendResult(interp, "Unable to change to Python interpreter.", 
          NULL);
      return TCL_ERROR;
    }
  } else if (argc == 3 && !strupncmp(argv[1], "-command", CMDLEN)) {
    // run the specified text in the Python interpreter.
    if (app->textinterp_change("python")) {
      cmdQueue->runcommand(new PythonEvalEvent(argv[2]));
      app->textinterp_change("tcl");
    } else {
      Tcl_AppendResult(interp, "Unable to change to Python interpreter.", 
          NULL);
      return TCL_ERROR;
    }
  } else {
    Tcl_AppendResult(interp, "gopython usage: \n", 
    "gopython            -- switch to python interpreter\n",
    "gopython <filename> -- run given file in Python interpreter\n",
    "gopython -command <cmd> -- execute <cmd> as literal Python command\n",
    NULL);
    return TCL_ERROR;
  }
  return TCL_OK;
}

