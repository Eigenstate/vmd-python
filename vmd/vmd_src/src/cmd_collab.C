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
 *      $RCSfile: cmd_collab.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.10 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for collaborative/remote control of VMD sessions
 ***************************************************************************/

#include "CommandQueue.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "utilities.h"
#include "config.h"    // for CMDLEN
#include <stdlib.h>
#include <tcl.h>

#include "VMDCollab.h"

int text_cmd_collab(ClientData cd, Tcl_Interp *interp, int argc,
                    const char *argv[]) {
  VMDApp *app = (VMDApp *)cd;

  if (argc == 4 && !strupncmp(argv[1], "connect", CMDLEN)) {
    const char *host = argv[2];
    int port;
    if (Tcl_GetInt(interp, argv[3], &port) != TCL_OK) return TCL_ERROR;
    if (!strcmp(host, Tcl_GetHostName())) {
      if (!app->vmdcollab->startserver(port)) {
        Tcl_AppendResult(interp, "Failed to start server on port ", argv[3], NULL);
        return TCL_ERROR;
      }
    }
    if (!app->vmdcollab->connect(host, port)) {
      Tcl_AppendResult(interp, "Failed to connect to vmdcollab at ", host,
          "port: ", argv[3], NULL);
      return TCL_ERROR;
    }
    return TCL_OK;
  }
  if (argc == 2 && !strupncmp(argv[1], "disconnect", CMDLEN)) {
    app->vmdcollab->disconnect();
    return TCL_OK;
  }

  Tcl_SetResult(interp, (char *) "Usage: vmdcollab [connect <host> <port> | disconnect",
      TCL_STATIC);
  return TCL_ERROR;
}


