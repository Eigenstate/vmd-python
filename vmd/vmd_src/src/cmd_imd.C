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
 *      $RCSfile: cmd_imd.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.27 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for interactive MD simulation connection control
 ***************************************************************************/

#ifdef VMDIMD
#include "CmdIMD.h"
#endif

#include "IMDMgr.h"
#include "CommandQueue.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "utilities.h"
#include "config.h"    // for CMDLEN
#include <stdlib.h>
#include <tcl.h>

int text_cmd_imd(ClientData cd, Tcl_Interp *interp, int argc,
                     const char *argv[]) {

#ifdef VMDIMD
  VMDApp *app = (VMDApp *)cd;
  CommandQueue *cmdQueue = app->commandQueue; 

  if (argc == 1) {
    Tcl_AppendResult(interp, 
    "Need parameters for 'imd' command.  Possibilities include: \n",
    "pause [on|off|toggle]\n", 
    "detach\n",
    "kill\n",
    "connect <hostname> <port>\n",
    "transfer <rate>\n",
    "keep <rate>\n",
    "copyunitcell <on|off>\n",
    NULL);
    return TCL_ERROR;

  } else if (!strupncmp(argv[1], "pause", CMDLEN)) {   
    if ((argc == 3) && (!strupncmp(argv[2], "toggle", CMDLEN))) { //"imd pause"
      app->imdMgr->togglepause();
      cmdQueue->runcommand(new CmdIMDSim(CmdIMDSim::PAUSE_TOGGLE));
    }
    else if ((argc == 3) && (!strupncmp(argv[2], "on", CMDLEN))) {   //"imd pause on"
      app->imdMgr->pause();
      cmdQueue->runcommand(new CmdIMDSim(CmdIMDSim::PAUSE_ON));  
    }   
    else if ((argc == 3) && (!strupncmp(argv[2], "off", CMDLEN))) {   //"imd pause off"
      app->imdMgr->unpause();
      cmdQueue->runcommand(new CmdIMDSim(CmdIMDSim::PAUSE_OFF));  
    }  
    else {
      Tcl_AppendResult(interp, "Wrong arguments: imd pause <on|off|toggle>", NULL); 
      return TCL_ERROR;
    }
    
    if (!app->imdMgr->connected()) { 
      Tcl_AppendResult(interp, "No IMD connection available.", NULL);
      return TCL_ERROR;
    }

  } else if ((argc == 4) && (!strupncmp(argv[1], "connect", CMDLEN)) ) {
    int port = atoi(argv[3]);
    Molecule *mol = app->moleculeList->top();
    if (!mol) {
      Tcl_AppendResult(interp,      
        "Can't connect, no molecule loaded", NULL); 
      return TCL_ERROR;
    }
    if (app->imdMgr->connected()) {
      char buf[500];
      sprintf(buf, "Can't connect; already connected to simulation running on"
                   "host %s over port %d", app->imdMgr->gethost(), 
                   app->imdMgr->getport());
      Tcl_SetResult(interp, buf, TCL_VOLATILE);
      return TCL_ERROR;
    }
    if (!app->imd_connect(mol->id(), argv[2], port)) {
      Tcl_AppendResult(interp, "Unable to connect to host ", argv[2], 
        " on port ", argv[3], NULL);
      return TCL_ERROR;
    }

  } else if (argc == 2) {
    if (!app->imdMgr->connected()) {
      Tcl_AppendResult(interp, "No IMD connection available.", NULL);
      return TCL_ERROR;
    }
    else if (!strupncmp(argv[1], "detach", CMDLEN)) {
      app->imdMgr->detach();
      cmdQueue->runcommand(new CmdIMDSim(CmdIMDSim::DETACH));
    } else if (!strupncmp(argv[1], "kill", CMDLEN)) {
      app->imdMgr->kill();
      cmdQueue->runcommand(new CmdIMDSim(CmdIMDSim::KILL));
    } else {
      Tcl_AppendResult(interp, 
        "Usage: imd [pause | detach | kill]", NULL); 
      return TCL_ERROR; 
    }

  } else if ((argc == 3) && (!strupncmp(argv[1], "transfer", CMDLEN)) ) {
    int rate = atoi(argv[2]);
    app->imdMgr->set_trans_rate(rate);
    cmdQueue->runcommand(new CmdIMDRate(CmdIMDRate::TRANSFER, rate));

  } else if ((argc == 3) && (!strupncmp(argv[1], "keep", CMDLEN)) ) {
    int rate = atoi(argv[2]);
    app->imdMgr->set_keep_rate(rate);
    cmdQueue->runcommand(new CmdIMDRate(CmdIMDRate::KEEP, rate));

  } else if ((argc == 3) && (!strupncmp(argv[1], "copyunitcell", CMDLEN)) ) {
    if (!strupncmp(argv[2], "on", CMDLEN)) {
      app->imdMgr->set_copyunitcell(1);
      cmdQueue->runcommand(new CmdIMDCopyUnitCell(CmdIMDCopyUnitCell::COPYCELL_ON));
    } else {
      app->imdMgr->set_copyunitcell(0);
      cmdQueue->runcommand(new CmdIMDCopyUnitCell(CmdIMDCopyUnitCell::COPYCELL_OFF));
    }
  } else {
    return TCL_ERROR;
  }

  return TCL_OK; // No error
#else
  Tcl_AppendResult(interp,
    "IMD functionality not present.  Recompile with IMD enabled.", NULL);
  return TCL_ERROR;
#endif
}


