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
 *      $RCSfile: cmd_videostream.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.9 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for interactive MD simulation connection control
 ***************************************************************************/

// #include "CmdVideoStream.h"
#include "VideoStream.h"
#include "CommandQueue.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "utilities.h"
#include "config.h"    // for CMDLEN
#include <stdlib.h>
#include <tcl.h>

int videostream_usage(Tcl_Interp *interp) {
  Tcl_AppendResult(interp,
    "Usage: videostream [options]: \n",
    "  client connect <hostname> <port>\n",
    "  client listen <port>\n",
    "  server connect <hostname> <port>\n",
    "  server listen <port>\n",
    "  disconnect",
//    "  detach\n",
//    "  pause [on|off|toggle]\n",
//    "  kill\n",
//    "  transfer <rate>\n",
    NULL);
  return TCL_ERROR;
}


int text_cmd_videostream(ClientData cd, Tcl_Interp *interp, int argc,
                         const char *argv[]) {

#if 1
  VMDApp *app = (VMDApp *)cd;
#if 0
  CommandQueue *cmdQueue = app->commandQueue; 
#endif

  if (app->uivs == NULL) {
    Tcl_AppendResult(interp, "Videostream support is not available.\n", NULL);
    return TCL_ERROR;
  }

  if (argc == 1) {
    return videostream_usage(interp);

#if 0
  } else if (!strupncmp(argv[1], "pause", CMDLEN)) {   
    if ((argc == 3) && (!strupncmp(argv[2], "toggle", CMDLEN))) { //"videostream pause"
      app->uivs->togglepause();
      cmdQueue->runcommand(new CmdVideoStream(CmdVideoStream::PAUSE_TOGGLE));
    }
    else if ((argc == 3) && (!strupncmp(argv[2], "on", CMDLEN))) {   //"videostream pause on"
      app->uivs->pause();
      cmdQueue->runcommand(new CmdVideoStream(CmdVideoStream::PAUSE_ON));  
    }   
    else if ((argc == 3) && (!strupncmp(argv[2], "off", CMDLEN))) {   //"videostream pause off"
      app->uivs->unpause();
      cmdQueue->runcommand(new CmdVideoStream(CmdVideoStream::PAUSE_OFF));  
    }  
    else {
      Tcl_AppendResult(interp, "Wrong arguments: videostream pause <on|off|toggle>", NULL); 
      return TCL_ERROR;
    }
    
    if (!app->uivs->cli_connected()) { 
      Tcl_AppendResult(interp, "No videostream connection available.", NULL);
      return TCL_ERROR;
    }
#endif

  } else if (!strupncmp(argv[1], "disconnect", CMDLEN)) {   
    if (app->uivs->cli_connected()) {
      app->uivs->cli_disconnect();
      Tcl_AppendResult(interp, "Client videostream disconnected.", NULL);
      return TCL_OK;
    } else if (app->uivs->srv_connected()) {
      app->uivs->srv_disconnect();
      Tcl_AppendResult(interp, "Server videostream disconnected.", NULL);
      return TCL_OK;
    } else {
      Tcl_AppendResult(interp, "No active videostream connection.", NULL);
      return TCL_ERROR;
    }
  } else if (!strupncmp(argv[1], "bitrate", CMDLEN)) {   
    if (argc == 3) {
      int brMbps = atoi(argv[3]);
      app->uivs->set_target_bitrate_Mbps(brMbps);
    } else {
      return videostream_usage(interp);
    }
  } else if (!strupncmp(argv[1], "framerate", CMDLEN)) {   
    if (argc == 3) {
      int tfps = atoi(argv[3]);
      app->uivs->set_target_frame_rate(tfps);
    } else {
      return videostream_usage(interp);
    }

//
// Handle "client" subcommands
//
  } else if ((argc >= 4) && (!strupncmp(argv[1], "client", CMDLEN)) ) {
    if ((argc == 5) && !strupncmp(argv[2], "connect", CMDLEN)) {
      int port = atoi(argv[4]);
      if (app->uivs->cli_connected()) {
        char buf[500];
        sprintf(buf, "Can't connect; already connected to videostream server on"
                     "host %s over port %d", app->uivs->cli_gethost(), 
                      app->uivs->cli_getport());
        Tcl_SetResult(interp, buf, TCL_VOLATILE);
        return TCL_ERROR;
      }
      if (app->uivs->cli_connect(argv[3], port)) {
        Tcl_AppendResult(interp, "Client unable to connect to server host ", argv[3], 
                         " on port ", argv[4], NULL);
        return TCL_ERROR;
      } else {
        Tcl_AppendResult(interp, "Client connected to server host ", argv[3], 
                         " on port ", argv[4], NULL);
        return TCL_OK;
      }
    } else if ((argc == 4) && !strupncmp(argv[2], "listen", CMDLEN)) {
      int port = atoi(argv[3]);
      if (app->uivs->cli_connected()) {
         Tcl_AppendResult(interp, "Can't listen for new server; already connected to videostream server", NULL);
        return TCL_ERROR;
      }
      if (app->uivs->cli_listen(port)) {
        Tcl_AppendResult(interp, "Unable to listen for videostream servers on port ", argv[3], NULL);
        return TCL_ERROR;
      }
    } else {
      return videostream_usage(interp);
    }
//
// Handle "server" subcommands
//
  } else if ((argc >= 4) && (!strupncmp(argv[1], "server", CMDLEN)) ) {
    if ((argc == 4) && (!strupncmp(argv[2], "listen", CMDLEN)) ) {
      int port = atoi(argv[3]);
      if (app->uivs->srv_connected()) {
         Tcl_AppendResult(interp, "Can't listen for new clients; already connected to videostream client", NULL);
        return TCL_ERROR;
      }
      if (app->uivs->srv_listen(port)) {
        Tcl_AppendResult(interp, "Unable to listen for videostream clients on port ", argv[3], NULL);
        return TCL_ERROR;
      }
    } else if ((argc == 5) && (!strupncmp(argv[2], "connect", CMDLEN)) ) {
      int port = atoi(argv[4]);
      if (app->uivs->srv_connected()) {
        char buf[500];
        sprintf(buf, "Can't connect; already connected to videostream client on"
                     "host %s over port %d", app->uivs->srv_gethost(),
                      app->uivs->srv_getport());
        Tcl_SetResult(interp, buf, TCL_VOLATILE);
        return TCL_ERROR;
      }
      if (app->uivs->srv_connect(argv[3], port)) {
        Tcl_AppendResult(interp, "Server unable to connect to client host ", argv[3],
                         " on port ", argv[4], NULL);
        return TCL_ERROR;
      } else {
        Tcl_AppendResult(interp, "Server connected to client host ", argv[3],         
                         " on port ", argv[4], NULL);
        return TCL_OK;
      }
    } else {
      return videostream_usage(interp);
    }
 
#if 0
  } else if (argc == 2) {
    if (!app->uivs->cli_connected()) {
      Tcl_AppendResult(interp, "No videostream connection available.", NULL);
      return TCL_ERROR;
    }
    else if (!strupncmp(argv[1], "detach", CMDLEN)) {
      app->uivs->detach();
      cmdQueue->runcommand(new CmdVideoStream(CmdVideoStream::DETACH));
    } else if (!strupncmp(argv[1], "kill", CMDLEN)) {
      app->uivs->kill();
      cmdQueue->runcommand(new CmdVideoStream(CmdVideoStream::KILL));
    } else {
      return videostream_usage(interp);
    }

  } else if ((argc == 3) && (!strupncmp(argv[1], "transfer", CMDLEN)) ) {
    int rate = atoi(argv[2]);
    app->uivs->set_trans_rate(rate);
    cmdQueue->runcommand(new CmdVideoStreamRate(CmdVideoStreamRate::TRANSFER, rate));
#endif

  } else {
    return videostream_usage(interp);
  }

  return TCL_OK; // No error
#else
  Tcl_AppendResult(interp,
    "Video streaming functionality not present.  Recompile with it enabled.", NULL);
  return TCL_ERROR;
#endif
}


