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
 *	$RCSfile: cmd_tool.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.36 $	$Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Text commands for control of the VMD VR "Tools" 
 *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <tcl.h>
#include "JString.h"
#include "config.h"
#include "UIObject.h"
#include "CommandQueue.h"
#include "Displayable.h"
#include "DispCmds.h"
#include "Matrix4.h"
#include "MoleculeList.h"
#include "Command.h"
#include "P_Tracker.h"
#include "P_Buttons.h"
#include "P_Feedback.h"
#include "P_Tool.h"
#include "P_CmdTool.h"
#include "VMDApp.h"

int text_cmd_tool(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;
  CommandQueue *cmdQueue = app->commandQueue;

  char buf[400];

  if(argc<2) {
    Tcl_SetResult(interp, 
      (char *)
      "tool create <type> [<name> [<name> ...]]\n"
      "tool change <type> [<toolid>]\n"
      "tool scale <scale> [<toolid>]\n"
      "tool scaleforce <scale> [<toolid>]\n"
      "tool offset <x> <y> <z> [<toolid>]\n"
      "tool delete [<toolid>]\n"
#if 0
      "tool info [<toolid>]\n"
#endif
      "tool rep <toolid> <mol id> <rep id>\n"
      "tool adddevice <name> [<toolid>]\n"
      "tool removedevice <name> [<toolid>]\n"
      "tool callback on/off",
      TCL_STATIC);
    return TCL_ERROR;
  }

  /* creating a new tool with some number of USLs */
  if(!strupncmp(argv[1], "create", CMDLEN) && argc>=3) {
    if (!app->tool_create(argv[2], argc-3, argv+3)) {
      Tcl_AppendResult(interp, "Failed to create new tool.", NULL);
      return TCL_ERROR;
    }
    return TCL_OK;
  } 

  /* changing the tool but preserving the sensor */
  if(!strupncmp(argv[1], "change", CMDLEN) && (argc==4 || argc==3)) {
    int i=0;

    if(argc==4) { // default to 0
      if (Tcl_GetInt(interp, argv[3], &i) != TCL_OK) 
        return TCL_ERROR;
    }
    if (!app->tool_change_type(i, argv[2])) {
      Tcl_AppendResult(interp, "Unable to change tool type.", NULL);
      return TCL_ERROR;
    }
    return TCL_OK;
  }

  /* Setting the scale of a tool */
  if(!strupncmp(argv[1], "scale", CMDLEN) && (argc==3 || argc==4)) {
    int i=0;
    double dscale=0.0;
    float scale=0.0f;
    if(argc==4) {  // default to 0
      if (Tcl_GetInt(interp, argv[3], &i) != TCL_OK) 
        return TCL_ERROR;
    }
    if (Tcl_GetDouble(interp, argv[2], &dscale) != TCL_OK)
      return TCL_ERROR;
    scale = (float)dscale;
    if (app->tool_set_position_scale(i, scale)) {
      return TCL_OK;
    }
    Tcl_AppendResult(interp, "Unable to set position scale", NULL);
    return TCL_ERROR;
  }

  /* Setting the scale of the force on a tool */
  if(!strupncmp(argv[1], "scaleforce", CMDLEN) && (argc==3 || argc==4)) {
    int i=0;
    double dscale=0;
    float scale=0;
    if(argc==4) {  // default to 0
      if (Tcl_GetInt(interp, argv[3], &i) != TCL_OK)
        return TCL_ERROR;
    }
    if (Tcl_GetDouble(interp, argv[2], &dscale) != TCL_OK)
      return TCL_ERROR;
    scale = (float)dscale;
    if (app->tool_set_force_scale(i, scale))
      return TCL_OK;
    Tcl_AppendResult(interp, "Unable to set force scale", NULL);
    return TCL_ERROR;
  }

  /* Setting the scale of the spring on a tool */
  if(!strupncmp(argv[1], "scalespring", CMDLEN) && (argc==3 || argc==4)) {
    int i=0;
    double dscale=0;
    float scale=0;
    if(argc==4) { // default to 0
      if (Tcl_GetInt(interp, argv[3], &i) != TCL_OK)
        return TCL_ERROR;
    }
    if (Tcl_GetDouble(interp, argv[2], &dscale) != TCL_OK)
      return TCL_ERROR;
    scale = (float)dscale;
    if (app->tool_set_spring_scale(i, scale))
      return TCL_OK;
    Tcl_AppendResult(interp, "Unable to set spring scale", NULL);
    return TCL_ERROR;
  }

  /* Setting the offset of a tool */
  if(!strupncmp(argv[1], "offset", CMDLEN) && (argc==5 || argc==6)) {
    int i=0,j;
    double d_offset[3];
    float offset[3];
    if(argc==6) { // default to 0
      if (Tcl_GetInt(interp, argv[5], &i) != TCL_OK)
        return TCL_ERROR;
    }
    
    if (Tcl_GetDouble(interp, argv[2], &d_offset[0]) != TCL_OK)
      return TCL_ERROR;
    if (Tcl_GetDouble(interp, argv[3], &d_offset[1]) != TCL_OK)
      return TCL_ERROR;
    if (Tcl_GetDouble(interp, argv[4], &d_offset[2]) != TCL_OK)
      return TCL_ERROR;
    for(j=0;j<3;j++) offset[j] = (float)d_offset[j];
    cmdQueue->runcommand(new CmdToolOffset(offset,i));

    sprintf(buf,"Setting offset of tool %i.", i);
    Tcl_AppendResult(interp, buf, NULL);
    return TCL_OK;

  }

  /* deleting a tool */
  if(!strupncmp(argv[1], "delete", CMDLEN) && (argc==3 || argc==2)) {
    int i=0;

    if(argc==3) { // default to 0
      if (Tcl_GetInt(interp, argv[2], &i) != TCL_OK)
        return TCL_ERROR;
    }
    cmdQueue->runcommand(new CmdToolDelete(i));
    sprintf(buf,"Deleting tool %i.\n",i);
    Tcl_AppendResult(interp, buf, NULL);
    return TCL_OK;
  }

#if 0 // XXX
  /* getting info about a tool */
  if(!strupncmp(argv[1], "info", CMDLEN) && (argc==3 || argc==2)) {
    int i=0;
    Tool *tool;

    if (argc==3) {  // default to 0
      if (Tcl_GetInt(interp, argv[2], &i) != TCL_OK)
        return TCL_ERROR;
    }
    tool = vmdGlobal.uiVR->gettool(i);
    if (tool==NULL) {
      Tcl_AppendResult(interp, "No such tool.", NULL);
      return TCL_ERROR;
    }

    sprintf(buf,"Info for tool %i (%s)\n",i,tool->type_name());
    Tcl_AppendResult(interp,buf, NULL);

    const float *pos = tool->position();
    const Matrix4 *rot = tool->orientation();
    if (pos==NULL) {
      Tcl_AppendResult(interp, "Tool has no position!", NULL);
      return TCL_ERROR;
    }
      
    sprintf(buf, "Postion: %.2f %.2f %.2f\n"
                 "Orientation: %.2f %.2f %.2f\n"
                 "             %.2f %.2f %.2f\n"
                 "             %.2f %.2f %.2f\n",
            pos[0],pos[1],pos[2],
            rot->mat[4*0+0],rot->mat[4*0+1],rot->mat[4*0+2],
            rot->mat[4*1+0],rot->mat[4*1+1],rot->mat[4*1+2],
            rot->mat[4*2+0],rot->mat[4*2+1],rot->mat[4*2+2]);
    Tcl_AppendResult(interp,buf, NULL);

    int j=0;
    char *devices[5];
    const float *offset;
    float scale;

    offset = tool->getoffset();
    if (offset==NULL) {
      Tcl_AppendResult(interp, "tool info:\n", "NULL Offset...?\n", NULL); 
      return TCL_ERROR;
    }

    scale = tool->getscale();

    tool->getdevices(devices);
    JString buf2;
    while(devices[j]!=NULL) {
      buf2 += devices[j++];
      buf2 += " ";
    }

    sprintf(buf,"Scale: %.2f\n"
            "Offset: %.2f %.2f %.2f\n"
            "USL: %s\n", scale, offset[0],
            offset[1], offset[2], (const char *)buf2);
    Tcl_AppendResult(interp,buf, NULL);
    return TCL_OK;
  }
#endif  

  /* Assigning a representation to a tool */
  if(!strupncmp(argv[1], "rep", CMDLEN)) {
    if (argc != 3 && argc != 5) {
      Tcl_AppendResult(interp, "tool rep usage:\n",
                       "Usage: tool rep toolnum [molid repnum]", NULL); 
      return TCL_ERROR;
    }
    int toolnum, molid, repnum;
    toolnum = atoi(argv[2]);
    if (argc == 5) {
      molid = atoi(argv[3]);
      repnum = atoi(argv[4]);
    } else {
      molid = repnum = -1;
    }
    cmdQueue->runcommand(new CmdToolRep(toolnum, molid, repnum));
    return TCL_OK;
  }

  /* Adding a device to a tool */
  if(!strupncmp(argv[1], "adddevice", CMDLEN) &&
     (argc == 3 || argc == 4)) {
    int i=0;

    if(argc==4) { // default to 0
      if (Tcl_GetInt(interp, argv[3], &i) != TCL_OK)
        return TCL_ERROR;
    }
    cmdQueue->runcommand(new CmdToolAddDevice(argv[2],i));
    return TCL_OK;
  }

  /* Removing a device to a tool */
  if(!strupncmp(argv[1], "removedevice", CMDLEN) &&
     (argc == 3 || argc == 4)) {
    int i=0;

    if(argc==4) { // default to 0
      if (Tcl_GetInt(interp, argv[3], &i) != TCL_OK)
        return TCL_ERROR;
    }
    cmdQueue->runcommand(new CmdToolDeleteDevice(argv[2],i));
    return TCL_OK;
  }

  /* Turning on callbacks for a tool */
  if(!strupncmp(argv[1], "callback", CMDLEN)) {
    if(argc==3) {
      int on=-1;
      if (Tcl_GetBoolean(interp, argv[2], &on) != TCL_OK)
        return TCL_ERROR;
      if (on!=-1) {
        cmdQueue->runcommand(new CmdToolCallback(on));
        return TCL_OK;
      }
    }
    Tcl_AppendResult(interp," tool callback usage:\n",
                     "Usage: tool callback on/off [<toolid>]",NULL);
    return TCL_ERROR;
  }
    
  return TCL_ERROR;
}

