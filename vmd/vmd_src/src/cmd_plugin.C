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
 *      $RCSfile: cmd_plugin.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.25 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for control of plugin loading
 ***************************************************************************/

#include <tcl.h>
#include "tcl_commands.h"
#include "VMDApp.h"
#include "config.h"

int text_cmd_plugin(ClientData cd, Tcl_Interp *interp, int argc,
                     const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;
  if (!app) 
    return TCL_ERROR;

  // plugin dlopen <filename>
  if (argc == 3 && !strupncmp(argv[1], "dlopen", CMDLEN)) {
    int rc = app->plugin_dlopen(argv[2]);
    if (rc < 0) {
      Tcl_AppendResult(interp, "Unable to dlopen plugin file ", argv[2], NULL);
      return TCL_ERROR;
    } 
    Tcl_SetObjResult(interp, Tcl_NewIntObj(rc));
    return TCL_OK;
  }
  // plugin update  -- updates list of plugins
  if (argc == 2 && !strupncmp(argv[1], "update", CMDLEN)) {
    app->plugin_update();
    return TCL_OK;
  }

  // plugin list [type]: returns list of category/name pairs.  If optional
  // type is specified, return only plugins of that type.
  if ((argc == 2 || argc == 3) && !strupncmp(argv[1], "list", CMDLEN)) {
    const char *type = NULL;
    if (argc == 3)
      type = argv[2];
    
    PluginList pluginlist;
    app->list_plugins(pluginlist, type);
    const int num = pluginlist.num();
    Tcl_Obj *result = Tcl_NewListObj(0, NULL);
    for (int i=0; i<num; i++) {
      vmdplugin_t *p = pluginlist[i];
      Tcl_Obj *listelem = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, listelem, Tcl_NewStringObj(p->type,-1));
      Tcl_ListObjAppendElement(interp, listelem, Tcl_NewStringObj(p->name,-1));
      Tcl_ListObjAppendElement(interp, result, listelem);
    }
    Tcl_SetObjResult(interp, result);
    return TCL_OK;
  }
  // plugin info <type> <name> <varname>
  // Puts plugin information for the specified plugin into the array variable
  // specified by varname.  The following array keys will be used: type,
  // name, author, majorversion, minorversion, reentrant.
  // returns 1 if plugin information was found, or 0 if no plugin information
  // is available for that type and name.
  if (argc == 5 && !strupncmp(argv[1], "info", CMDLEN)) {
    vmdplugin_t *p = app->get_plugin(argv[2], argv[3]);
    if (!p) {
      Tcl_SetResult(interp, (char *) "0", TCL_STATIC);
      return TCL_OK;
    }
    char major[32], minor[32], reentrant[32];
    sprintf(major, "%d", p->majorv);
    sprintf(minor, "%d", p->minorv);
    sprintf(reentrant, "%d", p->is_reentrant);

    if (!Tcl_SetVar2(interp,argv[4], "type", p->type, TCL_LEAVE_ERR_MSG) ||
        !Tcl_SetVar2(interp,argv[4], "name", p->name, TCL_LEAVE_ERR_MSG) ||
        !Tcl_SetVar2(interp,argv[4], "author", p->author, TCL_LEAVE_ERR_MSG)  ||
        !Tcl_SetVar2(interp,argv[4], "majorversion", major, TCL_LEAVE_ERR_MSG) ||
        !Tcl_SetVar2(interp,argv[4], "minorversion", minor, TCL_LEAVE_ERR_MSG) ||
        !Tcl_SetVar2(interp,argv[4], "reentrant", reentrant, TCL_LEAVE_ERR_MSG)) {
      Tcl_AppendResult(interp, "Unable to return plugin information in variable ", argv[4], NULL);
      return TCL_ERROR;
    }
    Tcl_SetResult(interp, (char *) "1", TCL_STATIC);
    return TCL_OK;
  }
  Tcl_AppendResult(interp, "Usage: \n\tplugin dlopen <filename> -- Load plugins from a dynamic library\n",
      "\tplugin update -- Update the list of plugins in the GUI\n",
      "\tplugin list [<plugin type>] -- List all plugins of the given type\n", 
      "\tplugin info <type> <name> <arrayname> -- Store info about plugin in array\n",
      NULL);
  return TCL_ERROR;
}

