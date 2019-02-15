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
 *      $RCSfile: cmd_material.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.31 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for manipulating materials
 ***************************************************************************/

#include "CmdMaterial.h"
#include "MaterialList.h"
#include "config.h"
#include "VMDApp.h"
#include <stdlib.h>
#include <ctype.h>
#include <tcl.h>

int text_cmd_material(ClientData cd, Tcl_Interp *interp, int argc,
                     const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;
  MaterialList *mlist = app->materialList; 

  if (argc < 2) {
    Tcl_SetResult(interp,
      (char *)
      "material list\n"
      "material settings <name>\n"
      "material add [<newname>] [copy <copyfrom>]\n"
      "material rename <oldname> <newname>\n"
      "material change [ambient|specular|diffuse|shininess|mirror|opacity|outline|outlinewidth|transmode] "
      "<name> <value>\n",
      TCL_STATIC);
    return TCL_ERROR;
  }
  if (!strupncmp(argv[1], "list", CMDLEN)) {
   
    for (int i=0; i<mlist->num(); i++) 
      Tcl_AppendElement(interp, mlist->material_name(i)); 

  } else if (!strupncmp(argv[1], "settings", CMDLEN) && argc >= 3) {
    int ind = mlist->material_index(argv[2]);
    if (ind < 0) {
      Tcl_AppendResult(interp, 
        "material settings: material '", argv[2], 
             "' has not been defined", NULL); 
      return TCL_OK;
    }
    char buf[20];
    sprintf(buf,"%f",mlist->get_ambient(ind));
    Tcl_AppendElement(interp, buf);
    sprintf(buf,"%f",mlist->get_specular(ind));
    Tcl_AppendElement(interp, buf);
    sprintf(buf,"%f",mlist->get_diffuse(ind));
    Tcl_AppendElement(interp, buf);
    sprintf(buf,"%f",mlist->get_shininess(ind));
    Tcl_AppendElement(interp, buf);
    sprintf(buf,"%f",mlist->get_mirror(ind));
    Tcl_AppendElement(interp, buf);
    sprintf(buf,"%f",mlist->get_opacity(ind));
    Tcl_AppendElement(interp, buf);
    sprintf(buf,"%f",mlist->get_outline(ind));
    Tcl_AppendElement(interp, buf);
    sprintf(buf,"%f",mlist->get_outlinewidth(ind));
    Tcl_AppendElement(interp, buf);
    sprintf(buf,"%f",mlist->get_transmode(ind));
    Tcl_AppendElement(interp, buf);
  } else if (!strupncmp(argv[1], "add", CMDLEN) && argc >= 2) {
    // material add
    const char *newname = NULL;
    const char *copyfrom = NULL;
    if (argc == 3) {
      // material add <name>
      newname = argv[2];
    } else if (argc == 4) {
      // material add copy <copyfrom>
      copyfrom = argv[3];
    } else if (argc == 5) {
      // material add <name> copy <copyfrom>
      newname = argv[2];
      copyfrom = argv[4];
    }
    const char *result = app->material_add(newname, copyfrom);
    if (!result) {
      Tcl_AppendResult(interp, "Unable to add material.", NULL);
      return TCL_ERROR;
    }
    Tcl_AppendResult(interp, result, NULL);

  } else if (!strupncmp(argv[1], "rename", CMDLEN) && argc >= 4) {
    if (!app->material_rename(argv[2], argv[3])) {
      Tcl_AppendResult(interp, "Unable to rename material '", argv[2],
          "' to '", argv[3], "'.", NULL);
      return TCL_ERROR;
    }
  } else if (!strupncmp(argv[1], "change", CMDLEN) && argc >= 5) {
    const char *prop = argv[2];
    const char *name = argv[3];
    const char *value = argv[4];
    int matprop = -1;
    if (!strupcmp(prop, "ambient")) matprop = MAT_AMBIENT;
    else if (!strupcmp(prop, "specular")) matprop = MAT_SPECULAR; 
    else if (!strupcmp(prop, "diffuse")) matprop = MAT_DIFFUSE; 
    else if (!strupcmp(prop, "shininess")) matprop = MAT_SHININESS; 
    else if (!strupcmp(prop, "mirror")) matprop = MAT_MIRROR; 
    else if (!strupcmp(prop, "opacity")) matprop = MAT_OPACITY; 
    else if (!strupcmp(prop, "outline")) matprop = MAT_OUTLINE; 
    else if (!strupcmp(prop, "outlinewidth")) matprop = MAT_OUTLINEWIDTH; 
    else if (!strupcmp(prop, "transmode")) matprop = MAT_TRANSMODE; 
    if (!app->material_change(name, matprop, (float) atof(value))) {
      Tcl_AppendResult(interp, "Unable to change property ", prop,
          " of material ", name, " to ", value, NULL);
      return TCL_ERROR;
    }
  } else if (!strupncmp(argv[1], "delete", CMDLEN) && argc == 3) {
    if (!app->material_delete(argv[2])) {
      Tcl_AppendResult(interp, "Unable to delete material: ", argv[2], NULL);
      return TCL_ERROR;
    }
  } else if (!strupncmp(argv[1], "default", CMDLEN) && argc ==3) {
    int id;
    if (Tcl_GetInt(interp, argv[2], &id) != TCL_OK) return TCL_ERROR;
    if (!app->material_restore_default(id)) {
      Tcl_AppendResult(interp, "Unable to restore default for material ",
          argv[2], NULL);
      return TCL_ERROR;
    }
  } else 
    return TCL_ERROR;
  return TCL_OK;
}

