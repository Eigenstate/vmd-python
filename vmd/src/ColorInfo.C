/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: ColorInfo.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.33 $	$Date: 2010/12/16 04:08:09 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Routines for Tcl to get the color and color category information
 *
 ***************************************************************************/

#include <string.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#include "tcl.h"
#include "TclCommands.h"
#include "VMDApp.h"

// They are:
// colorinfo categories
//   Display, Axes, ..., Structure, ...
// colorinfo category Display
//   Background
// colorinfo category Axes
//   X, Y, Z, Origin, Labels
// colorinfo num
//   value of REGCLRS
// colorinfo max
//   value of MAXCOLORS
// colorinfo colors
//   blue, red, ..., black
// colorinfo index blue
//   0
// colorinfo rgb blue
//   0.25 0.25 1.
// colorinfo scale method
//   0
// colorinfo scale midpoint
//   0.5
// colorinfo scale min
//   0
// colorinfo scale max
//   1


// return a list of the top-level categories
int tcl_colorinfo_categories(Tcl_Interp *interp, VMDApp *app, 
                             int argc, const char *[]) {
  if (argc != 0) {
    Tcl_SetResult(interp, (char *) "colorinfo: categories takes no parameters", TCL_STATIC);
    return TCL_ERROR;
  }
  
  int num = app->num_color_categories(); 
  for (int i=0; i<num; i++) {
    Tcl_AppendElement(interp, (char *) app->color_category(i));
  }
  return TCL_OK;
}


// return either a list elements in the category or, given an element
// in the category, return the color associated with it
int tcl_colorinfo_category(Tcl_Interp *interp, VMDApp *app, 
                           int argc, const char *argv[])
{
  if (argc != 1 && argc !=2) {
    Tcl_SetResult(interp, (char *) "colorinfo: category takes one parameter (for a list) or two for a mapping", TCL_STATIC);
    return TCL_ERROR;
  }

  // One of two possitilities ....
  if (argc == 1) { // ... list the categories
    for (int i=0; i<app->num_color_category_items(argv[0]); i++) {
      Tcl_AppendElement(interp, (char *) app->color_category_item(argv[0], i));
    }
    return TCL_OK;
  }
  //  ....  or return a mapping
  const char *mapping = app->color_mapping(argv[0], argv[1]);
  if (!mapping) {
    Tcl_SetResult(interp, (char *) "colorinfo: category: couldn't find category element", TCL_STATIC);
    return TCL_ERROR;
  }
  // return the color mapping
  // XXX why is this AppendElement?    
  Tcl_AppendElement(interp, (char *) mapping);
  return TCL_OK;
}

int tcl_colorinfo_num(Tcl_Interp *interp, VMDApp *app, int argc, const char *[]) {
  if (argc != 0) {
    Tcl_SetResult(interp, (char *) "colorinfo: numcolors takes no parameters", TCL_STATIC);
    return TCL_ERROR;
  }

  char tmpstring[64];
  sprintf(tmpstring, "%d", app->num_regular_colors());
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// ALL of the colors
int tcl_colorinfo_max(Tcl_Interp *interp, VMDApp *app, int argc, const char *[]) {
  if (argc != 0) {
    Tcl_SetResult(interp, (char *) "colorinfo: maxcolor takes no parameters", TCL_STATIC);
    return TCL_ERROR;
  }

  char tmpstring[64];
  sprintf(tmpstring, "%d", app->num_colors());
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// return a list of the regular colors
int tcl_colorinfo_colors(Tcl_Interp *interp, VMDApp *app, 
                         int argc, const char *[]) {
  if (argc != 0) {
    Tcl_SetResult(interp, (char *) "colorinfo: colors takes no parameters", TCL_STATIC);
    return TCL_ERROR;
  }
  for (int i=0; i<app->num_regular_colors(); i++) {
    Tcl_AppendElement(interp, (char *) app->color_name(i));
  }
  return TCL_OK;
}

//////////////// get the RGB value of the given color
int tcl_colorinfo_rgb(Tcl_Interp *interp, VMDApp *app, 
                      int argc, const char *argv[]) {
  if (argc != 1) {
    Tcl_SetResult(interp, (char *) "colorinfo: color takes one color name or index", TCL_STATIC);
    return TCL_ERROR;
  }
  float value[3];
  if (!app->color_value(argv[0], value, value+1, value+2)) {
    // Try to convert it to an int
    int id;
    if (Tcl_GetInt(interp, argv[0], &id) != TCL_OK ||
        !app->color_name(id) || 
        !app->color_value(app->color_name(id), value, value+1, value+2)) {
      Tcl_SetResult(interp, (char *) "colorinfo: color: couldn't find color name or index", TCL_STATIC);
    return TCL_ERROR;
    }
  }
  Tcl_Obj *tcl_result = Tcl_NewListObj(0,NULL);
  for (int i=0; i<3; i++)
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(value[i]));
  Tcl_SetObjResult(interp, tcl_result);
  return TCL_OK;
}

////////////////////// get the index value of the given color
int tcl_colorinfo_index(Tcl_Interp *interp, VMDApp *app, 
                        int argc, const char *argv[]) {
  if (argc != 1) {
    Tcl_SetResult(interp, (char *) "colorinfo: index takes one color name or index", TCL_STATIC);
    return TCL_ERROR;
  }
  
  int id = app->color_index(argv[0]); 
  if (id < 0) {
    Tcl_SetResult(interp, (char *) "colorinfo: index: couldn't find color name or index", TCL_STATIC);
    return TCL_ERROR;
  }

  char tmpstring[64];
  sprintf(tmpstring, "%d", id);
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}
////////////////////////////// info about the color scale
int tcl_colorinfo_scale(Tcl_Interp *interp, VMDApp *app, 
                        int argc, const char *argv[])
{
  if (argc != 1) {
    Tcl_SetResult(interp, (char *) "colorinfo: scale takes method|methods|midpoint|min|max", TCL_STATIC);
    return TCL_ERROR;
  }
  if (!strcmp(argv[0], "method")) {
    Tcl_SetResult(interp, (char *) app->colorscale_method_name(app->colorscale_method_current()), TCL_VOLATILE);
    return TCL_OK;
  }
  if (!strcmp(argv[0], "methods")) {
    for (int i=0; i<app->num_colorscale_methods(); i++) {
      Tcl_AppendElement(interp, (char *)app->colorscale_method_name(i));
    }
    return TCL_OK;
  }
  float mid, min, max;
  app->colorscale_info(&mid, &min, &max);
  if (!strcmp(argv[0], "midpoint")) {
    Tcl_SetObjResult(interp, Tcl_NewDoubleObj(mid));
    return TCL_OK;
  }
  if (!strcmp(argv[0], "min")) {
    Tcl_SetObjResult(interp, Tcl_NewDoubleObj(min));
    return TCL_OK;
  }
  if (!strcmp(argv[0], "max")) {
    Tcl_SetObjResult(interp, Tcl_NewDoubleObj(max));
    return TCL_OK;
  }
  Tcl_AppendResult(interp, "colorinfo: scale called with incorrect ",
		   "parameter '", argv[0], "'", NULL);
  return TCL_ERROR;
}

////////////////////////////////////////////////////////////////////////
int tcl_colorinfo(ClientData cd, Tcl_Interp *interp, int argc, const char *argv[])
{
  if (argc < 2) {
    Tcl_SetResult(interp, 
      (char *)
      "colorinfo categories\n"
      "colorinfo category <category>\n"
      "colorinfo category <category> <element>\n"
      "colorinfo [num|max|colors]\n"
      "colorinfo [index|rgb] <name|value>\n"
      "colorinfo scale [method|methods|midpoint|min|max]",
      TCL_STATIC);
    return TCL_ERROR;
  }

  VMDApp *app = (VMDApp *)cd;

  if (!strcmp(argv[1], "categories")) {
    return tcl_colorinfo_categories(interp, app, argc-2, argv+2);
  }
  if (!strcmp(argv[1], "category")) {
    return tcl_colorinfo_category(interp, app, argc-2, argv+2);
  }
  if (!strcmp(argv[1], "num")) {
    return tcl_colorinfo_num(interp, app, argc-2, argv+2);
  }
  if (!strcmp(argv[1], "max")) {
    return tcl_colorinfo_max(interp, app, argc-2, argv+2);
  }
  if (!strcmp(argv[1], "colors")) {
    return tcl_colorinfo_colors(interp, app, argc-2, argv+2);
  }
  if (!strcmp(argv[1], "index")) {
    return tcl_colorinfo_index(interp, app, argc-2, argv+2);
  }
  if (!strcmp(argv[1], "rgb")) {
    return tcl_colorinfo_rgb(interp, app, argc-2, argv+2);
  }
  // color scale info
  if (!strcmp(argv[1], "scale")) {
    return tcl_colorinfo_scale(interp, app, argc-2, argv+2);
  }

  Tcl_AppendResult(interp, "colorinfo: couldn't understand first parameter: ",
		   argv[1], NULL);
  return TCL_ERROR;
}

