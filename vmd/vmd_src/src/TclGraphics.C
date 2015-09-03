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
 *	$RCSfile: TclGraphics.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.47 $	$Date: 2011/02/25 19:16:30 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Parse the text input into a new graphics element.
 *
 ***************************************************************************/


#include <stdlib.h> 
#include "VMDApp.h"
#include "Molecule.h"
#include "MoleculeList.h"
#include "MaterialList.h"
#include "tcl.h"
#include "TclCommands.h" // for my own, external function definitions
#include "Scene.h"
#include "MoleculeGraphics.h"

// option 's' needs at least 'n' parameters
#define AT_LEAST(n, s)                                            \
{                                                                 \
  if (argc < n) {                                                 \
    Tcl_SetResult(interp,                                         \
              (char *) "graphics: " s ": not enough parameters",  \
              TCL_STATIC);                                        \
    return TCL_ERROR;                                             \
  }                                                               \
}

// option 's' takes exactly 'n' parameters
#define MUST_HAVE(n, s)                                           \
{                                                                 \
  if (argc != n) {                                                \
    Tcl_SetResult(interp,                                         \
      (char *) "graphics: " s ": incorrect number of parameters", \
      TCL_STATIC);                                                \
    return TCL_ERROR;                                             \
  }                                                               \
}

// evaluate data for a triangle
static int tcl_graphics_triangle(MoleculeGraphics *gmol, 
				 int argc, const char *argv[],
				 Tcl_Interp *interp)
{
  // the first three are {x, y, z} coordinates
  MUST_HAVE(3, "triangle");
  float vals[9];
  if (tcl_get_vector(argv[0], vals+0, interp) != TCL_OK ||
      tcl_get_vector(argv[1], vals+3, interp) != TCL_OK ||
      tcl_get_vector(argv[2], vals+6, interp) != TCL_OK   ) {
    return TCL_ERROR;
  }

  // I have a triangle, so add it
  char tmpstring[64];
  sprintf(tmpstring, "%d", gmol->add_triangle(vals+0, vals+3, vals+6));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// a triangle with the normals specified
static int tcl_graphics_trinorm(MoleculeGraphics *gmol, 
				int argc, const char *argv[],
				Tcl_Interp *interp)
{
  // the first three are {x, y, z} coordinates
  // the next three are {x, y, z} normals
  MUST_HAVE(6, "trinorm");
  float vals[19];
  if (tcl_get_vector(argv[0], vals+0,  interp) != TCL_OK ||
      tcl_get_vector(argv[1], vals+3,  interp) != TCL_OK ||
      tcl_get_vector(argv[2], vals+6,  interp) != TCL_OK ||
      tcl_get_vector(argv[3], vals+9,  interp) != TCL_OK ||
      tcl_get_vector(argv[4], vals+12, interp) != TCL_OK ||
      tcl_get_vector(argv[5], vals+15, interp) != TCL_OK   ) {
    return TCL_ERROR;
  }

  // I have a triangle, so add it
  char tmpstring[64];
  sprintf(tmpstring, "%d",
	  gmol->add_trinorm(vals+0, vals+3, vals+6, 
			    vals+9, vals+12, vals+15));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// a triangle with the normals and colors specified
static int tcl_graphics_tricolor(MoleculeGraphics *gmol, 
				int argc, const char *argv[],
				Tcl_Interp *interp)
{
  // the first three are {x, y, z} coordinates
  // the next three are {x, y, z} normals
  MUST_HAVE(9, "tricolor");
  float vals[19];
  if (tcl_get_vector(argv[0], vals+0,  interp) != TCL_OK ||
      tcl_get_vector(argv[1], vals+3,  interp) != TCL_OK ||
      tcl_get_vector(argv[2], vals+6,  interp) != TCL_OK ||
      tcl_get_vector(argv[3], vals+9,  interp) != TCL_OK ||
      tcl_get_vector(argv[4], vals+12, interp) != TCL_OK ||
      tcl_get_vector(argv[5], vals+15, interp) != TCL_OK   ) {
    return TCL_ERROR;
  }
  int c1, c2, c3;
  if (Tcl_GetInt(interp, argv[6], &c1) != TCL_OK ||
      Tcl_GetInt(interp, argv[7], &c2) != TCL_OK ||
      Tcl_GetInt(interp, argv[8], &c3) != TCL_OK) {
    return TCL_ERROR;
  }

  // I have a triangle, so add it
  char tmpstring[64];
  sprintf(tmpstring, "%d",
	  gmol->add_tricolor(vals+0, vals+3, vals+6, 
			    vals+9, vals+12, vals+15, c1, c2, c3));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// a cylinder has endpoints, radius, and resolution
static int tcl_graphics_cylinder(MoleculeGraphics *gmol, 
				 int argc, const char *argv[],
				 Tcl_Interp *interp)
{
  // the first two are {x, y, z} coordinates
  AT_LEAST(2, "cylinder");
  float vals[6];
  if (tcl_get_vector(argv[0], vals+0,  interp) != TCL_OK ||
      tcl_get_vector(argv[1], vals+3,  interp) != TCL_OK) {
    return TCL_ERROR;
  }
  
  // get the optional values
  double radius = 1.0;
  int resolution = 6;
  int filled = 0;
  argc -= 2;
  argv += 2;
  if (argc %2) {
    Tcl_SetResult(interp, (char *) "graphics: cylinder has wrong number of options", TCL_STATIC);
    return TCL_ERROR;
  }
  while (argc) {
    if (!strcmp(argv[0], "radius")) {
      if (Tcl_GetDouble(interp, argv[1], &radius) != TCL_OK) {
	return TCL_ERROR;
      }
      if (radius <0) radius = 0;
      argc -= 2;
      argv += 2;
      continue;
    }
    if (!strcmp(argv[0], "resolution")) {
      if (Tcl_GetInt(interp, argv[1], &resolution) != TCL_OK) {
	return TCL_ERROR;
      }
      if (resolution < 0) resolution = 0;
      if (resolution > 30) resolution = 30;
      argc -= 2;
      argv += 2;
      continue;
    }
    if (!strcmp(argv[0], "filled")) {
      if (Tcl_GetBoolean(interp, argv[1], &filled) != TCL_OK) {
	return TCL_ERROR;
      }
      argc -= 2;
      argv += 2;
      continue;
    }
    // reaching here is an error
    Tcl_AppendResult(interp, "graphics: unknown option for cylinder: ",
		     argv[0], NULL);
    return TCL_ERROR;
  }
  
  // I have a cylinder, so add it
  char tmpstring[64];
  sprintf(tmpstring, "%d",
	  gmol->add_cylinder(vals+0, vals+3, (float) radius, resolution, filled));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// only has coordinates
static int tcl_graphics_point(MoleculeGraphics *gmol, 
			      int argc, const char *argv[],
			      Tcl_Interp *interp)
{
  MUST_HAVE(1, "point");
  float vals[3];
  if (tcl_get_vector(argv[0], vals+0, interp) != TCL_OK) {
    return TCL_ERROR;
  }

  // we've got a point, so add it
  char tmpstring[64];
  sprintf(tmpstring, "%d", gmol->add_point(vals+0));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// only has coordinates
static int tcl_graphics_pickpoint(MoleculeGraphics *gmol, 
			      int argc, const char *argv[],
			      Tcl_Interp *interp)
{
  MUST_HAVE(1, "pickpoint");
  float vals[3];
  if (tcl_get_vector(argv[0], vals+0, interp) != TCL_OK) {
    return TCL_ERROR;
  }

  // we've got a point, so add it
  char tmpstring[64];
  sprintf(tmpstring, "%d", gmol->add_pickpoint(vals+0));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}


// has begin and end points, a "style", and a width
static int tcl_graphics_line(MoleculeGraphics *gmol, 
			     int argc, const char *argv[],
			     Tcl_Interp *interp)
{
  // just need the start and end values
  AT_LEAST(2, "line");
  float vals[6];
  if (tcl_get_vector(argv[0], vals+0, interp) != TCL_OK ||
      tcl_get_vector(argv[1], vals+3, interp) != TCL_OK) {
    return TCL_ERROR;
  }
  // options:
  //  'style' is "solid" or "dashed"
  //  'width' is 0 .. 255;
  int line_style = ::SOLIDLINE;
  int width = 1;
  argc -= 2;
  argv += 2;
  if (argc %2) {
    Tcl_SetResult(interp, (char *) "graphics: line has wrong number of options", TCL_STATIC);
    return TCL_ERROR;
  }
  while (argc) {
    if (!strcmp(argv[0], "style")) {
      if (!strcmp(argv[1], "solid")) {
	line_style = ::SOLIDLINE;
      } else if (!strcmp(argv[1], "dashed")) {
	line_style = ::DASHEDLINE;
      } else {
	Tcl_AppendResult(interp, "graphics: don't understand the line style ",
			 argv[1], NULL);
	return TCL_ERROR;
      }
    } else if (!strcmp(argv[0], "width")) {
      if (Tcl_GetInt(interp, argv[1], &width) != TCL_OK) {
	return TCL_ERROR;
      }
      if (width > 255) width = 255;
      if (width < 0) width = 0;
    } else {
      Tcl_AppendResult(interp, "graphics: don't understand the line option ",
		       argv[0], NULL);
      return TCL_ERROR;
    }
    argc -= 2;
    argv += 2;
  }

  // otherwise, just draw the line
  char tmpstring[64];
  sprintf(tmpstring, "%d", gmol->add_line(vals+0, vals+3, line_style, width));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// turn them on or off
static int tcl_graphics_materials(MoleculeGraphics *gmol,
				  int argc, const char *argv[],
				  Tcl_Interp *interp)
{
  MUST_HAVE(1, "materials");
  int val;
  if (Tcl_GetBoolean(interp, argv[0], &val) != TCL_OK) {
    return TCL_ERROR;
  }
  
  // enable/disable materials
  char tmpstring[64];
  sprintf(tmpstring, "%d", gmol->use_materials(val));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// set material for this graphics molecule
static int tcl_graphics_material(MoleculeGraphics *gmol,
				  int argc, const char *argv[],
				  Tcl_Interp *interp, MaterialList *mlist)
{
  MUST_HAVE(1, "material");
  int val = mlist->material_index(argv[0]);
  if (val < 0) {
    char tmpstring[1024];
    sprintf(tmpstring, "graphics: invalid material: %s", argv[0]);
    Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    return TCL_ERROR;
  }
  const Material *mat = mlist->material(val); 
  char tmpstring[64];
  sprintf(tmpstring, "%d", gmol->use_material(mat));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
} 

// takes:
//  a color index
//  a color name (like "red")
static int tcl_graphics_color(VMDApp *app, MoleculeGraphics *gmol,
			      int argc, const char *argv[],
			      Tcl_Interp *interp)
{
  MUST_HAVE(1, "color");
  // is it a valid number?
  int id;
  Tcl_ResetResult(interp);
  if (Tcl_GetInt(interp, argv[0], &id) == TCL_OK) {
    // is it valid?
    if (id >=0 && id < MAXCOLORS) {
      char tmpstring[64];
      sprintf(tmpstring, "%d", gmol->use_color(id));
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_OK;
    } else {
      char tmpstring[64];
      sprintf(tmpstring, "graphics: color index value '%d' out of range", id);
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_ERROR;
    }
  }
  // is it a color name?
  Tcl_ResetResult(interp);
  id = app->color_index(argv[0]);
  if (id >= 0) { 
    char tmpstring[64];
    sprintf(tmpstring, "%d", gmol->use_color(id));
    Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    return TCL_OK;
  }
  // Otherwise there is a problem
  Tcl_AppendResult(interp, "graphics: unknown color: ", argv[0], NULL);
  return TCL_ERROR;
}

// cone has base and tip coordinates, width at the base, width at the tip and resolution
static int tcl_graphics_cone(MoleculeGraphics *gmol,
			     int argc, const char *argv[],
			     Tcl_Interp *interp)
{
  // the first two are {x, y, z}
  AT_LEAST(2, "cone");
  float vals[6];
  if (tcl_get_vector(argv[0], vals+0,  interp) != TCL_OK ||
      tcl_get_vector(argv[1], vals+3,  interp) != TCL_OK) {
    return TCL_ERROR;
  }

  // get the optional values
  double radius = 1.0;
  double radius2 = 0.0;
  int resolution = 6;
  argc -= 2;
  argv += 2;
  if (argc %2) {
    Tcl_SetResult(interp, (char *) "graphics: cone has wrong number of options", TCL_STATIC);
    return TCL_ERROR;
  }
  while (argc) {
    if (!strcmp(argv[0], "radius2")) {
      if (Tcl_GetDouble(interp, argv[1], &radius2) != TCL_OK) {
	return TCL_ERROR;
      }
      if (radius2 <0) radius2 = 0;
      argc -= 2;
      argv += 2;
      continue;
    }
    if (!strcmp(argv[0], "radius")) {
      if (Tcl_GetDouble(interp, argv[1], &radius) != TCL_OK) {
	return TCL_ERROR;
      }
      if (radius <0) radius = 0;
      argc -= 2;
      argv += 2;
      continue;
    }
    if (!strcmp(argv[0], "resolution")) {
      if (Tcl_GetInt(interp, argv[1], &resolution) != TCL_OK) {
	return TCL_ERROR;
      }
      if (resolution < 0) resolution = 0;
      if (resolution > 300) resolution = 300;
      argc -= 2;
      argv += 2;
      continue;
    }

    // reaching here is an error
    Tcl_AppendResult(interp, "graphics: unknown option for cone: ",
		     argv[0], NULL);
    return TCL_ERROR;
  }

  // I have a cone, so add it
  char tmpstring[64];
  sprintf(tmpstring, "%d",
	  gmol->add_cone(vals+0, vals+3, (float) radius, (float) radius2, resolution));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}


// sphere has a center, radius, and resolution
static int tcl_graphics_sphere(MoleculeGraphics *gmol,
			       int argc, const char *argv[],
			       Tcl_Interp *interp)
{
  // only really need the coordinates
  AT_LEAST(1, "sphere");
  float vals[3];
  if (tcl_get_vector(argv[0], vals+0,  interp) != TCL_OK) {
    return TCL_ERROR;
  }

  // get the optional values
  double radius = 1.0;
  int resolution = 6;
  argc -= 1;
  argv += 1;
  if (argc %2) {
    Tcl_SetResult(interp, (char *) "graphics: sphere has wrong number of options", TCL_STATIC);
    return TCL_ERROR;
  }
  while (argc) {
    if (!strcmp(argv[0], "radius")) {
      if (Tcl_GetDouble(interp, argv[1], &radius) != TCL_OK) {
	return TCL_ERROR;
      }
      if (radius <0) radius = 0;
      argc -= 2;
      argv += 2;
      continue;
    }
    if (!strcmp(argv[0], "resolution")) {
      if (Tcl_GetInt(interp, argv[1], &resolution) != TCL_OK) {
	return TCL_ERROR;
      }
      if (resolution < 0) resolution = 0;
      if (resolution > 30) resolution = 30;
      argc -= 2;
      argv += 2;
      continue;
    }
    // reaching here is an error
    Tcl_AppendResult(interp, "graphics: unknown option for sphere: ",
		     argv[0], NULL);
    return TCL_ERROR;
  }

  // I have a sphere, so add it
  char tmpstring[64];
  sprintf(tmpstring, "%d",
	  gmol->add_sphere(vals+0, (float) radius, resolution));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// text has a start point and a string to display
static int tcl_graphics_text(MoleculeGraphics *gmol, int argc, const char *argv[],
			     Tcl_Interp *interp) {
  // have a vector and some text
  AT_LEAST(2, "text");
  float vals[3];
  if (tcl_get_vector(argv[0], vals+0,  interp) != TCL_OK) {
    return TCL_ERROR;
  }

  // get the optional size values
  const char* string = argv[1];
  double size = 1.0;
  double thickness = 1.0;
  argc -= 2;
  argv += 2;

  if (argc %2) {
    Tcl_SetResult(interp, (char *) "graphics: text has wrong number of options", TCL_STATIC);
    return TCL_ERROR;
  }

  while (argc) {
    if (!strcmp(argv[0], "size")) {
      if (Tcl_GetDouble(interp, argv[1], &size) != TCL_OK) {
        return TCL_ERROR;
      }
      if (size <0) size = 0;
      argc -= 2;
      argv += 2;
      continue;
    }

    if (!strcmp(argv[0], "thickness")) {
      if (Tcl_GetDouble(interp, argv[1], &thickness) != TCL_OK) {
        return TCL_ERROR;
      }
      if (thickness <0) thickness = 0;
      argc -= 2;
      argv += 2;
      continue;
    }

    // reaching here is an error
    Tcl_AppendResult(interp, "graphics: unknown option for text: ",
                     argv[0], NULL);
    return TCL_ERROR;
  }

  // add the text
  char tmpstring[64];
  sprintf(tmpstring, "%d", gmol->add_text(vals+0, string, (float) size, (float) thickness));
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}


//////////////////////
// delete the given id
static int tcl_graphics_delete(MoleculeGraphics *gmol, int argc, const char *argv[],
			       Tcl_Interp *interp) {
  if (argc != 1) {
    Tcl_SetResult(interp, (char *) "graphics: delete takes one parameter (either an index or 'all')", TCL_STATIC);
    return TCL_ERROR;
  }
  if (!strcmp(argv[0], "all")) {
    gmol->delete_all();
    return TCL_OK;
  }
  int id;
  if (Tcl_GetInt(interp, argv[0], &id) != TCL_OK) {
    return TCL_ERROR;
  }
  gmol->delete_id(id);
  return TCL_OK;
}

// delete the given id and have the next element replace this one
static int tcl_graphics_replace(MoleculeGraphics *gmol, int argc, const char *argv[],
				Tcl_Interp *interp) {
  if (argc != 1) {
    Tcl_SetResult(interp, (char *) "graphics: replace takes one parameter, the index", TCL_STATIC);
    return TCL_ERROR;
  }
  int id;
  if (Tcl_GetInt(interp, argv[0], &id) != TCL_OK) {
    return TCL_ERROR;
  }
  gmol->replace_id(id);
  return TCL_OK;
}

// does a given id exist?
static int tcl_graphics_exists(MoleculeGraphics *gmol, int argc, const char *argv[],
			       Tcl_Interp *interp) {
  if (argc != 1) {
    Tcl_SetResult(interp, (char *) "graphics: exists takes one parameter, the index", TCL_STATIC);
    return TCL_ERROR;
  }
  int id;
  if (Tcl_GetInt(interp, argv[0], &id) != TCL_OK) {
    return TCL_ERROR;
  }

  char tmpstring[64]; 
  sprintf(tmpstring, "%d", gmol->index_id(id) != -1);
  Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
  return TCL_OK;
}

// return info about the graphics with a given id
static int tcl_graphics_info(MoleculeGraphics *gmol,
			     int argc, const char *argv[],
			     Tcl_Interp *interp)
{
  if (argc != 1) {
    Tcl_SetResult(interp, (char *) "graphics: info takes one parameter, the index", TCL_STATIC);
    return TCL_ERROR;
  }
  int id;
  if (Tcl_GetInt(interp, argv[0], &id) != TCL_OK) {
    return TCL_ERROR;
  }
  // since either NULL or a static char * is returned, this will work
  Tcl_AppendResult(interp, gmol->info_id(id), NULL);
  return TCL_OK;
}


// already parsed the "graphics" and "number" terms, what remains are
//  add, delete, info, and list
static int tcl_graphics(VMDApp *app, int molid, int argc, const char *argv[], 
                        Tcl_Interp *interp)
{
  // Is this a graphics molecule?
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (mol == NULL) {
    Tcl_SetResult(interp, (char *) "graphics: invalid graphics molecule", TCL_STATIC);
    return TCL_ERROR;
  }
  if (argc < 1) {
    Tcl_SetResult(interp, (char *) "graphics: not enough parameters", TCL_STATIC);
    return TCL_ERROR;
  }
  MoleculeGraphics *gmol = mol->moleculeGraphics();
  // what am I to do?
  if (!strcmp(argv[0], "list")) {
    if (argc != 1) {
      Tcl_SetResult(interp, (char *) "graphics: list takes no parameters", TCL_STATIC);
      return TCL_ERROR;
    }
    int num = gmol->num_elements();
    for (int i=0; i<num; i++) {
      int id = gmol->element_id(i);
      if (id >=0) {
        char s[10];
        sprintf(s, "%d", id);
        Tcl_AppendElement(interp, s);
      }
    }
    return TCL_OK;
  }
  // all the rest take more than one parameter
  if (argc < 2) {
    Tcl_SetResult(interp, (char *) "graphics: not enough parameters", TCL_STATIC);
    return TCL_ERROR;
  }
  if (!strcmp(argv[0], "triangle")) {
    return tcl_graphics_triangle(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "trinorm")) {
    return tcl_graphics_trinorm(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "tricolor")) {
    return tcl_graphics_tricolor(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "point")) {
    return tcl_graphics_point(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "pickpoint")) {
    return tcl_graphics_pickpoint(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "line")) {
    return tcl_graphics_line(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "cylinder")) {
    return tcl_graphics_cylinder(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "cone")) {
    return tcl_graphics_cone(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "sphere")) {
    return tcl_graphics_sphere(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "text")) {
    return tcl_graphics_text(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "materials")) {
    return tcl_graphics_materials(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "material")) {
    return tcl_graphics_material(gmol, argc-1, argv+1, interp, 
      app->materialList);
  }
  if (!strcmp(argv[0], "color") || !strcmp(argv[1], "colour")) {
    return tcl_graphics_color(app, gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "delete")) {
    return tcl_graphics_delete(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "exists")) {
   return tcl_graphics_exists(gmol, argc-1, argv+1, interp);
  } 
  if (!strcmp(argv[0], "replace")) {
   return tcl_graphics_replace(gmol, argc-1, argv+1, interp);
  }
  if (!strcmp(argv[0], "info")) {
   return tcl_graphics_info(gmol, argc-1, argv+1, interp);
  }
  Tcl_AppendResult(interp, "graphics: don't understand the command: ",
		   argv[0], NULL);
  return TCL_ERROR;
}


// interface to the 3d graphics
int graphics_tcl(ClientData cd, Tcl_Interp *interp, int argc, const char *argv[]) {
  // need at least two arguments
  if (argc < 2) {
    Tcl_SetResult(interp, (char *) "graphics: not enough parameters", TCL_STATIC);
    return TCL_ERROR;
  }

  VMDApp *app = (VMDApp *)cd;
  // get the molid 
  int mol = -1;
  if (!strcmp(argv[1], "top")) {
    if (app->moleculeList->top()) {
      mol = app->moleculeList->top()->id();
    }
  } else {
    if (Tcl_GetInt(interp, argv[1], &mol) != TCL_OK) {
      return TCL_ERROR;
    }
  }
  // and call the "real" function
  return tcl_graphics(app, mol, argc-2, argv+2, interp);
}


