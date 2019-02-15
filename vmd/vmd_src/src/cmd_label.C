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
 *      $RCSfile: cmd_label.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.49 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for label control
 ***************************************************************************/

#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <tcl.h>
#include "config.h"
#include "GeometryList.h"
#include "GeometryMol.h"
#include "utilities.h"
#include "VMDApp.h"

static int find_atom_from_name(Tcl_Interp *interp, const char *str, int *molid,
                               int *atomid) {
  // string must be of format ##/## where ## are numbers.

  // find first occurence of '/'
  const char *slash = strchr(str, '/');
  if (!slash) {
    Tcl_AppendResult(interp, "Illegal format: ", str, NULL); 
    return -1;
  }

  // make sure we have digits
  if (slash == str) {
    Tcl_AppendResult(interp, "Missing molecule specification: ", str, NULL); 
    return -1;
  }
  if (strlen(slash+1) == 0) {
    Tcl_AppendResult(interp, "Missing atom specification: ", str, NULL); 
    return -1;
  }
  const char *s;
  for (s = str; s < slash; s++) {
    if (!isdigit(*s)) {
      Tcl_AppendResult(interp, "Illegal molecule specification: ", str, NULL); 
      return -1;
    }
  }
  for (s = slash+1; *s; s++) {
    if (!isdigit(*s)) {
      Tcl_AppendResult(interp, "Illegal atom specification: ", str, NULL); 
      return -1;
    }
  }

  // Looks ok; extract the molecule and atom
  char *buf = new char[slash-str + 1];
  strncpy(buf, str, slash-str);
  buf[slash-str] = '\0';
  *molid = atoi(buf);
  *atomid = atoi(slash+1);
  return 0;
}


int text_cmd_label(ClientData cd, Tcl_Interp *interp, int argc,
                     const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if (argc < 2) {
    Tcl_SetResult(interp,
      (char *)
      "label add [Atoms|Bonds|Angles|Dihedrals] {atoms as <molid>/<atomid>}\n"
      "label addspring <molid> <atomid> <atomid> <k>\n"
      "label list              -- return label categories\n"
      "label list <category>   -- return id's of labels in given category\n"
      "label [show|hide|delete] <category> [index] -- \n\tControl specific label or all labels in category\n"
      "label graph <category> <index> -- Return a list of values for the given label\n\tfor all animation frames\n"
      "label textsize [<newsize>]\n"
      "label textthickness [<newthick>]\n" ,
      TCL_STATIC);
    return TCL_ERROR;
  }
  if(!strupncmp(argv[1], "add", CMDLEN)) {
    if(argc > 3) {
      int n = argc-3;
      const char **items = argv+3; 
      int *molid= new int[n];
      int *atmid = new int[n];
      int i;
      for(i=0; i < n; i++) {
        if (find_atom_from_name(interp, items[i], molid+i, atmid+i))
          break;
      }
      int rc = -1;
      if(i == n) {  // all successfully parsed
        rc = app->label_add(argv[2], argc-3, molid, atmid, NULL, 0.0f, 1);
      }
      delete [] molid;
      delete [] atmid;
      if (rc < 0) {
        Tcl_AppendResult(interp, "\nUnable to add label.", NULL);
        return TCL_ERROR;
      }
    }
    else
      return TCL_ERROR;

  }
  else if(!strupncmp(argv[1],"addspring",CMDLEN)) { /* add a spring */
    if(argc != 6) {
      Tcl_AppendResult(interp, "usage: label addspring <molid> <atomid> <atomid> <k>", NULL);
      return TCL_ERROR;
    }
    int molid[2];
    int atomid[2];
    float k;
    sscanf(argv[2],"%d",molid); /* convert all of the args to numbers */
    sscanf(argv[3],"%d",atomid);
    sscanf(argv[4],"%d",atomid+1);
    sscanf(argv[5],"%f",&k);
    molid[1]=molid[0];
    if (app->label_add("Springs", 2, molid, atomid, NULL, k, 1) < 0) {
      Tcl_AppendResult(interp, "Unable to add spring.", NULL);
      return TCL_ERROR;
    }
  }
  else if(!strupncmp(argv[1], "list", CMDLEN)) {
    if(argc == 3) {
      int cat =  app->geometryList->geom_list_index(argv[2]);
      if (cat < 0) {
	Tcl_AppendResult(interp, "graph list category '", argv[2], 
			 "' was not found", NULL);
	return TCL_ERROR;
      }
      // go through the list by hand
      GeomListPtr glist = app->geometryList->geom_list(cat);
      int gnum = glist->num();
      char s[30];
      GeometryMol *g;
      for (int i=0; i<gnum; i++) {
	g = (*glist)[i];
	Tcl_AppendResult(interp, i==0 ? "" : " ",   "{", NULL);
	for (int j=0; j<g->items(); j++) {
	  // append the molecule id/atom index
	  sprintf(s, "%d %d", g -> obj_index(j), g -> com_index(j));
	  Tcl_AppendElement(interp, s);
	}
	// and the value and the status
	sprintf(s, "%f", g->ok() ? g->calculate() : 0.0);
	Tcl_AppendElement(interp, s);
	Tcl_AppendElement(interp, g -> displayed() ?  "show" : "hide");
	Tcl_AppendResult(interp, "}", NULL);
      }
      return TCL_OK;
    }
    else if (argc == 2) {
      // return the main categories
      for (int i=0; i<app->geometryList -> num_lists(); i++) {
	Tcl_AppendElement(interp,  app->geometryList -> geom_list_name(i));
      }
      return TCL_OK;
    } else
      return TCL_ERROR;

  } else if(!strupncmp(argv[1], "show", CMDLEN) ||
	    !strupncmp(argv[1], "hide", CMDLEN)) {
    int item;
    if(argc == 3 || (argc == 4 && !strupncmp(argv[3], "all", CMDLEN)))
      item = (-1);
    else if(argc == 4) {
      if (Tcl_GetInt(interp, argv[3], &item) != TCL_OK) {
          Tcl_AppendResult(interp, " in label ", argv[1],  NULL);
          return TCL_ERROR;
      }
    } else
      return TCL_ERROR;
    
    app->label_show(argv[2], item, !strupncmp(argv[1], "show", CMDLEN));
    // XXX check return code

  } else if(!strupncmp(argv[1], "delete", CMDLEN)) {
    int item;
    if(argc == 3 || (argc == 4 && !strupncmp(argv[3], "all", CMDLEN))) {
      item = (-1);
    } else if(argc == 4) {
      if (Tcl_GetInt(interp, argv[3], &item) != TCL_OK) {
	      Tcl_AppendResult(interp, " in label ", argv[1],  NULL);
	      return TCL_ERROR;
      }
    } else {
      return TCL_ERROR;
    } 

    app->label_delete(argv[2], item);
    // XXX check return code

  } else if(!strupncmp(argv[1], "graph", CMDLEN) && argc > 3) {
    int item;
    if (Tcl_GetInt(interp, argv[3], &item) != TCL_OK) {
      return TCL_ERROR;
    };
    // find the geometry
    int cat =  app->geometryList->geom_list_index(argv[2]);
    if (cat < 0) {
      Tcl_AppendResult(interp, "Invalid geometry type: ", argv[2], NULL);
      return TCL_ERROR;
    }
    // get the correct geometry pointer
    GeomListPtr glist = app->geometryList -> geom_list(cat);
    int gnum = glist -> num();
    if (item < 0 || item >= gnum) {
      char buf[512];
      sprintf(buf, "label %s index %d out of range", argv[2], item);
      Tcl_SetResult(interp, buf, TCL_VOLATILE);
      return TCL_ERROR;
    }
    // compute all the values
    GeometryMol *g = (*glist)[item];
    if (!g->has_value()) {
      Tcl_AppendResult(interp, "Geometry type ", argv[2], " has no values to graph.", NULL);
      return TCL_ERROR;
    }
    ResizeArray<float> gValues(1024);
    if (!g->calculate_all(gValues)) {
      Tcl_AppendResult(interp, "label has no value", NULL);
      return TCL_ERROR;
    }
    if (argc > 4) {
      // save the values in the given filename
      const char *filename = argv[4];
      FILE *outfile = fopen(filename, "w");
      if (!outfile) {
        Tcl_AppendResult(interp, "Cannot write graph data to file ",
          filename, NULL);
        return TCL_ERROR;
      }
      for (int i=0; i<gValues.num(); i++) {
        fprintf(outfile, "%f  %f\n", float(i), gValues[i]);
      }
      fclose(outfile);
    } else {
      char s[20];
	    for (int count = 0; count < gValues.num(); count++) {
	      sprintf(s, "%f", gValues[count]);
	      Tcl_AppendElement(interp, s);
	    }
    }
  } else if (!strupncmp(argv[1], "textsize", CMDLEN)) {
    if (argc == 2) {
      // return the current size
      Tcl_SetObjResult(interp, Tcl_NewDoubleObj(app->label_get_text_size()));
      return TCL_OK;
    } else if (argc == 3) {
      // set new size
      double newsize = 1;
      if (Tcl_GetDouble(interp, argv[2], &newsize) != TCL_OK)
        return TCL_ERROR;
      if (!app->label_set_text_size((float) newsize)) {
        Tcl_AppendResult(interp, "label textsize: Unable to set size to ",
            argv[2], NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_SetResult(interp, (char *) "label textsize: wrong number of arguments",
          TCL_STATIC);
      return TCL_ERROR;
    }
  } else if (!strupncmp(argv[1], "textthickness", CMDLEN)) {
    if (argc == 2) {
      // return the current thickness
      Tcl_SetObjResult(interp, Tcl_NewDoubleObj(app->label_get_text_thickness()));
      return TCL_OK;
    } else if (argc == 3) {
      // set new size
      double newthick = 1;
      if (Tcl_GetDouble(interp, argv[2], &newthick) != TCL_OK)
        return TCL_ERROR;
      if (!app->label_set_text_thickness((float) newthick)) {
        Tcl_AppendResult(interp, "label textthickness: Unable to set thickness to ", argv[2], NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_SetResult(interp, (char *) "label textthickness: wrong number of arguments",
          TCL_STATIC);
      return TCL_ERROR;
    }
  } else if (!strupncmp(argv[1], "textoffset", CMDLEN)) {
    if (argc == 4) {
      // return current offset;
      const char *geomtype = argv[2];
      int geom;
      if (Tcl_GetInt(interp, argv[3], &geom) != TCL_OK) return TCL_ERROR;
      const float *offset = app->geometryList->getTextOffset(geomtype, geom);
      if (!offset) {
        Tcl_SetResult(interp, (char *) "label textoffset: Invalid geometry specified", TCL_STATIC);
        return TCL_ERROR;
      }
      Tcl_Obj *result = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, result, Tcl_NewDoubleObj(offset[0]));
      Tcl_ListObjAppendElement(interp, result, Tcl_NewDoubleObj(offset[1]));
      Tcl_SetObjResult(interp, result);
    } else if (argc == 5) {
      const char *geomtype = argv[2];
      int geom;
      if (Tcl_GetInt(interp, argv[3], &geom) != TCL_OK) return TCL_ERROR;
      float x, y;
      if (sscanf(argv[4], "%f %f", &x, &y) != 2) {
        Tcl_AppendResult(interp, "Could not understand argument to label textoffset:", argv[2], NULL);
        return TCL_ERROR;
      }
      if (!app->label_set_textoffset(geomtype, geom, x, y)) {
        Tcl_SetResult(interp, (char *) "label textoffset: Invalid geometry specified", TCL_STATIC);
        return TCL_ERROR;
      }
    } else {
      Tcl_SetResult(interp, (char *) "label textoffset: wrong number of arguments",
          TCL_STATIC);
      return TCL_ERROR;
    }
  } else if (!strupncmp(argv[1], "textformat", CMDLEN)) {
    if (argc == 4) {
      // return current format
      const char *geomtype = argv[2];
      int geom;
      if (Tcl_GetInt(interp, argv[3], &geom) != TCL_OK) return TCL_ERROR;
      const char *format = app->geometryList->getTextFormat(geomtype, geom);
      if (!format) {
        Tcl_SetResult(interp, (char *) "label textformat: Invalid geometry specified", TCL_STATIC);
        return TCL_ERROR;
      }
      Tcl_SetResult(interp, (char *)format, TCL_VOLATILE);
    } else if (argc == 5) {
      const char *geomtype = argv[2];
      int geom;
      if (Tcl_GetInt(interp, argv[3], &geom) != TCL_OK) return TCL_ERROR;
      if (!app->label_set_textformat(geomtype, geom, argv[4])) {
        Tcl_SetResult(interp, (char *) "label textformat failed.", TCL_STATIC);
        return TCL_ERROR;
      }
    } else {
      Tcl_SetResult(interp, (char *) "label textformat: wrong number of arguments",
                    TCL_STATIC);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}
