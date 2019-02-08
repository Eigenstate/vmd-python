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
 *      $RCSfile: cmd_color.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.40 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for color controls
 ***************************************************************************/

#include <tcl.h>
#include <stdlib.h>
#include "VMDApp.h"
#include "TclCommands.h"
#include "config.h"

int text_cmd_color(ClientData cd, Tcl_Interp *interp, int argc, const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if (argc < 2) {
    // might as well return some useful information 
    Tcl_SetResult(interp, 
      (char *)
      "color change rgb <color> [<grayscale> | <r g b>]\n"
      "   (when no value is specified, the color is reset to its default value)\n"
      "color scale [method|midpoint|min|max] <value>\n"
      "color scale colors <method> [<mincolor> <midcolor> <maxcolor>]\n"
      "color <category> <name> [new color]",
      TCL_STATIC);
    return TCL_ERROR;
  }

  if (!strupncmp(argv[1], "change", CMDLEN) && argc > 3) {
    if (!strupncmp(argv[2], "rgb", CMDLEN)) {
      float r = 0.5, g = 0.5, b = 0.5;
      if (app->color_index(argv[3]) < 0) { 
        Tcl_SetResult(interp,  (char *) "color change: invalid color specified", TCL_STATIC);
        return TCL_ERROR;
      }
      if (argc == 4) {
        // Get the default values for the color
        if (!app->color_default_value(argv[3], &r, &g, &b)) {
          Tcl_SetResult(interp, (char *) "Unable to get default values for color", TCL_STATIC);
          return TCL_ERROR;
        } 
        app->color_change_rgb(argv[3], r, g, b);
        return TCL_OK;
      } else {
        double rr;
        if (Tcl_GetDouble(interp, argv[4], &rr) != TCL_OK) {
          Tcl_AppendResult(interp,  " in color change", NULL);
          return TCL_ERROR;
        }
        r = (float) rr;
        if (argc == 5) {
          if (!app->color_change_rgb(argv[3], r, r, r)) {
            Tcl_SetResult(interp, (char *) "Unable to change color", TCL_STATIC);
            return TCL_ERROR;
          }
        } else if (argc == 7) {
          double gg, bb;
          if (Tcl_GetDouble(interp, argv[5], &gg) != TCL_OK ||
            Tcl_GetDouble(interp, argv[6], &bb) != TCL_OK) {
            Tcl_AppendResult(interp, " in color change", NULL);
            return TCL_ERROR;
          }
          g = (float) gg;
          b = (float) bb;
          if (!app->color_change_rgb(argv[3], r, g, b)) {
            Tcl_SetResult(interp, (char *) "Unable to change color", TCL_STATIC);
            return TCL_ERROR;
          }
        } else {
          Tcl_SetResult(interp, (char *) "color change needs 1 (or 3) parameters", TCL_STATIC);
          return TCL_ERROR;
        }
        return TCL_OK;
      }
    } else if (!strupncmp(argv[2], "rgblist", CMDLEN)) {
      const char **colcmds;
      int i, num_colcmds;
      if (Tcl_SplitList(interp, argv[3], &num_colcmds, &colcmds) != TCL_OK) {
        Tcl_AppendResult(interp, "cannot split color command name list", NULL);
        return TCL_ERROR;
      }

      char **colnames=(char **) calloc(1, num_colcmds * sizeof(char *));
      float *colors=(float *) calloc(1, num_colcmds * 3 * sizeof(float));

      int fail=0;
      for (i=0; i<num_colcmds; i++) {
        char tmpbuf[1024];
        if (sscanf(colcmds[i], "%s %f %f %f", tmpbuf, 
                   &colors[i*3], &colors[i*3 + 1], &colors[i*3 + 2]) != 4) {
          fail=i;
          break;
        }
        colnames[i] = strdup(tmpbuf);
      }

      if (!fail &&
          !app->color_change_rgblist(num_colcmds, (const char **) colnames, colors)) {
        Tcl_SetResult(interp, (char *) "Unable to change color namelist", TCL_STATIC); 
      }

      int freecnt=num_colcmds;
      if (fail) 
        freecnt=fail;
 
      for (i=0; i<freecnt; i++) {
        free(colnames[i]);
      }
      free(colnames);

      if (colcmds) 
        Tcl_Free((char *) colcmds);

      if (fail) {
        Tcl_AppendResult(interp, "cannot split/copy color command name list", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argv[2], "namelist", CMDLEN)) {
      const char **colcmds;
      int i, num_colcmds;
      if (Tcl_SplitList(interp, argv[3], &num_colcmds, &colcmds) != TCL_OK) {
        Tcl_AppendResult(interp, "cannot split color command name list", NULL);
        return TCL_ERROR;
      }

      char **colcats=(char **) calloc(1, num_colcmds * sizeof(char *));
      char **colnames=(char **) calloc(1, num_colcmds * sizeof(char *));
      char **colors=(char **) calloc(1, num_colcmds * sizeof(char *));

      int fail=0;
      for (i=0; i<num_colcmds; i++) {
        int cnt;
        const char **cmdparts;
        if (Tcl_SplitList(interp, colcmds[i], &cnt, &cmdparts) == TCL_OK) {
          if (cnt == 3) {
            colcats[i] = strdup(cmdparts[0]);
            colnames[i] = strdup(cmdparts[1]);
            colors[i] = strdup(cmdparts[2]);
          } else {
            fail=i;
            break;
          }
          Tcl_Free((char *) cmdparts);
        }
      }

      if (!fail &&
          !app->color_change_namelist(num_colcmds, colcats, colnames, colors)) {
        Tcl_SetResult(interp, (char *) "Unable to change color namelist", TCL_STATIC); 
      }

      int freecnt=num_colcmds;
      if (fail) 
        freecnt=fail;
 
      for (i=0; i<freecnt; i++) {
        free(colcats[i]);
        free(colnames[i]);
        free(colors[i]);
      }
      free(colcats);
      free(colnames);
      free(colors);

      if (colcmds) 
        Tcl_Free((char *) colcmds);

      if (fail) {
        Tcl_AppendResult(interp, "cannot split/copy color command name list", NULL);
        return TCL_ERROR;
      }
    }
  } else if (!strupncmp(argv[1], "scale", CMDLEN)) {
    /// color scale colors <method> [<color1> <color2> <color3>]
    if (argc >= 4 && !strupncmp(argv[2], "colors", CMDLEN)) {
      if (argc == 4) {
        int ind = app->colorscale_method_index(argv[3]);
        float vals[3][3];
        if (!app->get_colorscale_colors(ind, vals[0], vals[1], vals[2])) {
          Tcl_AppendResult(interp, "no colors available for method '", argv[3], "'.", NULL);
          return TCL_ERROR;
        }
        Tcl_Obj *result = Tcl_NewListObj(0,NULL);
        for (int i=0; i<3; i++) {
          Tcl_Obj *elem = Tcl_NewListObj(0, NULL);
          for (int j=0; j<3; j++) {
            Tcl_ListObjAppendElement(interp,elem,Tcl_NewDoubleObj(vals[i][j]));
          }
          Tcl_ListObjAppendElement(interp, result, elem);
        }
        Tcl_SetObjResult(interp, result);
        return TCL_OK;
      } else if (argc == 7) {
        int ind = app->colorscale_method_index(argv[3]);
        float vals[3][3];
        for (int i=0; i<3; i++) {
          // first interpret as vector
          int rc = tcl_get_vector(argv[4+i], vals[i], interp);
          if (rc != TCL_OK) {
            if (!app->color_value(argv[4+i], vals[i]+0, vals[i]+1, vals[i]+2)) {
              return TCL_ERROR;
            }
            // clear error since lookup by color name was successful.
            Tcl_ResetResult(interp);
          }
        }
        if (!app->set_colorscale_colors(ind, vals[0], vals[1], vals[2])) {
          Tcl_AppendResult(interp, "Unable to set colorscale colors for method '", argv[3], "'.", NULL);
          return TCL_ERROR;
        }
        return TCL_OK;
      }
    }
    if (argc == 4) {
      if (!strupncmp(argv[2], "method", CMDLEN)) {
        int ind = app->colorscale_method_index(argv[3]);
        if (ind < 0) {
          char tmpstring[1024];
          sprintf(tmpstring, "color scale method '%s' not recognized", argv[3]);
          Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
          return TCL_ERROR;
        }
        app->colorscale_setmethod(ind); 
      } else {
        float mid=0, min=0, max=0;
        app->colorscale_info(&mid, &min, &max); 
        float newval = (float) atof(argv[3]);
        if (!strupncmp(argv[2], "midpoint", CMDLEN)) mid = newval;
        else if (!strupncmp(argv[2], "min", CMDLEN)) min = newval; 
        else if (!strupncmp(argv[2], "max", CMDLEN)) max = newval;
        else {
          char tmpstring[1024]; 
          sprintf(tmpstring, "color scale option '%s' not recognized", argv[2]);
          Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
          return TCL_ERROR;
        }
        app->colorscale_setvalues(mid, min, max);
      }
    } else {
      Tcl_SetResult(interp, (char *) "color scale [method|midpoint|min|max] <value>", TCL_STATIC);
      return TCL_ERROR;
    }  
  } else if ((argc == 3 || argc == 4) && !strupncmp(argv[1], "restype", CMDLEN)) {
    if (argc == 3) {
      const char *result = app->color_get_restype(argv[2]);
      if (!result) {
        Tcl_AppendResult(interp, "No restype for residue '", argv[2], "'",
            NULL);
        return TCL_ERROR;
      }
      Tcl_SetResult(interp, (char *)result, TCL_STATIC);
    } else { //argc==4
      if (!app->color_set_restype(argv[2], argv[3])) {
        if (!app->color_change_name(argv[1], argv[2], argv[3])) {
          Tcl_AppendResult(interp, "Unable to set restype: invalid restype '",
                           argv[3], 
                           "' specified -- or unable to change color name", 
                           NULL);
          return TCL_ERROR;
        }
      }
    }
  } else if(argc == 3) {
    // Return the color string for the specified color category/name
    const char *colorname;
    if (app->color_get_from_name(argv[1], argv[2], &colorname)) {
      Tcl_SetResult(interp, (char*) colorname, TCL_STATIC); 
      return TCL_OK;
    } else {
      Tcl_SetResult(interp, (char *) "Unable to get color name", TCL_STATIC); 
      return TCL_ERROR;
    }
  } else if(argc == 4) {
    if (!app->color_change_name(argv[1], argv[2], argv[3])) {
      Tcl_SetResult(interp, (char *) "Unable to change color name", TCL_STATIC); 
      return TCL_ERROR;
    }
  } else if (argc == 6 && !strupncmp(argv[1], "add", CMDLEN) 
                       && !strupncmp(argv[2], "item", CMDLEN)) {
    if (!app->color_add_item(argv[3], argv[4], argv[5])) {
      Tcl_SetResult(interp, (char *) "Error adding color item.", TCL_STATIC);
      return TCL_ERROR;
    }
  } else {
    return TCL_ERROR;
  }
  
  // if here, everything worked out ok
  return TCL_OK;
}

