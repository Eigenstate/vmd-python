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
 *      $RCSfile: cmd_render.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.42 $       $Date: 2011/02/24 20:56:31 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for rendering control
 ***************************************************************************/

#include <stdlib.h>
#include <tcl.h>
#include "config.h"
#include "utilities.h"
#include "VMDApp.h"

#define CHECK_RENDER(x) if (!app->filerender_valid(x)) { delete [] extstr; Tcl_AppendResult(interp, "No render method: ", x, NULL); return TCL_ERROR; }

int text_cmd_render(ClientData cd, Tcl_Interp *interp, int argc,
                    const char *argv[]) {
  VMDApp *app = (VMDApp *)cd;

  if (argc >= 3) {
    char *extstr = NULL;
    if (argc > 3)
      extstr = combine_arguments(argc, argv, 3);
   
    if (!strupncmp(argv[1], "options", CMDLEN)) {
      const char *opt = app->filerender_option(argv[2], NULL);
      if (!opt) {
        Tcl_AppendResult(interp, "render:\n",
        "No rendering method '", argv[2], "' available.", NULL);
        return TCL_ERROR;
      } 
      if (extstr == NULL) { //print the option
        Tcl_AppendResult(interp, opt, NULL);
      } else {
        app->filerender_option(argv[2], extstr);
        delete [] extstr;
      }
      return TCL_OK;  

    } else if (!strupncmp(argv[1], "default", CMDLEN)) { 
      const char *opt = app->filerender_default_option(argv[2]);
      if (!opt) {
        Tcl_AppendResult(interp, "render:\n",
        "No rendering method '", argv[2], "' available.", NULL);
        return TCL_ERROR;
      }
      Tcl_AppendResult(interp, opt, NULL);
      return TCL_OK;

    } else if (!strupncmp(argv[1], "hasaa", CMDLEN)) {
      CHECK_RENDER(argv[2])
      Tcl_SetObjResult(interp, Tcl_NewIntObj(app->filerender_has_antialiasing(argv[2])));
      return TCL_OK;

    } else if (!strupncmp(argv[1], "aasamples", CMDLEN)) {
      int aasamples = -1;
      if (argc ==4) {
        if (Tcl_GetInt(interp, argv[3], &aasamples) != TCL_OK) 
          return TCL_ERROR;
      }
      CHECK_RENDER(argv[2])
      Tcl_SetObjResult(interp, Tcl_NewIntObj(app->filerender_aasamples(argv[2], aasamples)));
      return TCL_OK;

    } else if (!strupncmp(argv[1], "aosamples", CMDLEN)) {
      int aosamples = -1;
      if (argc ==4) {
        if (Tcl_GetInt(interp, argv[3], &aosamples) != TCL_OK) 
          return TCL_ERROR;
      }
      CHECK_RENDER(argv[2])
      Tcl_SetObjResult(interp, Tcl_NewIntObj(app->filerender_aosamples(argv[2], aosamples)));
      return TCL_OK;

    } else if (!strupncmp(argv[1], "imagesize", CMDLEN)) {
      int w=0, h=0;
      CHECK_RENDER(argv[2])
      if (argc == 4) {
        int listn;
        const char **listelem;
        if (Tcl_SplitList(interp, argv[3], &listn, &listelem) != TCL_OK) {
          return TCL_ERROR;
        }
        if (listn != 2) {
          Tcl_SetResult(interp, (char *) "Image size list must have two elements", TCL_STATIC);
          Tcl_Free((char *)listelem);
        }
        if (Tcl_GetInt(interp, listelem[0], &w) != TCL_OK ||
            Tcl_GetInt(interp, listelem[1], &h) != TCL_OK) {
          Tcl_Free((char *)listelem);
          return TCL_ERROR;
        }
        Tcl_Free((char *)listelem);
      } else if (argc != 3 && argc > 4) {
        Tcl_SetResult(interp, (char *) "Usage: render imagesize <method> {width height}", TCL_STATIC);
        return TCL_ERROR;
      }
      if (!app->filerender_imagesize(argv[2], &w, &h)) {
        Tcl_SetResult(interp, (char *) "Unable to set/get image size.", TCL_STATIC);
        return TCL_ERROR;
      }
      char tmpstring[128];
      sprintf(tmpstring, "%d %d", w, h);
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_OK;

    } else if (!strupncmp(argv[1], "hasimagesize", CMDLEN)) {
      int rc = app->filerender_has_imagesize(argv[2]);
      Tcl_SetObjResult(interp, Tcl_NewIntObj(rc));
      return TCL_OK;

    } else if (!strupncmp(argv[1], "aspectratio", CMDLEN)) {
      CHECK_RENDER(argv[2])
      double daspect = -1;
      if (argc == 4) {
        if (Tcl_GetDouble(interp, argv[3], &daspect) != TCL_OK) 
          return TCL_ERROR;
      }
      float aspect = (float)daspect;
      if (!app->filerender_aspectratio(argv[2], &aspect)) {
        Tcl_SetResult(interp, (char *) "Unable to get aspect ratio.", TCL_STATIC);
        return TCL_ERROR;
      }
      Tcl_SetObjResult(interp, Tcl_NewDoubleObj(aspect));
      return TCL_OK;
      
    } else if (!strupncmp(argv[1], "formats", CMDLEN)) {
      CHECK_RENDER(argv[2])
      int n = app->filerender_numformats(argv[2]);
      for (int i=0; i<n; i++) {
        Tcl_AppendElement(interp, app->filerender_get_format(argv[2], i));
      }
      return TCL_OK;

    } else if (!strupncmp(argv[1], "format", CMDLEN)) {
      CHECK_RENDER(argv[2])
      if (argc == 3) {
        Tcl_AppendElement(interp, app->filerender_cur_format(argv[2]));
        return TCL_OK;
      }
      if (app->filerender_set_format(argv[2], argv[3])) return TCL_OK;
      Tcl_AppendResult(interp, 
          "Unable to set render output format to ", argv[3], NULL);
      return TCL_ERROR;

    } else {
      app->display_update();
      int retval = app->filerender_render(argv[1], argv[2], extstr);
      if(extstr)
        delete [] extstr;
      return retval ? TCL_OK : TCL_ERROR;
    }
  } else if (argc == 2) {
    for (int i=0; i<app->filerender_num(); i++) 
      Tcl_AppendElement(interp, app->filerender_name(i));
    return TCL_OK;
  } 

  // if here, something went wrong, so return an error message
  Tcl_AppendResult(interp, "render usage:\n",
                   "render list\n",
                   "render hasaa <method>\n",
                   "render aasamples <method> [aasamples]\n",
                   "render aosamples <method> [aosamples]\n",
//                   "render hasimagesize <method>\n",
//                   "render imagesize <method> [{width height}]\n",
                   "render formats <method>\n",
                   "render format <method> <format>\n",
                   "render options <method> <new default exec command>\n",
                   "render default <method>\n",
                   "render <method> <filename> [exec command]\n",
                   NULL);
  return TCL_ERROR;
}


// XXX this fails to compile on Windows due to Tk's ridiculous 
// insistence of including X11 headers as part of tk.h, we've gotta
// find a better way of coping with this.
#if defined(VMDTK)  && !defined(_MSC_VER)
#include <tk.h>
#include "DisplayDevice.h"
int text_cmd_tkrender(ClientData cd, Tcl_Interp *interp, int argc,
                      const char *argv[]) {
  if (!Tcl_PkgPresent(interp, "Tk", TK_VERSION, 0)) {
    Tcl_SetResult(interp, "Tk not available.", TCL_STATIC);
    return TCL_ERROR;
  }
  if (argc != 2) {
    Tcl_SetResult(interp, "tkrender usage:\ntkrender <photo handle>\n",
        TCL_STATIC);
    return TCL_ERROR;
  }
  Tk_PhotoHandle handle = Tk_FindPhoto(interp, argv[1]);
  if (!handle) {
    Tcl_AppendResult(interp, "photo handle '", argv[1], "' has not been created.", NULL);
    return TCL_ERROR;
  }

  int xs=0, ys=0;
  DisplayDevice *display = ((VMDApp *)cd)->display;
  display->update(TRUE);
  unsigned char *img = display->readpixels(xs, ys);
  display->update(TRUE);

  if (!img) {
    Tcl_SetResult(interp, "Error reading pixel data from display device.",
        TCL_STATIC);
    return TCL_ERROR;
  }

  // OpenGL and Tk use opposite row order for pixel data.
  // Here we assume that readpixels returned data in packed RGB format.
  Tk_PhotoImageBlock blk = {
    img+3*xs*(ys-1),  // pixel pointer; points to start of top row on screen.
    xs,        // width
    ys,        // height
    -3*xs,     // address difference between two vertically adjacent pixels
    3,         // address difference between two horizontally adjacent pixels
    {0,1,2,3}  // r, g, b, a offsets within each pixel.
  };

  // set the size of the photo object to match the screen size.  One could
  // also create a new photo object with no size information, but that would
  // likely to lead to memory leaks in various scripts.  There's also currently
  // no way to read out the size of the display in VMD (sad, I know), so
  // it's just easier to set it here.
#if TCL_MAJOR_VERSION == 8 && TCL_MINOR_VERSION < 5
  Tk_PhotoSetSize(handle, xs, ys);
  Tk_PhotoPutBlock(handle, &blk, 0, 0, xs, ys, TK_PHOTO_COMPOSITE_SET);
#else
  Tk_PhotoSetSize(interp, handle, xs, ys);
  Tk_PhotoPutBlock(interp, handle, &blk, 0, 0, xs, ys, TK_PHOTO_COMPOSITE_SET);
#endif
  free(img);
  return TCL_OK;
}

#endif


