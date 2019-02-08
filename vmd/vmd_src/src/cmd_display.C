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
 *      $RCSfile: cmd_display.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.78 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for controlling the OpenGL display window
 ***************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <tcl.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#include "config.h"
#include "DisplayDevice.h"
#include "Axes.h"
#include "CommandQueue.h"
#include "MoleculeList.h"
#include "VMDApp.h"
#include "Stage.h"
#include "TclCommands.h"
#include "Scene.h"
#include "FPS.h"

// These are the Command:: enums used by this file:
//  DISP_RESHAPE, DISP_STEREO, DISP_STEREOSWAP, 
//  DISP_PROJ, DISP_EYESEP, DISP_FOCALLEN, 
//  DISP_LIGHT_ON, DISP_LIGHT_HL, DISP_LIGHT_ROT, DISP_MATERIALS_CHANGE,
//  DISP_CLIP, DISP_DEPTHCUE, DISP_ANTIALIAS, DISP_SCRHEIGHT, DISP_SCRDIST,
//  CMD_AXES, CMD_STAGE


////////////////////////////////////////////////////////////////////
///////////////////////  text processors
////////////////////////////////////////////////////////////////////

// text callback routine for 'display'; return TCL_ERROR if an error occurs.
#define TCL_RET(fmt, val)      \
sprintf(s, fmt, val);          \
Tcl_AppendElement(interp, s);  \
return TCL_OK

int text_cmd_display(ClientData cd, Tcl_Interp *interp, int argc, 
                     const char *argv[]) {
  VMDApp *app = (VMDApp *)cd;

  // not much help, but anything is nice
  if (argc <= 1) {
    Tcl_SetResult(interp,
       (char *)
       "display get <eyesep | focallength | height | distance | antialias |\n"
       "             depthcue | culling | size | \n"
       "             stereo | stereomodes | stereoswap |\n"
       "             cachemode | cachemodes | rendermode | rendermodes |\n"
       "             projection | projections | nearclip | farclip |\n"
       "             cuestart | cueend | cuedensity | cuemode |\n" 
       "             shadows | ambientocclusion | aoambient | aodirect |\n"
       "             dof | dof_fnumber | dof_focaldist | backgroundgradient>\n"
       "display <reshape | resetview | resize | reposition>\n"
       "display <eyesep | focallength | height | distance | antialias |\n"
       "         depthcue | culling | cachemode | rendermode |\n"
       "         stereo | stereoswap |\n"
       "         shadows | ambientocclusion | aoambient | aodirect |\n"
       "         dof | dof_fnumber | dof_focaldist |\n"
       "         backgroundgradient> newvalue\n"
       "display <nearclip | farclip> <set | add> newvalue\n"
       "display <cuestart | cueend | cuedensity | cuemode> newvalue\n"
       "display fps [on | off ]\n"
       "display update [on | off | status | ui]",
       TCL_STATIC);
    return TCL_ERROR;
  }

  // the new 'get' commands
  if (argc == 3 && !strupncmp(argv[1], "get", CMDLEN)) {
    char s[128];
    if        (!strupncmp(argv[2], "eyesep", CMDLEN)) {
      TCL_RET("%f", app->display->eyesep());
#if 1
    // XXX undocumented eye manipulation commands
    // To use these in the model coordinate system, one must set the 
    // model to world coordinate system transformation matrices to identity
    // matrices.  Future code should automatically transform the provided
    // eye position/direction/up etc into the model coordinate system.
    } else if (!strupncmp(argv[2], "eyepos", CMDLEN)) {
      float pos[3];
      app->display->get_eye_pos(&pos[0]);
      for (int i=0; i<3; i++) {
        sprintf(s, "%f",  pos[i]);
        Tcl_AppendElement(interp, s);
      }
      return TCL_OK;
    } else if (!strupncmp(argv[2], "eyedir", CMDLEN)) {
      float dir[3];
      app->display->get_eye_dir(&dir[0]);
      for (int i=0; i<3; i++) {
        sprintf(s, "%f",  dir[i]);
        Tcl_AppendElement(interp, s);
      }
      return TCL_OK;
    } else if (!strupncmp(argv[2], "eyeup", CMDLEN)) {
      float up[3];
      app->display->get_eye_up(&up[0]);
      for (int i=0; i<3; i++) {
        sprintf(s, "%f",  up[i]);
        Tcl_AppendElement(interp, s);
      }
      return TCL_OK;
#endif
    } else if (!strupncmp(argv[2], "focallength", CMDLEN)) {
      TCL_RET("%f", app->display->eye_dist());
    } else if (!strupncmp(argv[2], "height", CMDLEN)) {
      TCL_RET("%f", app->display->screen_height());
    } else if (!strupncmp(argv[2], "distance", CMDLEN)) {
      TCL_RET("%f", app->display->distance_to_screen());

    } else if (!strupncmp(argv[2], "ambientocclusion", CMDLEN)) {
      Tcl_AppendElement(interp, 
        app->display->ao_enabled() ? "on" : "off");
      return TCL_OK;
    } else if (!strupncmp(argv[2], "aoambient", CMDLEN)) {
      TCL_RET("%f", app->display->get_ao_ambient());
    } else if (!strupncmp(argv[2], "aodirect", CMDLEN)) {
      TCL_RET("%f", app->display->get_ao_direct());

    } else if (!strupncmp(argv[2], "dof", CMDLEN)) {
      Tcl_AppendElement(interp, 
        app->display->dof_enabled() ? "on" : "off");
      return TCL_OK;
    } else if (!strupncmp(argv[2], "dof_fnumber", CMDLEN)) {
      TCL_RET("%f", app->display->get_dof_fnumber());
    } else if (!strupncmp(argv[2], "dof_focaldist", CMDLEN)) {
      TCL_RET("%f", app->display->get_dof_focal_dist());

    } else if (!strupncmp(argv[2], "antialias", CMDLEN)) {
      Tcl_AppendElement(interp, 
        app->display->aa_enabled() ? "on" : "off");
      return TCL_OK;
    } else if (!strupncmp(argv[2], "depthcue", CMDLEN)) {
      Tcl_AppendElement(interp, 
        app->display->cueing_enabled() ? "on" : "off");
      return TCL_OK;
    } else if (!strupncmp(argv[2], "backgroundgradient", CMDLEN)) {
      Tcl_AppendElement(interp, 
        app->scene->background_mode() ? "on" : "off");
      return TCL_OK;
    } else if (!strupncmp(argv[2], "culling", CMDLEN)) {
      Tcl_AppendElement(interp, 
        app->display->culling_enabled() ? "on" : "off");
      return TCL_OK;
    } else if (!strupncmp(argv[2], "shadows", CMDLEN)) {
      Tcl_AppendElement(interp, 
        app->display->shadows_enabled() ? "on" : "off");
      return TCL_OK;
    } else if (!strupncmp(argv[2], "size", CMDLEN)) {
      int w, h;
      app->display_get_size(&w, &h);
      sprintf(s, "%d", w);
      Tcl_AppendElement(interp, s);
      sprintf(s, "%d", h);
      Tcl_AppendElement(interp, s);
      return TCL_OK;
    } else if (!strupncmp(argv[2], "fps", CMDLEN)) {
      Tcl_AppendElement(interp, app->fps->displayed() ? "on" : "off");
      return TCL_OK;
    } else if (!strupncmp(argv[2], "stereo", CMDLEN)) {
      Tcl_AppendElement(interp, 
                        app->display->stereo_name(app->display->stereo_mode()));
      return TCL_OK;
    } else if (!strupncmp(argv[2], "stereoswap", CMDLEN)) {
      Tcl_AppendElement(interp, app->display->stereo_swap() ? "on" : "off");
      return TCL_OK;
    } else if (!strupncmp(argv[2], "stereomodes", CMDLEN)) {
      int i;
      for (i=0; i<app->display->num_stereo_modes(); i++) {
        Tcl_AppendElement(interp, app->display->stereo_name(i));
      }
      return TCL_OK;
    } else if (!strupncmp(argv[2], "cachemode", CMDLEN)) {
      Tcl_AppendElement(interp, 
        app->display->cache_name(app->display->cache_mode()));
      return TCL_OK;
    } else if (!strupncmp(argv[2], "cachemodes", CMDLEN)) {
      int i;
      for (i=0; i<app->display->num_cache_modes(); i++) {
        Tcl_AppendElement(interp, app->display->cache_name(i));
      }
      return TCL_OK;
    } else if (!strupncmp(argv[2], "rendermode", CMDLEN)) {
      Tcl_AppendElement(interp, 
        app->display->render_name(app->display->render_mode()));
      return TCL_OK;
    } else if (!strupncmp(argv[2], "rendermodes", CMDLEN)) {
      int i;
      for (i=0; i<app->display->num_render_modes(); i++) {
        Tcl_AppendElement(interp, app->display->render_name(i));
      }
      return TCL_OK;
    } else if (!strupncmp(argv[2], "projection", CMDLEN)) {
      Tcl_AppendResult(interp, app->display->get_projection(), NULL);
      return TCL_OK;
    } else if (!strupncmp(argv[2], "projections", CMDLEN)) {
      for (int i=0; i<app->display->num_projections(); i++)
        Tcl_AppendElement(interp, app->display->projection_name(i));
      return TCL_OK;
    } else if (!strupncmp(argv[2], "nearclip", CMDLEN)) {
      TCL_RET("%f", app->display->near_clip());
    } else if (!strupncmp(argv[2], "farclip", CMDLEN)) {
      TCL_RET("%f", app->display->far_clip());
    } else if (!strupncmp(argv[2], "cuestart", CMDLEN)) {
      TCL_RET("%f", app->display->get_cue_start());
    } else if (!strupncmp(argv[2], "cueend", CMDLEN)) {
      TCL_RET("%f", app->display->get_cue_end());
    } else if (!strupncmp(argv[2], "cuedensity", CMDLEN)) {
      TCL_RET("%f", app->display->get_cue_density());
    } else if (!strupncmp(argv[2], "cuemode", CMDLEN)) {
      Tcl_AppendResult(interp, app->display->get_cue_mode(), NULL);
      return TCL_OK;
    } else {
      Tcl_SetResult(interp,
        (char *)
        "possible parameters to 'display get' are:\n"
        "eyesep focallength height distance antialias depthcue culling\n"
        "stereo stereomodes stereoswap nearclip farclip\n" 
        "cuestart cueend cuedensity cuemode\n"
        "shadows, ambientocclusion, aoambient, aodirect\n",
        TCL_STATIC);
      return TCL_ERROR;
    }
    /// return TCL_OK;  // never reached
  }

  if(argc == 2) {
    if(!strupncmp(argv[1],"resetview",CMDLEN)) {
      app->scene_resetview();
      return TCL_OK;
    } else if(!strupncmp(argv[1],"update",CMDLEN)) {
      app->display_update();
      return TCL_OK;
    } else
      return TCL_ERROR;

  } else if(argc == 3) {
    if (!strupncmp(argv[1], "fps", CMDLEN)) {
      int on;
      if (Tcl_GetBoolean(interp, argv[2], &on) != TCL_OK) return TCL_ERROR;
      app->display_set_fps(on);
#if 1
    // XXX undocumented eye manipulation commands
    // To use these in the model coordinate system, one must set the 
    // model to world coordinate system transformation matrices to identity
    // matrices.  Future code should automatically transform the provided
    // eye position/direction/up etc into the model coordinate system.
    } else if (!strupncmp(argv[1], "eyepos", CMDLEN)) {
      float pos[3];
      if (tcl_get_vector(argv[2], &pos[0],  interp) != TCL_OK) {
        return TCL_ERROR;
      }
      app->display->set_eye_pos(&pos[0]);
      return TCL_OK;
    } else if (!strupncmp(argv[1], "eyedir", CMDLEN)) {
      float dir[3];
      if (tcl_get_vector(argv[2], &dir[0],  interp) != TCL_OK) {
        return TCL_ERROR;
      }
      app->display->set_eye_dir(&dir[0]);
      return TCL_OK;
    } else if (!strupncmp(argv[1], "eyeup", CMDLEN)) {
      float up[3];
      if (tcl_get_vector(argv[2], &up[0],  interp) != TCL_OK) {
        return TCL_ERROR;
      }
      app->display->set_eye_up(&up[0]);
      return TCL_OK;
#endif
    } else if(!strupncmp(argv[1],"eyesep",CMDLEN)) {
      app->display_set_eyesep((float)atof(argv[2]));
    } else if(!strupncmp(argv[1],"focallength",CMDLEN)) {
      app->display_set_focallen((float)atof(argv[2]));
    } else if(!strupncmp(argv[1],"height",CMDLEN)) {
      app->display_set_screen_height((float) atof(argv[2]));
    } else if(!strupncmp(argv[1],"distance",CMDLEN)) {
      app->display_set_screen_distance((float) atof(argv[2]));

    } else if(!strupncmp(argv[1],"ambientocclusion",CMDLEN)) {
      int onoff=0;
      if (Tcl_GetBoolean(interp, argv[2], &onoff) != TCL_OK) return TCL_ERROR;
      app->display_set_ao(onoff);
    } else if(!strupncmp(argv[1],"aoambient",CMDLEN)) {
      double val=0;
      if (Tcl_GetDouble(interp, argv[2], &val) != TCL_OK) return TCL_ERROR;
      app->display_set_ao_ambient((float)val);
    } else if(!strupncmp(argv[1],"aodirect",CMDLEN)) {
      double val=0;
      if (Tcl_GetDouble(interp, argv[2], &val) != TCL_OK) return TCL_ERROR;
      app->display_set_ao_direct((float)val);

    } else if(!strupncmp(argv[1],"dof",CMDLEN)) {
      int onoff=0;
      if (Tcl_GetBoolean(interp, argv[2], &onoff) != TCL_OK) return TCL_ERROR;
      app->display_set_dof(onoff);
    } else if(!strupncmp(argv[1],"dof_fnumber",CMDLEN)) {
      double val=0;
      if (Tcl_GetDouble(interp, argv[2], &val) != TCL_OK) return TCL_ERROR;
      app->display_set_dof_fnumber((float)val);
    } else if(!strupncmp(argv[1],"dof_focaldist",CMDLEN)) {
      double val=0;
      if (Tcl_GetDouble(interp, argv[2], &val) != TCL_OK) return TCL_ERROR;
      app->display_set_dof_focal_dist((float)val);

    } else if(!strupncmp(argv[1],"antialias",CMDLEN)) {
      int onoff=0;
      if (Tcl_GetBoolean(interp, argv[2], &onoff) != TCL_OK) return TCL_ERROR;
      app->display_set_aa(onoff);
    } else if(!strupncmp(argv[1],"depthcue",CMDLEN)) {
      int onoff=0;
      if (Tcl_GetBoolean(interp, argv[2], &onoff) != TCL_OK) return TCL_ERROR;
      app->display_set_depthcue(onoff);
    } else if(!strupncmp(argv[1],"backgroundgradient",CMDLEN)) {
      int onoff=0;
      if (Tcl_GetBoolean(interp, argv[2], &onoff) != TCL_OK) return TCL_ERROR;
      app->display_set_background_mode(onoff);
    } else if(!strupncmp(argv[1],"culling",CMDLEN)) {
      int onoff=0;
      if (Tcl_GetBoolean(interp, argv[2], &onoff) != TCL_OK) return TCL_ERROR;
      app->display_set_culling(onoff);
    } else if(!strupncmp(argv[1],"shadows",CMDLEN)) {
      int onoff=0;
      if (Tcl_GetBoolean(interp, argv[2], &onoff) != TCL_OK) return TCL_ERROR;
      app->display_set_shadows(onoff);
    } else if(!strupncmp(argv[1],"stereo",CMDLEN)) {
      app->display_set_stereo(argv[2]);
    } else if(!strupncmp(argv[1],"stereoswap",CMDLEN)) {
      int onoff=0;
      if (Tcl_GetBoolean(interp, argv[2], &onoff) != TCL_OK) return TCL_ERROR;
      app->display_set_stereo_swap(onoff);
    } else if(!strupncmp(argv[1],"cachemode",CMDLEN)) {
      app->display_set_cachemode(argv[2]);
    } else if(!strupncmp(argv[1],"rendermode",CMDLEN)) {
      app->display_set_rendermode(argv[2]);
    } else if (!strupncmp(argv[1], "projection", CMDLEN) ||
               !strupncmp(argv[1], "proj", CMDLEN)) {
      if (!app->display_set_projection(argv[2])) {
        Tcl_AppendResult(interp, "Invalid projection: ", argv[2], NULL);
        return TCL_ERROR;
      }
    } else if(!strupncmp(argv[1],"update",CMDLEN)) {
      int booltmp;
      if (!strcmp(argv[2], "status")) {
        char s[20];
        TCL_RET("%d", app->display_update_status());
      } else if (!strcmp(argv[2], "ui")) {
        app->display_update_ui();
        return TCL_OK;
      } else if (Tcl_GetBoolean(interp, argv[2], &booltmp) == TCL_OK) {
        app->display_update_on(booltmp); 
        return TCL_OK;
      } else {
        return TCL_ERROR;
      }
    } else if(!strupncmp(argv[1],"cuemode",CMDLEN)) {
      if (!app->depthcue_set_mode(argv[2])) {
        Tcl_AppendResult(interp, "Illegal cuemode: ", argv[2], NULL);
        return TCL_ERROR;
      }
    } else if(!strupncmp(argv[1],"cuestart",CMDLEN)) {
      double val=0;
      if (Tcl_GetDouble(interp, argv[2], &val) != TCL_OK) return TCL_ERROR;
      app->depthcue_set_start((float)val);
    } else if(!strupncmp(argv[1],"cueend",CMDLEN)) {
      double val=0;
      if (Tcl_GetDouble(interp, argv[2], &val) != TCL_OK) return TCL_ERROR;
      app->depthcue_set_end((float)val);
    } else if(!strupncmp(argv[1],"cuedensity",CMDLEN)) {
      double val=0;
      if (Tcl_GetDouble(interp, argv[2], &val) != TCL_OK) return TCL_ERROR;
      app->depthcue_set_density((float)val);
    } else
      return TCL_ERROR;

  } else if(argc == 4) {
    if(!strupncmp(argv[1],"nearclip",CMDLEN)) {
      int isdelta = -1;
      if(!strupncmp(argv[2],"set",CMDLEN))
        isdelta = 0;
      else if(!strupncmp(argv[2],"add",CMDLEN))
        isdelta = 1;
      if (isdelta < 0) return TCL_ERROR;
      app->display_set_nearclip((float)atof(argv[3]), isdelta);
    } else if(!strupncmp(argv[1],"farclip",CMDLEN)) {
      int isdelta = -1;
      if(!strupncmp(argv[2],"set",CMDLEN))
        isdelta = 0;
      else if(!strupncmp(argv[2],"add",CMDLEN))
        isdelta = 1;
      if (isdelta < 0) return TCL_ERROR;
      app->display_set_farclip((float)atof(argv[3]), isdelta);
    } else if (!strupncmp(argv[1], "resize", CMDLEN)) {
      int w, h;
      if (Tcl_GetInt(interp, argv[2], &w) != TCL_OK ||
          Tcl_GetInt(interp, argv[3], &h) != TCL_OK)
        return TCL_ERROR;
      app->display_set_size(w, h);
    } else if (!strupncmp(argv[1], "reposition", CMDLEN)) {
      int x, y;
      if (Tcl_GetInt(interp, argv[2], &x) != TCL_OK ||
          Tcl_GetInt(interp, argv[3], &y) != TCL_OK)
        return TCL_ERROR;
      app->display_set_position(x, y);
    } else
      return TCL_ERROR;
  } else
    return TCL_ERROR;

  // if here, completed successfully
  return TCL_OK;
}


int text_cmd_light(ClientData cd, Tcl_Interp *interp, int argc, 
                   const char *argv[]) {
  VMDApp *app = (VMDApp *)cd;
  Scene *scene = app->scene;

  if (argc <= 1) {
    Tcl_SetResult(interp, 
      (char *) 
      "light <number> [on|off|highlight|unhighlight|status]\n"
      "light <number> rot <axis> <deg>\n"
      "light <number> pos\n"
      "light <number> pos [{x y z} | default]\n"
      "light num\n", 
      TCL_STATIC);
    return TCL_ERROR;
  }

  if ((argc == 3 || argc == 4) && !strupncmp(argv[2], "pos", CMDLEN)) {
    int num = atoi(argv[1]);
    if (argc == 4) {
      float pos[3];
      if (!strupncmp(argv[3], "default", 8)) {
        const float *def = scene->light_pos_default(num);
        if (!def) return TCL_ERROR;
        for (int i=0; i<3; i++) {
          char buf[20];
          sprintf(buf, "%f", def[i]);
          Tcl_AppendElement(interp, buf);
        }
        return TCL_OK;
        
      } else if (tcl_get_vector(argv[3], pos, interp) != TCL_OK) {
        return TCL_ERROR;
      }
      app->light_move(num, pos);
      return TCL_OK;
    } else {
      const float *pos = scene->light_pos(num);
      if (!pos) return TCL_ERROR;
      for (int i=0; i<3; i++) {
        char buf[20];
        sprintf(buf, "%f", pos[i]);
        Tcl_AppendElement(interp, buf);
      }
      return TCL_OK;
    }
  }  

  if (argc == 2 && !strupncmp(argv[1], "num", CMDLEN)) {
    // return the number of lights
    char tmpstring[64];
    sprintf(tmpstring, "%d", DISP_LIGHTS);
    Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    return TCL_OK;
  }
  int n;
  if (Tcl_GetInt(interp, argv[1], &n) != TCL_OK) {
    Tcl_AppendResult(interp, " -- light <number> ...", NULL);
    return TCL_ERROR;
  }

  if (argc == 3) {
    if(!strupncmp(argv[2],"on",CMDLEN))
      app->light_on(n, 1);
    else if(!strupncmp(argv[2],"off",CMDLEN))
      app->light_on(n, 0);
    else if(!strupncmp(argv[2],"highlight",CMDLEN))
      app->light_highlight(n, 1);
    else if(!strupncmp(argv[2],"unhighlight",CMDLEN))
      app->light_highlight(n, 0);
    else if(!strupncmp(argv[2],"status",CMDLEN)) {
      char tmpstring[1024];

      // return the pair { is on , is highlight} as eg: {on unhighlight}
      if (n < 0 || n >= DISP_LIGHTS) {
        sprintf(tmpstring, "light value %d out of range", n);
        Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
        return TCL_ERROR;
      }
      sprintf(tmpstring, "%s %s", 
        app->scene->light_active(n) ? "on" : "off",
        app->scene->light_highlighted(n) ?  "highlight" : "unhighlight");
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_OK;
    } else {
      return TCL_ERROR;
    }
  } else if(argc == 5 && !strupncmp(argv[2],"rot",CMDLEN)) {
    char axis = (char)(tolower(argv[3][0]));
    float deg = (float) atof(argv[4]);
    app->light_rotate(n, deg, axis);

  } else {
    return TCL_ERROR;
  }

  // if here, completed successfully
  return TCL_OK;
}


int text_cmd_point_light(ClientData cd, Tcl_Interp *interp, int argc, 
                         const char *argv[]) {
  VMDApp *app = (VMDApp *)cd;
  Scene *scene = app->scene;

  if (argc <= 1) {
    Tcl_SetResult(interp, 
      (char *) 
      "pointlight <number> [on|off|highlight|unhighlight|status]\n"
//      "pointlight <number> rot <axis> <deg>\n"
      "pointlight <number> pos\n"
      "pointlight <number> pos [{x y z} | default]\n"
      "pointlight <number> attenuation [{constant linear quadratic} | default]\n"
      "pointlight num\n", 
      TCL_STATIC);
    return TCL_ERROR;
  }
    if ((argc == 3 || argc == 4) && !strupncmp(argv[2], "pos", CMDLEN)) {
      int num = atoi(argv[1]);
      if (argc == 4) {
        float pos[3];
        if (!strupncmp(argv[3], "default", 8)) {
          const float *def = scene->adv_light_pos_default(num);
          if (!def) return TCL_ERROR;
          for (int i=0; i<3; i++) {
            char buf[20];
            sprintf(buf, "%f", def[i]);
            Tcl_AppendElement(interp, buf);
          }
          return TCL_OK;
          
        } else if (tcl_get_vector(argv[3], pos, interp) != TCL_OK) {
          return TCL_ERROR;
        }
// XXX hack
#if 1
        // XXX need to save the active coordinate transform so the 
        //     user's local model coordinates can be used here.
        scene->move_adv_light(num, pos);
#else
//        app->light_move(num, pos);
        scene->move_adv_light(num, pos);
#endif
        return TCL_OK;
      } else {
        const float *pos = scene->adv_light_pos(num);
        if (!pos) return TCL_ERROR;
        for (int i=0; i<3; i++) {
          char buf[20];
          sprintf(buf, "%f", pos[i]);
          Tcl_AppendElement(interp, buf);
        }
        return TCL_OK;
      }
    }  

    if ((argc == 3 || argc == 4) && !strupncmp(argv[2], "attenuation", CMDLEN)) {
      int num = atoi(argv[1]);
      if (argc == 4) {
        float factors[3] = { 1.0f, 0.0f, 0.0f };
        if (!strupncmp(argv[3], "default", 8)) {
          scene->adv_light_attenuation(num, factors[0], factors[1], factors[2]);
          return TCL_OK;
          
        } else if (tcl_get_vector(argv[3], factors, interp) != TCL_OK) {
          return TCL_ERROR;
        }
// XXX hack
#if 1
        scene->adv_light_attenuation(num, factors[0], factors[1], factors[2]);
#endif
        return TCL_OK;
      } else {
        float factors[3] = { 1.0f, 0.0f, 0.0f };
        scene->adv_light_get_attenuation(num, factors[0], factors[1], factors[2]);
        for (int i=0; i<3; i++) {
          char buf[20];
          sprintf(buf, "%f", factors[i]);
          Tcl_AppendElement(interp, buf);
        }
        return TCL_OK;
      }
    }  

    if (argc == 2 && !strupncmp(argv[1], "num", CMDLEN)) {
      // return the number of lights
      char tmpstring[64];
      sprintf(tmpstring, "%d", DISP_LIGHTS);
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_OK;
    }
    int n;
    if (Tcl_GetInt(interp, argv[1], &n) != TCL_OK) {
      Tcl_AppendResult(interp, " -- pointlight <number> ...", NULL);
      return TCL_ERROR;
    }

    if (argc == 3) {
      if(!strupncmp(argv[2],"on", CMDLEN))
        scene->activate_adv_light(n, 1);
      else if(!strupncmp(argv[2],"off", CMDLEN))
        scene->activate_adv_light(n, 0);
      else if(!strupncmp(argv[2],"highlight", CMDLEN))
        scene->highlight_adv_light(n, 1);
      else if(!strupncmp(argv[2],"unhighlight", CMDLEN))
        scene->highlight_adv_light(n, 0);
      else if(!strupncmp(argv[2],"status", CMDLEN)) {
        char tmpstring[1024];

      // return the pair { is on , is highlight} as eg: {on unhighlight}
      if (n < 0 || n >= DISP_LIGHTS) {
        sprintf(tmpstring, "light value %d out of range", n);
        Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
        return TCL_ERROR;
      }
      sprintf(tmpstring, "%s %s", 
        app->scene->adv_light_active(n) ? "on" : "off",
        app->scene->adv_light_highlighted(n) ?  "highlight" : "unhighlight");
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_OK;
    } else {
      return TCL_ERROR;
    }

#if 0
    } else if(argc == 5 && !strupncmp(argv[2],"rot",CMDLEN)) {
      char axis = (char)(tolower(argv[3][0]));
      float deg = (float) atof(argv[4]);
      app->light_rotate(n, deg, axis);
#endif
  } else {
    return TCL_ERROR;
  }

  // if here, completed successfully
  return TCL_OK;
}


int text_cmd_axes(ClientData cd, Tcl_Interp *interp, int argc, 
                     const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if (app->axes && argc == 2) {
    if (!strupncmp(argv[1],"location", CMDLEN)) {
      // return the current location
      Tcl_SetResult(interp, app->axes->loc_description(app->axes->location()), TCL_VOLATILE);
      return TCL_OK;
    } else if(!strupncmp(argv[1],"locations", CMDLEN)) {
      // return all the possible locations
      for (int ii=0; ii<app->axes->locations(); ii++) {
        Tcl_AppendElement(interp, app->axes->loc_description(ii));
      }
      return TCL_OK;
    }
    // else we are at an error, so return a short list
    Tcl_AppendResult(interp, 
                     "axes [location|locations]\n",
                     "axes location [off|origin|lowerleft|lowerright|"
                     "upperleft|upperright]",
                     NULL);
    return TCL_ERROR;
  }
  if (app->axes && argc == 3) {
    if (!strupncmp(argv[1],"location", CMDLEN)) {
      if  (!app->axes_set_location(argv[2])) {
        Tcl_AppendResult(interp, "Invalid axes location: ", argv[2], NULL);
        return TCL_ERROR;
      }
    }
  } else
    return TCL_ERROR;
 
  // if here, completed successfully
  return TCL_OK;
}

// text callback routine for 'stage'; return TCL_ERROR if an error occurs.
int text_cmd_stage(ClientData cd, Tcl_Interp *interp, int argc, 
                     const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;
  Stage *stage = app->stage;
  if (!stage)
    return TCL_ERROR;

  if (argc < 2 || argc > 3) {
    int i;
    Tcl_AppendResult(interp, (char *) "stage location <", NULL);
    for (i=0; i < stage->locations(); i++) {
      Tcl_AppendResult(interp, stage->loc_description(i), NULL);
      if (i < (stage->locations()-1))
        Tcl_AppendResult(interp, " | ", NULL);
    }
    Tcl_AppendResult(interp, (char *) ">\n", NULL);
    Tcl_AppendResult(interp, (char *) "stage locations\n", NULL);
    Tcl_AppendResult(interp, (char *) "stage panels [ numpanels ]\n", NULL);
    Tcl_AppendResult(interp, (char *) "stage size [ value ]\n", NULL);
    return TCL_ERROR;
  }

  if (argc == 2) {
    if (!strupncmp(argv[1], "location", CMDLEN)) {
      Tcl_AppendElement(interp, stage->loc_description(stage->location()));
      return TCL_OK;
    } else if (!strupncmp(argv[1], "locations", CMDLEN)) {
      int i;
      for (i=0; i < stage->locations(); i++) {
        Tcl_AppendElement(interp, stage->loc_description(i));
      }
      return TCL_OK;
    } else if (!strupncmp(argv[1], "panels", CMDLEN)) {
      char s[20];
      sprintf(s, "%d", stage->panels());
      Tcl_AppendElement(interp, s);
      return TCL_OK;
    } else if (!strupncmp(argv[1], "size", CMDLEN)) {
      char s[20];
      sprintf(s, "%f", stage->size());
      Tcl_AppendElement(interp, s);
      return TCL_OK;
    } else {
      Tcl_AppendResult(interp, "possible commands are: location, locations, "
                       "panels [value]",  NULL);
      return TCL_ERROR;
    }
    // doesn't get here
  }

  if (argc == 3) {
    int i;
    if (!strupncmp(argv[1],"location",CMDLEN)) {
      if (app->stage_set_location(argv[2])) return TCL_OK;
      Tcl_AppendResult(interp, "Possible locations are ",  NULL);
      for (i=0; i<stage->locations(); i++) 
        Tcl_AppendElement(interp, stage->loc_description(i));
      return TCL_ERROR;
    } else if(!strupncmp(argv[1],"panels",CMDLEN)) {
      int num=0;
      if (Tcl_GetInt(interp, argv[2], &num) != TCL_OK ||
          !app->stage_set_numpanels(num)) 
        return TCL_ERROR;
    } else if(!strupncmp(argv[1],"size",CMDLEN)) {
      double sz=1.0;
      if (Tcl_GetDouble(interp, argv[2], &sz) != TCL_OK ||
          !app->stage_set_size((float) sz))
        return TCL_ERROR;
    } else {
      return TCL_ERROR;
    } 
  }
 
  // if here, completed successfully
  return TCL_OK;
}



