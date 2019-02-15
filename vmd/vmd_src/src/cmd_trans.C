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
 *      $RCSfile: cmd_trans.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.26 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for transformation control
 ***************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <tcl.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#include "config.h"
#include "utilities.h"
#include "VMDApp.h"

int text_cmd_rotmat(ClientData cd, Tcl_Interp *, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

   // XXX This is crappy, it should take a list or matrix as third argument
   if (argc != 11) {
      return TCL_ERROR;
   }
   int rotBy = !strcasecmp(argv[1], "by");
   
   float tmp[16];
   memset(tmp, 0, sizeof(tmp));
   for (int i=0; i<9; i++) {
      tmp[i+i/3] = (float) atof(argv[i+2]);
   }
   tmp[15] = 1.0;
   int retval = rotBy ? app->scene_rotate_by(tmp)
                      : app->scene_rotate_to(tmp); 
   if (retval)
     return TCL_OK;
   return TCL_ERROR;
}

int text_cmd_rotate(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if(argc == 2 && !strupncmp(argv[1],"stop",CMDLEN)) {
    if (app->scene_stoprotation())
      return TCL_OK; 

  } else if(argc >= 4 && argc <= 5) {
    char axis = (char)(tolower(*(argv[1])));
    int rotby = !strupcmp(argv[2],"by");
    float deg = (float) atof(argv[3]);
    float incr = (argc == 5 ? (float)atof(argv[4]) : 0);
    int retval = rotby ? app->scene_rotate_by(deg, axis, incr)
                       : app->scene_rotate_to(deg, axis);
    if (retval)
      return TCL_OK;
  }      
  Tcl_AppendResult(interp, "rotate usage:\n",
    "rotate stop -- stop current rotation\n", 
    "rotate [x | y | z] by <angle> -- rotate in one step\n", 
    "rotate [x | y | z] by <angle> <increment> -- smooth transition\n", 
    NULL);  // XXX Fix me!  Need rotate to commands
  
  return TCL_ERROR;
}

int text_cmd_translate(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;
  if(argc == 5) {
    int trby=!strupcmp(argv[1],"by");
    float x, y, z;
    x = (float) atof(argv[2]);
    y = (float) atof(argv[3]);
    z = (float) atof(argv[4]);
    int retval = trby ? app->scene_translate_by(x, y, z)
                      : app->scene_translate_to(x, y, z);
    if (retval)
      return TCL_OK;
  }

  Tcl_AppendResult(interp, "translate usage:\n",
    "translate [by | to] <x> <y> <z> -- move viewpoint by/to given vector\n",
    NULL); 
  return TCL_ERROR;
}

int text_cmd_scale(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if(argc == 3) {
    int scby = !strupcmp(argv[1],"by");
    float s = (float) atof(argv[2]);
    int retval = scby ? app->scene_scale_by(s)
                      : app->scene_scale_to(s);

    if (retval)
      return TCL_OK;
  }

  Tcl_AppendResult(interp, "scale usage:\n",
                   "scale [by | to] <scalefactor>", NULL); 
  return TCL_ERROR;
}
    
int text_cmd_rock(ClientData cd, Tcl_Interp *interp, int argc,
                            const char *argv[]) {

  VMDApp *app = (VMDApp *)cd;

  if (argc == 2) {
    if (!strupncmp(argv[1],"off",CMDLEN)) {
      app->scene_rockoff();
      return TCL_OK;
    } else if (!strupncmp(argv[1],"on",CMDLEN)) {
      Tcl_AppendResult(interp, "Totally, dude.", NULL); 
      return TCL_OK;
    }
  }

  if(argc >= 4 && argc <= 5) {
    char axis = (char)(tolower(argv[1][0]));
    float deg = (float) atof(argv[3]);
    int steps = (argc == 5 ? atoi(argv[4]) : -1);
    if (app->scene_rock(axis, deg, steps))
      return TCL_OK;
  }
  Tcl_AppendResult(interp, "rock usage:\n",
    "rock off -- stop continuous rotation of the scene\n", 
    "rock [x | y | z] by <increment> [steps] -- spin the scene\n",
    "     about the given axis by the given increment.  Optionally\n",
    "     specify the number of steps before reversing direction.",
    NULL); 
  return TCL_ERROR;
}

