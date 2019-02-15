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
 *      $RCSfile: cmd_profile.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.5 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for profiling CPU/GPU hardware performance
 ***************************************************************************/

#include <tcl.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Benchmark.h"
#include "config.h"
#include "VMDApp.h"
#include "TclCommands.h"
#include "CUDAKernels.h"
#include "CUDAAccel.h"
#include "WKFThreads.h"
#include "ProfileHooks.h"

static void cmd_profile_usage(Tcl_Interp *interp) {
  Tcl_AppendResult(interp,
      "usage: profile <command> [args...]\n"
      "profile start\n",
      "profile stop\n",
      "profile push_range <tag string>\n",
      "profile pop_range\n",
      "(*) Only available in CUDA-enabled builds of VMD\n",
      NULL);
}

int text_cmd_profile(ClientData cd, Tcl_Interp *interp, int argc, const char *argv[]) {
#if defined(VMDCUDA) && defined(VMDNVTX)
  VMDApp *app = (VMDApp *)cd; // need VMDApp ptr GPU threadpool access
#endif

  if (argc == 1) {
    cmd_profile_usage(interp);
    return TCL_ERROR;
  }

  if (argc >= 2) {
#if !(defined(VMDCUDA) && defined(VMDNVTX))
    Tcl_AppendResult(interp, "CUDA Acceleration not available in this build", NULL);
    return TCL_ERROR;
#else
    if (!strupncmp(argv[1], "start", CMDLEN)) {
      PROFILE_START();
      Tcl_AppendResult(interp, "Starting profiling", NULL);
      return TCL_OK;
    } else if (!strupncmp(argv[1], "stop", CMDLEN)) {
      PROFILE_STOP();
      Tcl_AppendResult(interp, "Stopping profiling", NULL);
      return TCL_OK;
    } else if (!strupncmp(argv[1], "push_range", CMDLEN)) {
      if (argc >= 3) {
        PROFILE_PUSH_RANGE(argv[2], 2);
      }
      return TCL_OK;
    } else if (!strupncmp(argv[1], "pop_range", CMDLEN)) {
      PROFILE_POP_RANGE();
      return TCL_OK;
    } else {
      cmd_profile_usage(interp);
      return TCL_ERROR;
    }
#endif
  } else {
    cmd_profile_usage(interp);
    return TCL_ERROR;
  }
  
  // if here, everything worked out ok
  return TCL_OK;
}


