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
 *      $RCSfile: vmdmain.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.18 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Main program.
 *
 ***************************************************************************/
#include "vmd.h"
#include "ProfileHooks.h"

#if defined(ANDROID)
int VMDmain(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
  int mpienabled = 0;

  PROFILE_MAIN_THREAD(); // mark main VMD thread within profiler timelines

  PROFILE_PUSH_RANGE("VMD main() initialization", 4);

#if defined(VMDMPI)
  // so long as the caller has not disabled MPI, we initialize MPI when VMD
  // has been compiled to support it.
  if (getenv("VMDNOMPI") == NULL) {
    mpienabled = 1;
  }
#endif

  if (!VMDinitialize(&argc, &argv, mpienabled)) {
    return 0;
  }

  const char *displayTypeName = VMDgetDisplayTypeName();
  int displayLoc[2], displaySize[2];
  VMDgetDisplayFrame(displayLoc, displaySize);

  VMDApp *app = new VMDApp(argc, argv, mpienabled);

  if (!app->VMDinit(argc, argv, displayTypeName, displayLoc, displaySize)) {
    delete app;
    return 1;
  }

  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("VMD startup scripts", 1);

  // read various application defaults
  VMDreadInit(app);

  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("VMD process user script(s)", 1);

  // read user-defined startup files
  VMDreadStartup(app);

  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("VMD interactive event loop", 1);

  // main event loop
  do {
    // If we knew that there were no embedded python interpreter, we could
    // process Tcl events here, rather than within the VMD instance. 
#ifdef VMDTCL
    // Loop on the Tcl event notifier
    // while (Tcl_DoOneEvent(TCL_DONT_WAIT));
#endif

    // handle Fltk events
    VMDupdateFltk();

#if 0
    // take over the console
    if (vmd_check_stdin()) {
      app->process_console();
    }
#endif

  } while(app->VMDupdate(VMD_CHECK_EVENTS));

  PROFILE_POP_RANGE();

  // end of program
  delete app;
  VMDshutdown(mpienabled);

  return 0;
}

