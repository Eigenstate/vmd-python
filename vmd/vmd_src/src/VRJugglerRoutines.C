/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr VRJuggler patches contributed by Martijn Kragtwijk: m.kragtwijk@rug.nl
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: VRJugglerRoutines.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.4 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * routines to get memory from and return memory to the 
 * VRJuggler shared memory arena
 ***************************************************************************/
//#include <vrjuggler_ogl.h>

#include "VRJugglerRoutines.h"
#include "Inform.h"
#include "VMDApp.h"
#include "VRJugglerDisplayDevice.h"
#include "VRJugglerScene.h"

#define VRJugglerMAGICINIT 3142
static int vmd_vrjuggler_initialized_flag = 0; // global init state variable

int vmd_vrjuggler_is_initialized() {
  if (vmd_vrjuggler_initialized_flag == VRJugglerMAGICINIT) 
    return 1;
  else 
    return 0;
} 

void vmd_set_vrjuggler_is_initialized() {
  vmd_vrjuggler_initialized_flag = VRJugglerMAGICINIT;
}

// set up the graphics, called from VRJugglerInitApplication
void vrjuggler_gl_init_fn(void) {
  // nothing to do
}

// XXX globals to keep track of the display and scene data structures
static VRJugglerScene *vrjugglerscene;
static VRJugglerDisplayDevice *vrjugglerdisplay;

void set_vrjuggler_pointers(VRJugglerScene *scene, VRJugglerDisplayDevice *display) {
  msgInfo << "set_vrjuggler_pointers" << sendmsg;
  vrjugglerscene = scene;
  vrjugglerdisplay = display;
}

// call the child display renderer, and wait until they are done
void vrjuggler_renderer(void) {
  // msgInfo << "vrjuggler_renderer" << sendmsg;
	if (vrjugglerscene){
		vrjugglerscene->draw(vrjugglerdisplay);
		vrjugglerscene->draw_finished();
	} else {
		msgErr << "vrjuggler_renderer(): vrjugglerscene is NULL" << sendmsg;
	}
}

