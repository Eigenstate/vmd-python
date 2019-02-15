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
 *	$RCSfile: FreeVRRoutines.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.24 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * routines to get memory from and return memory to the 
 * FreeVR shared memory arena
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "Inform.h"
#include "FreeVRRoutines.h"
#include "VMDApp.h"
#include "FreeVRDisplayDevice.h"
#include "FreeVRScene.h"

#include <freevr.h>

void *malloc_from_FreeVR_memory(size_t size) {
  return vrShmemAlloc(size);
}

void free_to_FreeVR_memory(void *data) {
  vrShmemFree(data);
}

// get megs o' memory from FreeVR, and create the arena
// Warning:  Don't make me do this twice.
void grab_FreeVR_memory(size_t megs) {
  size_t size = ((megs>1) ? megs : 1) * 1024L * 1024L;
  size_t sz=0;

  while (((sz = vrShmemInit(size)) == 0) && (size > 64*1024*1024)) {
    msgErr << "Failed to create FreeVR arena of size " 
           << (size / (1024*1024)) 
           << ", reducing allocation by half." << sendmsg;
    size >>= 1; // cut allocation in half
  }
 
  if (sz == 0) 
    msgErr << "Failed to create FreeVR arena.  We're gonna die!" << sendmsg;
  else
    msgInfo << "Created arena, size " << (sz / (1024*1024)) 
            << "MB." << sendmsg;
}


// set up the graphics, called from FreeVRInitApplication
void freevr_gl_init_fn(void) {
}

static FreeVRScene *freevrscene;
static DisplayDevice *freevrdisplay;

void set_freevr_pointers(Scene *scene, DisplayDevice *display) {
  freevrscene = (FreeVRScene *)scene;
  freevrdisplay = display;
}

// call the child display renderer, and wait until they are done
void freevr_renderer(DisplayDevice *display, void *rendinfo) {
  //printf((char*)"hey in freevr_renderer -- rendinfo = %p\n", rendinfo);
#if 1
  freevrscene->draw(freevrdisplay, (vrRenderInfo *)rendinfo);	/* BS: the use of this "freevrdisplay" global isn't necessary now that we've got it as an argument. */
#else
  freevrscene->draw(freevrdisplay);	/* BS: the use of this "freevrdisplay" global isn't necessary now that we've got it as an argument. */
#endif
}

