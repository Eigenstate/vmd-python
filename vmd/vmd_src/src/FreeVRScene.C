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
 *	$RCSfile: FreeVRScene.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.41 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   The FreeVR specific Scene.  It has to get information from the
 * shared memory arena, since the display lists are shared amoung
 * the different machines
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <freevr.h>

#include "FreeVRScene.h"
#include "FreeVRRoutines.h"
#include "DisplayDevice.h"
#include "Inform.h"
#include "utilities.h"
#include "VMDApp.h" // for VMDexit()

#define FREEVRUSERWLOCK 1

////////////////////////////  constructor  
FreeVRScene::FreeVRScene(VMDApp *vmdapp) : app(vmdapp) {
#if defined(FREEVRUSERWLOCK)
  msgInfo << "Creating R/W lock for shared mem access sync.\n" << sendmsg;
//  draw_rwlock = vrLockCreate(vmdapp->freevrcontext); // allocate an RW Lock
  draw_rwlock = vrLockCreate();    // allocate an RW Lock
  vrLockWriteSet(draw_rwlock);     // start out locked for writing
#else
  msgInfo << "Creating draw barrier for " << vrContext->config->num_windows 
          << " active FreeVR walls and master process.\n" << sendmsg;

  // setup the drawing barrier used to control slave rendering processes
  draw_barrier =(vmd_barrier_t *) malloc_from_FreeVR_memory(sizeof(vmd_barrier_t));
  if (draw_barrier == NULL) {
    msgErr << "CANNOT ALLOCATE SHARED MEMORY FOR FreeVRScene CLASS !!!"
           << sendmsg;
  }
  memset(draw_barrier, 0, sizeof(vmd_barrier_t));

  if (getenv("VMDFREEVRDRAWPROCS") != NULL) {
    vmd_thread_barrier_init_proc_shared(draw_barrier, atoi(getenv("VMDFREEVRDRAWPROCS")) + 1);
  } else {
    vmd_thread_barrier_init_proc_shared(draw_barrier, vrContext->config->num_windows + 1);
  }
#endif
}

////////////////////////////  destructor  
FreeVRScene::~FreeVRScene(void) {
#if defined(FREEVRUSERWLOCK)
  // XXX somehow we need to get the rendering slaves to 
  //     exit cleanly rather than getting stuck in mid-draw
  //     attempting to do various stuff when the world goes away. 
  vrLockFree(draw_rwlock); // delete the drawing rwlock
#else
  // free things allocated from shared memory
  free_to_FreeVR_memory(draw_barrier);
#endif
}

// Called by FreeVR, on each of the child rendering processes
void FreeVRScene::draw(DisplayDevice *display) {
#if defined(FREEVRUSERWLOCK)
  /* NOTE: the default build uses this RW-LOCK code */
  if (display->is_renderer_process()) {
    vrLockReadSet(draw_rwlock);
    Scene::draw(display);
    vrLockReadRelease(draw_rwlock);
  } else {
    vrLockWriteRelease(draw_rwlock);
    vmd_msleep(1); // give 'em 1 millisecond to draw before re-locking
    vrLockWriteSet(draw_rwlock); 
  }
#else
  // Barrier synchronization for all drawing processes and the master
  vmd_thread_barrier(draw_barrier, 0); // wait for start of drawing
  Scene::draw(display);                // draw the scene
  vmd_thread_barrier(draw_barrier, 0); // wait for end of drawing.
#endif
}

/* BS: this version makes use of the special vrRenderInfo conduit that passes properties from FreeVR into the render routine */
// Called by FreeVR, on each of the child rendering processes
void FreeVRScene::draw(DisplayDevice *display, vrRenderInfo *rendinfo) {
#if defined(FREEVRUSERWLOCK)
  /* NOTE: the default build uses this RW-LOCK code */
  if (display->is_renderer_process()) {
    //vrLockReadSet(draw_rwlock);	/* BS: this is hanging in the version that uses vrRenderInfo! */
    vrRenderTransformUserTravel(rendinfo);
    Scene::draw(display);
    //vrLockReadRelease(draw_rwlock);	/* BS: this is hanging in the version that uses vrRenderInfo! */
  } else {
    /* BS: a question -- why do non-rendering processes get here? */
    vrLockWriteRelease(draw_rwlock);
    vmd_msleep(1); // give 'em 1 millisecond to draw before re-locking
    vrLockWriteSet(draw_rwlock); 
  }
#else
  // Barrier synchronization for all drawing processes and the master
  vmd_thread_barrier(draw_barrier, 0); // wait for start of drawing
  Scene::draw(display);                // draw the scene
  vmd_thread_barrier(draw_barrier, 0); // wait for end of drawing.
#endif
}

// this is called by the parent!
int FreeVRScene::prepare() {
  // check if the ESC key is being pressed; if so, quit
  // note: THIS IS A HACK, just for Bill Sherman
  if(vrGet2switchValue(0)) {
    vrExit();
    app->VMDexit("Exiting due to FreeVR escape key being pressed", 10, 4);
  }

  return Scene::prepare(); // call regular scene prepare method
}

void *FreeVRScene::operator new(size_t s) {
  return malloc_from_FreeVR_memory(s);
}

void FreeVRScene::operator delete(void *p, size_t) {
  free_to_FreeVR_memory(p);
}

