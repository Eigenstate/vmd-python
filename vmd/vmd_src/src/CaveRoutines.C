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
 *	$RCSfile: CaveRoutines.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.38 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * routines to get memory from and return memory to the 
 * CAVE shared memory arena
 ***************************************************************************/

#include <cave_ogl.h>

#include "CaveRoutines.h"
#include "Inform.h"
#include "VMDApp.h"
#include "CaveDisplayDevice.h"
#include "CaveScene.h"

#if defined(__irix)
static void *shared_CAVE_memory_arena = NULL;
#endif

#define CAVEMAGICINIT 3141
static int vmd_cave_initialized_flag = 0; // global init state variable

int vmd_cave_is_initialized() {
  if (vmd_cave_initialized_flag == CAVEMAGICINIT) 
    return 1;
  else 
    return 0;
} 

void vmd_set_cave_is_initialized() {
  vmd_cave_initialized_flag = CAVEMAGICINIT;
}

void *malloc_from_CAVE_memory(size_t size) {
#if defined(__irix)
  // Allocate memory from our shared memory arena
  if (shared_CAVE_memory_arena == NULL) {
     // this should be fun.
     msgErr << "Shared CAVE memory not allocated.  Prepare to crash and burn."
            << sendmsg;
     return NULL;
  } else {
    // get memory from the CAVE shared memory arena
    return amalloc(size, shared_CAVE_memory_arena); 
  }
#else
  // Allocate from shared CAVE pool
  void *retval = CAVEMalloc(size);
  if (!retval)
    // this should be fun.
    msgErr << "Not enough shared CAVE memory. Prepare to crash and burn."
           << sendmsg;
  return retval;
#endif
}

void free_to_CAVE_memory(void *data) {
#if defined(__irix)
  // Free memory from our shared memory arena
  afree(data, shared_CAVE_memory_arena);
#else
  // Free from CAVE shared memory pool
  CAVEFree(data);
#endif
}

// get megs o' memory from the CAVE, and create the arena
// Warning:  Don't make me do this twice.
void grab_CAVE_memory(size_t megs) {
#if defined(__irix)
  // Make our own shared memory arena using the CAVE to set it up,
  // done on IRIX due to old revs of the CAVE library having bugs etc.
  size_t size = (megs>1?megs:1) * 1024L * 1024L;
  shared_CAVE_memory_arena = CAVEUserSharedMemory(size);

  if (!shared_CAVE_memory_arena)
    msgErr << "Bad juju in the arena.  We're gonna die!" << sendmsg;
  else
    msgInfo <<  "Created arena." << sendmsg;
#else
  // Trust the CAVE library to setup enough shared mem for subsequent calls
  size_t size = (megs>1?megs:1) * 1024L * 1024L;
  CAVESetOption(CAVE_SHMEM_SIZE, size);
#endif
}


// set up the graphics, called from CAVEInitApplication
void cave_gl_init_fn(void) {
  // nothing to do
}

// XXX globals to keep track of the display and scene data structures
static Scene *cavescene;
static DisplayDevice *cavedisplay;

void set_cave_pointers(Scene *scene, DisplayDevice *display) {
  cavescene = scene;
  cavedisplay = display;
}

// call the child display renderer, and wait until they are done
void cave_renderer(void) {
  cavescene->draw(cavedisplay);
}

