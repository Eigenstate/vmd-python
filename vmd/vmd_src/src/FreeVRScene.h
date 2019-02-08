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
 *	$RCSfile: FreeVRScene.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.30 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   The FreeVR specific Scene.  It has to get information from the
 * shared memory arena, since the display lists are shared amoung
 * the different machines
 *
 ***************************************************************************/
#ifndef FreeVR_SCENE_H
#define FreeVR_SCENE_H

#include "Scene.h"
#include <freevr.h>
#include "FreeVRRoutines.h"
#include "FreeVRDisplayDevice.h" // for manipulating lights etc
#include "VMDThreads.h"          // for the barrier synchronization code

class VMDApp;

// This needs to grab shared memory for use in the FreeVR 
// environment.  It does it with one means.
//  1) use a FreeVRScene::operator new so that the internal
//      scene information is shared (get_disp_storage and
//      free_disp_storage)
// The shared memory is allocated through a global function,
//  new_from_FreeVR_memory.
// This must also call the left eye and right eye draws correctly

/// Scene subclass that allocates from a FreeVR shared memory arena,
/// and coordinates multiple rendering slave processes.
class FreeVRScene : public Scene {
private:
  VMDApp *app;

  /// shared memory barrier synchronization for draw processes
  vmd_barrier_t * draw_barrier;

  /// shared memory reader/writer locks for process synchronization
  vrLock draw_rwlock;

public:
  /// pass in VMDApp handle, needed for VMDexit
  FreeVRScene(VMDApp *);
  virtual ~FreeVRScene(void);
  
  /// Called by the FreeVR display function, copies shared mem variables to
  /// process-local variables, then calls draw() in the parent
  void draw(DisplayDevice *);
  void draw(DisplayDevice *, vrRenderInfo *rendinfo);	/* This version passes the rendering properties through to the render routine */
  
  /// Call the parent's prepare, then update the shared memory info
  virtual int prepare();

  /// Use FreeVR allocator-deallocator for FreeVRScene object
  void *operator new(size_t);
  void operator delete(void *, size_t);
};

#endif /* FREEVR_SCENE_H */

