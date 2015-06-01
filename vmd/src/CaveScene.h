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
 *	$RCSfile: CaveScene.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.48 $	$Date: 2010/12/16 04:08:07 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   The CAVE specific Scene.  It has to get information from the
 * shared memory arena, since the display lists are shared amoung
 * the different machines
 *
 ***************************************************************************/
#ifndef CAVE_SCENE_H
#define CAVE_SCENE_H

#include "Scene.h"
#if 1
#include <cave.h>
#else
#include <cave_ogl.h>
#endif
#include "CaveRoutines.h"
#include "CaveDisplayDevice.h" // for manipulating lights etc
#include "VMDThreads.h"        // for the barrier synchronization code

class VMDApp;

// This needs to grab shared memory for use in the CAVE
// environment.  It does it with one means.
//  1) use a CaveScene::operator new so that the internal
//      scene information is shared (get_disp_storage and
//      free_disp_storage)
// The shared memory is allocated through a global function,
//  new_from_CAVE_memory.
// This must also call the left eye and right eye draws correctly

/// Scene subclass that allocates from a CAVE shared memory arena,
/// and coordinates multiple rendering slave processes.
class CaveScene : public Scene {
private:
  VMDApp *app;

  /// shared memory barrier synchronization for draw processes
  vmd_barrier_t * draw_barrier; 

  /// shared memory reader/writer locks for process synchronization
  CAVELOCK draw_rwlock;  

public:
  /// pass in VMDApp handle, needed for VMDexit
  CaveScene(VMDApp *);
  virtual ~CaveScene(void);
  
  /// Called by the CAVE display function, copies shared mem variables to
  /// process-local variables, then calls draw() in the parent
  void draw(DisplayDevice *);
  
  /// Call the parent's prepare, then update the shared memory info
  virtual int prepare();

  /// Use CAVE allocator-deallocator for CaveScene object
  void *operator new(size_t);
  void operator delete(void *, size_t);
};

#endif /* CAVE_SCENE_H */

