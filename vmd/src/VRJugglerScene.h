/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr VRJuggler patches contributed by Martijn Kragtwijk: m.kragtwijk@rug.nl
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: VRJugglerScene.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $       $Date: 2010/12/16 04:08:48 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   The VRJuggler specific Scene.  It has to get information from the
 * shared memory arena, since the display lists are shared amoung
 * the different machines
 ***************************************************************************/
#ifndef VRJUGGLER_SCENE_H
#define VRJUGGLER_SCENE_H

#include "Scene.h"
#include "Matrix4.h"
//#include <vrjuggler.h>
#include "VRJugglerRoutines.h"
//#include "VRJugglerDisplayDevice.h" // for manipulating lights etc
//#include "VMDThreads.h"        // for the barrier synchronization code
namespace vrj
{
	class Kernel;
}
class VMDApp;
class M_VRJapp;
class DisplayDevice;
// This needs to grab shared memory for use in the VRJuggler
// environment.  It does it with one means.
//  1) use a VRJugglerScene::operator new so that the internal
//      scene information is shared (get_disp_storage and
//      free_disp_storage)
// The shared memory is allocated through a global function,
//  new_from_VRJuggler_memory.
// This must also call the left eye and right eye draws correctly

/// Scene subclass that allocates from a VRJuggler shared memory arena,
/// and coordinates multiple rendering slave processes.

using namespace vrj;

class VRJugglerScene : public Scene {
public:
  VMDApp *app;

  /// shared memory barrier synchronization for draw processes
  //vmd_barrier_t * draw_barrier; 

  /// shared memory reader/writer locks for process synchronization
  //VRJugglerLOCK draw_rwlock;  

public:
  /// pass in VMDApp handle, needed for VMDexit
  VRJugglerScene(VMDApp *);

  void init();

  virtual ~VRJugglerScene(void);
  
  
  virtual void draw(DisplayDevice *);// Called by the VRJuggler display function
  virtual void draw_finished(void);// Called by the VRJuggler display function
  
  /// Call the parent's prepare, then update the shared memory info
  virtual int prepare();

  // wait until the Kernel is done
  void waitForKernelStop();

  /// Use VRJuggler allocator-deallocator for VRJugglerScene object
  //void *operator new(size_t);
  //void operator delete(void *, size_t);
	   
  // send a string to the shared data
  void appendCommand(const char* str);

  ///////////////// Martijn

  Kernel* kernel;
  M_VRJapp* application;   

  // Wanda data
  void getWandXYZ(float& x, float& y, float& z);
  void getWandRotMat(Matrix4 &rot); 
  bool getWandButton(unsigned nr);

};

#endif /* VRJUGGLER_SCENE_H */

