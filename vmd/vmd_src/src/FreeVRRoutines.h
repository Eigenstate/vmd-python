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
 *	$RCSfile: FreeVRRoutines.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.15 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * routines to get memory from and return memory to the 
 * FreeVR shared memory arena
 ***************************************************************************/
#ifndef FREEVRROUTINES_H
#define FREEVRROUTINES_H

#include <malloc.h>

#include <freevr.h>

class Scene;
class DisplayDevice;

/// use the already alloced memory as the memory arena and amalloc from it
void *malloc_from_FreeVR_memory(size_t size);

/// return shared memory to the arena
void free_to_FreeVR_memory(void *data);

/// get a large chunk of memory from FreeVR and remember it for future use
void grab_FreeVR_memory(size_t megs);  

// global routines which call the Scene from the FreeVR 
// set up the graphics, called from FreeVRInitApplication
void freevr_gl_init_fn(void);

/// set static pointers to the Scene and DisplayDevice
void set_freevr_pointers(Scene *, DisplayDevice *);

// call the renderer, on the first call this just counts the number of
// drawing processes which were started
//void freevr_renderer(void);
void freevr_renderer(DisplayDevice *display, void *rendinfo);
#endif

