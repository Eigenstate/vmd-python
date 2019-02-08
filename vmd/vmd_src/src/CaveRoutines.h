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
 *	$RCSfile: CaveRoutines.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.26 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * routines to get memory from and return memory to the 
 * CAVE shared memory arena
 ***************************************************************************/
#ifndef CAVEROUTINES_H
#define CAVEROUTINES_H

#include <malloc.h>

class Scene;
class DisplayDevice;

// check and set when the CAVE routines are available
int vmd_cave_is_initialized(void);
void vmd_set_cave_is_initialized(void);

/// use the already alloced memory as the memory arena and amalloc from it
void *malloc_from_CAVE_memory(size_t size);

/// return shared memory to the arena
void free_to_CAVE_memory(void *data);

/// get a large chunk of memory from the CAVE and remember it for future use
void grab_CAVE_memory(size_t megs);  

// global routines which call the Scene from the CAVE
// set up the graphics, called from CAVEInitApplication
void cave_gl_init_fn(void);

/// set static pointers to the Scene and DisplayDevice
void set_cave_pointers(Scene *, DisplayDevice *);

// call the renderer, on the first call this just counts the number of
// drawing processes which were started
void cave_renderer(void);
#endif

