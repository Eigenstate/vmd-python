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
 *      $RCSfile: VRJugglerRoutines.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * routines to get memory from and return memory to the 
 * VRJuggler shared memory arena
 ***************************************************************************/
#ifndef VRJugglerROUTINES_H
#define VRJugglerROUTINES_H

//#include <malloc.h>

class VRJugglerScene;
class VRJugglerDisplayDevice;

// check and set when the VRJuggler routines are available
int vmd_vrjuggler_is_initialized(void);
void vmd_set_vrjuggler_is_initialized(void);

/// use the already alloced memory as the memory arena and amalloc from it
//void *malloc_from_VRJuggler_memory(size_t size);

/// return shared memory to the arena
//void free_to_VRJuggler_memory(void *data);

/// get a large chunk of memory from the VRJuggler and remember it for future use
//void grab_VRJuggler_memory(int megs);  

// global routines which call the Scene from the VRJuggler
// set up the graphics, called from VRJugglerInitApplication
void vrjuggler_gl_init_fn(void);

/// set static pointers to the Scene and DisplayDevice
void set_vrjuggler_pointers(VRJugglerScene *, VRJugglerDisplayDevice *);

// call the renderer, on the first call this just counts the number of
// drawing processes which were started
void vrjuggler_renderer(void);
#endif

