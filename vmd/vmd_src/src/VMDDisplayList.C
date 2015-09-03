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
 *      $RCSfile: VMDDisplayList.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.37 $      $Date: 2011/06/01 19:35:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Display list data structure used to hold all of the rendering commands
 *   VMD generates and interprets in order to do its 3-D rendering. 
 ***************************************************************************/

#include "VMDDisplayList.h"
#include "VMDApp.h"
#include "Matrix4.h"
#include "Inform.h"

// required word size alignment in bytes, 16 bytes is large enough for 
// all currently known systems.
#define ALLOC_ALIGNMENT 16 
#define ALLOC_ALIGNMASK (ALLOC_ALIGNMENT - 1)

// Initial number of bytes in the display list memory block; not allocated
// until the list receives its first command.

// Typical "empty" representations need about 64 bytes to hold their 
// initial material on, line thickness and resolution parameters.
#define BASE_DISPLAYLIST_SIZE  64

// once the representation goes past being empty, it's likely going to need
// a lot more space in subsequent allocations, so the next growth step is
// much bigger to reduce the total number of calls without creating too much
// memory fragmentation for simple representations.
#define GROWN_DISPLAYLIST_SIZE 16384

void *VMDDisplayList::operator new(size_t n) {
  return vmd_alloc(n);
}

void VMDDisplayList::operator delete(void *p, size_t) {
  vmd_dealloc(p);
}

VMDDisplayList::VMDDisplayList() {
  materialtag = 0;
  serial=0;
  cacheskip=0;
  pbc = PBC_NONE;
  npbc = 1;

  // Begin with no memory allocated for the pool; many display lists,
  // like those for DrawMolecule, DrawForce, and MoleculeGraphics, spend
  // their whole lives with no commands, so it's wasteful to allocate
  // anything for them.
  // 
  // XXX In fact, DrawForce and MoleculeGraphics are creating single-element
  // display lists that contain nothing but DMATERIALON.  This should be
  // fixed; better might even be to not create these Displayables at all 
  // until/unless they are needed.  
  listsize = 0;
  poolsize = 0;
  poolused = 0;
  pool = NULL;
}

VMDDisplayList::~VMDDisplayList() {
  if (pool) vmd_dealloc(pool);
}

void *VMDDisplayList::append(int code, long size) {
  unsigned long neededBytes = (sizeof(CommandHeader) + size + ALLOC_ALIGNMASK) & ~(ALLOC_ALIGNMASK);
  if (neededBytes + poolused > poolsize) {
    unsigned long newsize;
    if (!pool) {
      newsize = (neededBytes < BASE_DISPLAYLIST_SIZE) ? 
        BASE_DISPLAYLIST_SIZE : neededBytes;
    } else {
      newsize = (unsigned long) (1.2f * (poolsize + neededBytes));
      if (newsize < GROWN_DISPLAYLIST_SIZE)
        newsize = GROWN_DISPLAYLIST_SIZE;
    }
//printf("bumping displist size from %d to %d to handle %d from cmd %d\n", poolsize, newsize, size, code);
    char *tmp = (char *) vmd_resize_alloc(pool, poolused, newsize);
    // check for failed allocations
    if (!tmp) {
      msgErr << "Failed to increase display list memory pool size, system out of memory" << sendmsg;
      msgErr << "  Previous pool size: " << ((unsigned long) (poolsize / (1024*1024))) << "MB" << sendmsg;
      msgErr << "  Requested pool size: " << ((unsigned long) (newsize / (1024*1024))) << "MB" << sendmsg;
      return NULL;
    }
    poolsize = newsize;
    pool = tmp;
  }
  // store header and size of header + data into pool
  CommandHeader *header = (CommandHeader *)(pool + poolused);
  poolused += neededBytes;

  header->code = code;
  header->size = neededBytes;

  // return pointer to data following header
  ++listsize;
  return header+1;
}

void VMDDisplayList::reset_and_free(unsigned long newserial) {
  // if we used less than 1/4 of the total pool size, trim the pool
  // back to empty so that we don't hog memory.
  if (poolsize > BASE_DISPLAYLIST_SIZE && 
      poolused / (float)poolsize < 0.25f) {
    vmd_dealloc(pool);
    pool = NULL;
    poolsize = 0;
  }
  poolused = 0;
  listsize = 0;
  serial = newserial;
}

int VMDDisplayList::set_clip_normal(int i, const float *normal) { 
  if (i < 0 || i >= VMD_MAX_CLIP_PLANE) return 0;
  float length = norm(normal);
  if (!length) return 0;
  clipplanes[i].normal[0] = normal[0]/length;
  clipplanes[i].normal[1] = normal[1]/length;
  clipplanes[i].normal[2] = normal[2]/length;
  return 1;
}

int VMDDisplayList::set_clip_center(int i, const float *center) { 
  if (i < 0 || i >= VMD_MAX_CLIP_PLANE) return 0;
  clipplanes[i].center[0] = center[0];
  clipplanes[i].center[1] = center[1];
  clipplanes[i].center[2] = center[2];
  return 1;
}

int VMDDisplayList::set_clip_color(int i, const float *color) { 
  if (i < 0 || i >= VMD_MAX_CLIP_PLANE) return 0;
  clipplanes[i].color[0] = color[0];
  clipplanes[i].color[1] = color[1];
  clipplanes[i].color[2] = color[2];
  return 1;
}

int VMDDisplayList::set_clip_status(int i, int mode) {
  if (i < 0 || i >= VMD_MAX_CLIP_PLANE) return 0;
  clipplanes[i].mode = mode;
  return 1;
}

  
