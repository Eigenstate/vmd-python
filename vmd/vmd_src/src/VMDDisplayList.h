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
 *      $RCSfile: VMDDisplayList.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.44 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Display list data structure used to hold all of the rendering commands
 *   VMD generates and interprets in order to do its 3-D rendering.
 ***************************************************************************/

#ifndef VMDLINKEDLIST_H__
#define VMDLINKEDLIST_H__

#include <string.h>
#include "ResizeArray.h"
#include "Matrix4.h"

/// data structure describing user specified clipping planes
struct VMDClipPlane {
  float center[3]; ///< origin point in the clipping plane
  float normal[3]; ///< normal for orienting the clipping plane
  float color[3];  ///< color to use when doing solid/CSG clipping
  int mode;        ///< whether the clipping plane is on and what type

  /// default constructor 
  VMDClipPlane() {
    center[0] = center[1] = center[2] = 0;
    normal[0] = normal[1] = 0; normal[2] = 1;
    color[0] = color[1] = color[2] = 0.5;
    mode = 0;
  }
};

#define VMD_MAX_CLIP_PLANE 6

/// Controls for display of periodic boundary conditions
/// These flags are ORed together to create a value for set_pbc.
#define PBC_NONE   0x00  // don't draw any PBC images
#define PBC_X      0x01  // +X images
#define PBC_Y      0x02  // +Y images
#define PBC_Z      0x04  // +Z images
#define PBC_OPX    0x08  // -X images
#define PBC_OPY    0x10  // -Y images
#define PBC_OPZ    0x20  // -Z images
#define PBC_NOSELF 0x40  // set this flag to NOT draw the original image

/// Controls for display of molecule instances
/// These flags are ORed together to create a value for set_instance.
#define INSTANCE_NONE   0x0      // don't draw any instance images
#define INSTANCE_ALL    0x00ffff // draw all of the instance images
#define INSTANCE_NOSELF 0x01ffff // don't draw the original instance

/// Display list data structure used to hold all of the rendering commands
/// VMD generates and interprets in order to do its 3-D rendering.
class VMDDisplayList {
private:
  struct CommandHeader {
    int code;  ///< Display command code
    long size; ///< Display command size
  };

public:
  VMDDisplayList();   ///< constructor
  ~VMDDisplayList();  ///< destructor

  void *operator new(size_t);           ///< potentially shared mem allocation
  void operator delete(void *, size_t); ///< potentially shared mem allocation

  Matrix4 mat;                     ///< transform matrix for this display list
  unsigned long serial;            ///< globally unique serial# for cur contents
  int cacheskip;                   ///< display list cache skip flag
  int pbc;                         ///< periodic boundary condition flags
  int npbc;                        ///< number of times to replicate the image 
  Matrix4 transX, transY, transZ;  ///< how to create periodic images
  Matrix4 transXinv, transYinv, transZinv; ///< the inverse transforms
  int instanceset;                 ///< molecule instance flags
  ResizeArray<Matrix4> instances;  ///< molecule instance list

  //@{
  /// Material properties for this display list
  float ambient, specular, diffuse, shininess, mirror, opacity; 
  float outline, outlinewidth, transmode;
  int materialtag;  ///< used to avoid unnecessary material changes
  //@}

  VMDClipPlane clipplanes[VMD_MAX_CLIP_PLANE]; ///< user clip planes

  /// Append a new item.  Return space for the requested number of bytes
  void *append(int code, long size);

  // Reset and also free up any linked memory blocks.  The original block will 
  // not be freed; it doesn't go away until you delete the object.
  void reset_and_free(unsigned long newserial);

  /// return clip plane info for read-only access
  const VMDClipPlane *clipplane(int i) {
    if (i < 0 || i >= VMD_MAX_CLIP_PLANE) return NULL;
    return clipplanes+i;
  }

  //@{ 
  /// Set clip plane properties. Return true on success, false if failed.  
  // normals need not be normalized; it will be be normalized internally. 
  int set_clip_center(int i, const float *);
  int set_clip_normal(int i, const float *);
  int set_clip_color(int i, const float *);
  int set_clip_status(int i, int);
  //@}

  struct VMDLinkIter {
    char *ptr;    // pointer to current place in memory pool
    int ncmds;    // commands remaining in the list
  };

  /// get head of the list
  void first(VMDLinkIter *it) const {
    it->ptr = pool;
    it->ncmds = listsize;
  }

  /// get next item in the list
  int next(VMDLinkIter *it, char *&d) const {
    if (!(it->ncmds--)) return -1; // corresponds to DLASTCOMMAND
    CommandHeader *header = (CommandHeader *)(it->ptr);
    int code = header->code;
    long size = header->size;
    d = it->ptr + sizeof(CommandHeader);
    it->ptr += size;
    return code;
  }

private:
  // number of commands in the display list
  int listsize; 

  // size of memory pool
  unsigned long poolsize;

  // amount of memory pool consumed by the current display list
  unsigned long poolused;

  // our memory pool
  char *pool;
};    

#endif
