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
 *      $RCSfile: OpenGLCache.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $       $Date: 2010/12/16 04:08:25 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class to manage caching of OpenGL-related resources and handles
 ***************************************************************************/
#ifndef OPENGLCACHE_H
#define OPENGLCACHE_H

#include <stdlib.h>
#include "OpenGLExtensions.h"

#define GLCACHE_FAIL 0

/// Class to manage caching of OpenGL-related resources and handles
/// such as display lists, textures, vertex buffer objects, etc.
/// The IDs being stored are sparse unsigned long integer keys
class OpenGLCache {
private:
  struct idlink {
    idlink *next;           ///< next list item
    int used;               ///< whether the item is "used" or not 
    const unsigned long id; ///< unique serial number or ID from VMD
    const GLuint gltag;     ///< matching OpenGL handle/tag/resource specifier

    idlink(unsigned long theid, GLuint tag, idlink *thenext)
    : next(thenext), used(1), id(theid), gltag(tag) {}
  };

  idlink * cache;           ///< linked list of all tracked IDs

public:
  OpenGLCache();
  ~OpenGLCache(); 

  void encache(unsigned long id, GLuint tag); /// Add ID to the cache 
  void markUnused();                 ///< mark everything unused for new frame
  GLuint markUsed(unsigned long id); ///< mark given id used and return tag
  GLuint deleteUnused();             ///< Delete first unused ID and return tag 
};

#endif
