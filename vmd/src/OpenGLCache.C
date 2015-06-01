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
 *      $RCSfile: OpenGLCache.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $      $Date: 2010/12/16 04:08:25 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class to manage caching of OpenGL-related resources and handles
 ***************************************************************************/

#include <stdio.h>
#include "OpenGLCache.h"

OpenGLCache::OpenGLCache() {
  cache = NULL; // initialize the cache to be empty
}

OpenGLCache::~OpenGLCache() {
  idlink *tmp;
  for (idlink *cur = cache; cur; cur=tmp) {
    tmp = cur->next;
    delete cur;
  }
}

void OpenGLCache::encache(unsigned long id, GLuint tag) {
  cache = new idlink(id, tag, cache);
}

void OpenGLCache::markUnused() {
  for (idlink *cur = cache; cur; cur= cur->next) {
    cur->used = 0;
  }
}

GLuint OpenGLCache::markUsed(unsigned long id) {
  for (idlink *lnk = cache; lnk; lnk = lnk->next) {
    if (lnk->id == id) {
      lnk->used = 1;
      return lnk->gltag;
    }
  }
  return GLCACHE_FAIL; // return 0 for default OpenGL tag.
}

GLuint OpenGLCache::deleteUnused() {
  idlink *prev=NULL, *cur = cache;
  while (cur) {
    if (!cur->used) {
      GLuint tag = cur->gltag;
      if (prev) { // link previous to next before deleting
        prev->next = cur->next;
        delete cur;
      } else {    // we're deleting the head of the list
        idlink *tmp = cur->next;
        delete cur;
        cache = tmp;
      }
      return tag;
    }
    prev = cur;
    cur = cur->next;
  }
  return GLCACHE_FAIL; // failed to find any unused items
}

