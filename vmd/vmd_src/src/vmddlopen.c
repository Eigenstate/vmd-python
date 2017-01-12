/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: vmddlopen.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.20 $      $Date: 2016/11/28 03:05:08 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Routines for loading dynamic link libraries and shared object files
 *   on various platforms, abstracting from machine dependent APIs.
 *
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 *
 ***************************************************************************/

#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include "vmddlopen.h"

#if defined(__hpux)

#include <dl.h>
#include <errno.h>
#include <string.h>

void *vmddlopen( const char *path) {
    void *ret;
    ret = shl_load( path, BIND_IMMEDIATE | BIND_FIRST | BIND_VERBOSE, 0);
    return ret;
}

int vmddlclose( void *handle ) {
    return shl_unload( (shl_t) handle );
}

void *vmddlsym( void *handle, const char *sym ) {
    void *value=0;

    if ( shl_findsym( (shl_t*)&handle, sym, TYPE_UNDEFINED, &value ) != 0 ) 
	return 0;
    return value;
}

const char *vmddlerror( void  ) {
    return strerror( errno );
}

#elif 0 && defined(__APPLE__)
/*
 * This is only needed for MacOS X version 10.3 or older
 */
#include <mach-o/dyld.h>

void *vmddlopen( const char *path) {
  NSObjectFileImage image;
  NSObjectFileImageReturnCode retval;
  NSModule module;

  retval = NSCreateObjectFileImageFromFile(path, &image);
  if (retval != NSObjectFileImageSuccess)
    return NULL;

  module = NSLinkModule(image, path,
            NSLINKMODULE_OPTION_BINDNOW | NSLINKMODULE_OPTION_PRIVATE
            | NSLINKMODULE_OPTION_RETURN_ON_ERROR);
  return module;  /* module will be NULL on error */
}

int vmddlclose( void *handle ) {
  NSModule module = (NSModule *)handle;
  NSUnLinkModule(module, NSUNLINKMODULE_OPTION_NONE);
  return 0;
}

void *vmddlsym( void *handle, const char *symname ) {
  char *realsymname;
  NSModule module;
  NSSymbol sym;
  /* Hack around the leading underscore in the symbol name */
  realsymname = (char *)malloc(strlen(symname)+2);
  strcpy(realsymname, "_");
  strcat(realsymname, symname);
  module = (NSModule)handle;
  sym = NSLookupSymbolInModule(module, realsymname);
  free(realsymname);
  if (sym) 
    return (void *)(NSAddressOfSymbol(sym));
  return NULL;
}

const char *vmddlerror( void  ) {
  NSLinkEditErrors c;
  int errorNumber;
  const char *fileName;
  const char *errorString = NULL;
  NSLinkEditError(&c, &errorNumber, &fileName, &errorString);
  return errorString;
}

#elif defined(_MSC_VER)

#include <windows.h>

void *vmddlopen(const char *fname) {
  return (void *)LoadLibrary(fname);
}

const char *vmddlerror(void) {
  static CHAR szBuf[80]; 
  DWORD dw = GetLastError(); 
 
  sprintf(szBuf, "vmddlopen failed: GetLastError returned %u\n", dw); 
  return szBuf;
}

void *vmddlsym(void *h, const char *sym) {
  return (void *)GetProcAddress((HINSTANCE)h, sym);
}

int vmddlclose(void *h) {
  /* FreeLibrary returns nonzero on success */
  return !FreeLibrary((HINSTANCE)h);
}

#else

/* All remaining platforms (not Windows, HP-UX, or MacOS X <= 10.3) */
#include <dlfcn.h>

void *vmddlopen(const char *fname) {
  return dlopen(fname, RTLD_NOW);
}
const char *vmddlerror(void) {
  return dlerror();
}
void *vmddlsym(void *h, const char *sym) {
  return dlsym(h, sym);
}
int vmddlclose(void *h) {
  return dlclose(h);
}

#endif 

