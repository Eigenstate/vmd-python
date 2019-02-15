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
 *      $RCSfile: VMDDir.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.14 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Low level platform-specific code for scanning/querying directories.
 ***************************************************************************/

#include <stdio.h>

#if defined(_MSC_VER)
#include <windows.h>

/// Win32 directory traversal handle
typedef struct {
  HANDLE h;
  WIN32_FIND_DATA fd;
} VMDDIR;

#else
#include <dirent.h>

/// Unix directory traversal handle
typedef struct {
  DIR * d;
} VMDDIR;
#endif


VMDDIR * vmd_opendir(const char *);
char * vmd_readdir(VMDDIR *);
void vmd_closedir(VMDDIR *);
int vmd_file_is_executable(const char * filename);
