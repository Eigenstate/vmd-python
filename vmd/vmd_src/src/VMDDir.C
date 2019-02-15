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
 *      $RCSfile: VMDDir.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Low level platform-specific directory read/scan code
 ***************************************************************************/

#include "VMDDir.h"
#include <string.h>
#include <stdlib.h>

#define VMD_FILENAME_MAX 1024

#if defined(_MSC_VER) 

/* Windows version */

VMDDIR * vmd_opendir(const char * filename) {
  VMDDIR * d;
  char dirname[VMD_FILENAME_MAX];

  strcpy(dirname, filename);
  strcat(dirname, "\\*");
  d = (VMDDIR *) malloc(sizeof(VMDDIR));
  if (d != NULL) {
    d->h = FindFirstFile(dirname, &(d->fd));
    if (d->h == ((HANDLE)(-1))) {
      free(d);
      return NULL;
    }
  }
  return d;
}

char * vmd_readdir(VMDDIR * d) {
  if (FindNextFile(d->h, &(d->fd))) {
    return d->fd.cFileName; 
  }
  return NULL;     
}

void vmd_closedir(VMDDIR * d) {
  if (d->h != NULL) {
    FindClose(d->h);
  }
  free(d);
}


int vmd_file_is_executable(const char * filename) {
  FILE * fp = NULL;
  char *newfilename = (char *) malloc(strlen(filename)+1);

  if (newfilename != NULL) {
    char *ns = newfilename;
    const char *s = filename;
  
    // windows chokes on filenames containing " characters, so we remove them
    while ((*s) != '\0') {
      if ((*s) != '\"') {
        *ns = *s;
        ns++;
      }
      s++;
    }
    *ns = '\0';

    fp=fopen(newfilename, "rb");
    free(newfilename);

    if (fp != NULL) {
      fclose(fp);
      return 1;
    }
  }

  return 0;
} 

#else

/* Unix version */

#include <sys/types.h>
#include <sys/stat.h>

VMDDIR * vmd_opendir(const char * filename) {
  VMDDIR * d;

  d = (VMDDIR *) malloc(sizeof(VMDDIR));
  if (d != NULL) {
    d->d = opendir(filename);
    if (d->d == NULL) {
      free(d);
      return NULL;
    }
  }

  return d;
}

char * vmd_readdir(VMDDIR * d) {
  struct dirent * p;
  if ((p = readdir(d->d)) != NULL) {
    return p->d_name;
  }

  return NULL;     
}

void vmd_closedir(VMDDIR * d) {
  if (d->d != NULL) {
    closedir(d->d);
  }
  free(d);
}

int vmd_file_is_executable(const char * filename) {
  struct stat buf;
  memset(&buf, 0, sizeof(buf));

  if (!stat(filename, &buf) &&
           ((buf.st_mode & S_IXUSR) ||
            (buf.st_mode & S_IXGRP) ||
            (buf.st_mode & S_IXOTH)) &&
            (buf.st_mode & S_IFREG)) {
    return 1;
  }

  return 0;
}

#endif




