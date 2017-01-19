
/* 
 * OpenMP multithreading setup util for VMD tcl plugins.
 * 
 * Copyright (c) 2006-2009 akohlmey@cmm.chem.upenn.edu
 */

#ifndef OPENMP_UTIL_H
#define OPENMP_UTIL_H

#include <tcl.h>

#if defined(_OPENMP)

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static int nthr = 0;

#endif

/* this results in an empty function without OpenMP */
static void check_thread_count(Tcl_Interp *interp, const char *name)
{
#if defined(_OPENMP)
  char *forcecount, buffer[256];
  int newcpucount = nthr;

  /* Handle VMD's way to allow the user to override the number 
   * of CPUs for use in scalability testing, debugging, etc. */
  forcecount = getenv("VMDFORCECPUCOUNT");
  if (forcecount != NULL) {
    if (sscanf(forcecount, "%d", &newcpucount) != 1) {
      newcpucount=1;
    }
    omp_set_num_threads(newcpucount);
  }

/* first time setup */    
  if (newcpucount < 1) {
#pragma omp parallel shared(nthr)
    {
#pragma omp master
      { newcpucount = omp_get_num_threads(); }
    }
  }
  
  /* print a message to the console, whenever the number of threads changes. */
  if (nthr!=newcpucount) {
    nthr=newcpucount;
    sprintf(buffer,"vmdcon -info \"'%s' will use %d thread%s through OpenMP.\"\n", name, nthr, (nthr>1)? "s":"");
    Tcl_Eval(interp,buffer);
  }
#endif
}

#endif /* OPENMP_UTIL_H */
