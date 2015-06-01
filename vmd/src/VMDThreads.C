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
 *      $RCSfile: VMDThreads.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.89 $       $Date: 2012/10/01 22:28:19 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * VMDThreads.C - code for spawning threads on various platforms.
 *                Code donated by John Stone, john.stone@gmail.com 
 *                This code was originally written for the
 *                Tachyon Parallel/Multiprocessor Ray Tracer. 
 *                Improvements have been donated by Mr. Stone on an 
 *                ongoing basis. 
 *
 ***************************************************************************/

#if 0
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* If compiling on Linux, enable the GNU CPU affinity functions in both */
/* libc and the libpthreads                                             */
/* #define VMDTHR_COMPAT_GLIBC232 1 */
#if defined(__linux)
#if defined(VMDTHR_COMPAT_GLIBC232)
#include <gnu/libc-version.h>
#endif
#define _GNU_SOURCE 1
#endif

#include "VMDThreads.h"

#ifdef _MSC_VER
#include <windows.h>
#include <winbase.h>
#endif

/* needed for call to sysconf() */
#if defined(__sun) || defined(__linux) || defined(__irix) || defined(_CRAY) || defined(__osf__) || defined(_AIX)
#include<unistd.h>
#endif

#if defined(__APPLE__) && defined(VMDTHREADS)
#if 1
#include <sys/types.h>
#include <sys/sysctl.h>     /**< OSX >= 10.7 queries sysctl() for CPU count */
#else
#include <Carbon/Carbon.h>  /**< Deprecated Carbon APIs for Multiprocessing */
#endif
#endif

#if defined(__hpux)
#include <sys/mpctl.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif


int vmd_thread_numphysprocessors(void) {
  int a=1;

#ifdef VMDTHREADS
#if defined(__APPLE__)
#if 1
  int rc;
  int mib[2];
  u_int miblen;
  size_t alen = sizeof(a);
  mib[0] = CTL_HW;
  mib[1] = HW_AVAILCPU;
  miblen = 2;
  rc = sysctl(mib, miblen, &a, &alen, NULL, 0); /**< Number of active CPUs */
  if (rc < 0) {
    perror("Error during sysctl() query for CPU count");
    a = 1;
  }
#else
  a = MPProcessorsScheduled();       /**< Number of active/running CPUs */
#endif
#endif

#ifdef _MSC_VER
  struct _SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  a = sysinfo.dwNumberOfProcessors; /* total number of CPUs */
#endif /* _MSC_VER */

#if defined(_CRAY)
  a = sysconf(_SC_CRAY_NCPU);
#endif

#if defined(ANDROID)
  /* Android toggles cores on/off according to system activity, */
  /* thermal management, and battery state.  For now, we will   */
  /* use as many threads as the number of physical cores since  */
  /* the number that are online may vary even over a 2 second   */
  /* time window.  We will likely have this issue on other      */
  /* platforms as power management becomes more important...    */

  /* use sysconf() for initial guess, although it produces incorrect    */
  /* results on the older android releases due to a bug in the platform */
  a = sysconf(_SC_NPROCESSORS_CONF); /**< Number of physical CPU cores  */

  /* check CPU count by parsing /sys/devices/system/cpu/present and use */
  /* whichever result gives the larger CPU count...                     */
  {
    int rc=0, b=1, i=-1, j=-1;
    FILE *ifp;

    ifp = fopen("/sys/devices/system/cpu/present", "r");
    if (ifp != NULL) {
      rc = fscanf(ifp, "%d-%d", &i, &j); /* read and interpret line */
      fclose(ifp);

      if (rc == 2 && i == 0) {
        b = j+1; /* 2 or more cores exist */
      }
    }

    /* return the greater CPU count result... */
    a = (a > b) ? a : b;
  }
#else
#if defined(__sun) || defined(__linux) || defined(__osf__) || defined(_AIX)
  a = sysconf(_SC_NPROCESSORS_ONLN); /**< Number of active/running CPUs */
#endif /* SunOS, and similar... */
#endif /* Android */

#if defined(__irix)
  a = sysconf(_SC_NPROC_ONLN); /* number of active/running CPUs */
#endif /* IRIX */

#if defined(__hpux)
  a = mpctl(MPC_GETNUMSPUS, 0, 0); /* total number of CPUs */
#endif /* HPUX */
#endif /* VMDTHREADS */

  return a;
}

int vmd_thread_numprocessors(void) {
  int a=1;

#ifdef VMDTHREADS
  /* Allow the user to override the number of CPUs for use */
  /* in scalability testing, debugging, etc.               */
  char *forcecount = getenv("VMDFORCECPUCOUNT");
  if (forcecount != NULL) {
    if (sscanf(forcecount, "%d", &a) == 1) {
      return a; /* if we got a valid count, return it */
    } else {
      a=1;      /* otherwise use the real available hardware CPU count */
    }
  }

  /* otherwise return the number of physical processors currently available */
  a = vmd_thread_numphysprocessors();

  /* XXX we should add checking for the current CPU affinity masks here, */
  /* and return the min of the physical processor count and CPU affinity */
  /* mask enabled CPU count.                                             */
#endif /* VMDTHREADS */

  return a;
}

#if defined(__linux) && defined(VMDTHR_COMPAT_GLIBC232)
int glib232compat_getaffinity(pid_t        pid,
                              unsigned int cpusetsize,
                              cpu_set_t*   mask,
                              int          (*p)(pid_t,cpu_set_t*)) {
  typedef int(*pt2Func)(pid_t,unsigned int,cpu_set_t*);
  if (strncmp(gnu_get_libc_version(),"2.3.2",6) == 0){
    return (*p)(pid,mask);
  }
  return ((pt2Func)(*p))(pid,cpusetsize,mask);
}

int glib232compat_setaffinity(pid_t        pid,
                              unsigned int cpusetsize,
                              cpu_set_t*   mask,
                              int          (*p)(pid_t,const cpu_set_t*)) {
  typedef int(*pt2Func)(pid_t,unsigned int,cpu_set_t*);
  if(strncmp(gnu_get_libc_version(),"2.3.2",6) == 0){
    return (*p)(pid,mask);
  }
  return ((pt2Func)(*p))(pid,cpusetsize,mask);
}
#endif /* __linux && VMDTHR_COMPAT_GLIBC232 */

				

int * vmd_cpu_affinitylist(int *cpuaffinitycount) {
  int *affinitylist = NULL;
  *cpuaffinitycount = -1; /* return count -1 if unimplemented or err occurs */

/* Win32 process affinity mask query */
/* XXX untested, but based on the linux code, may work with a few tweaks */
#if 0 && defined(_MSC_VER)
  HANDLE myproc = GetCurrentProcess(); /* returns a psuedo-handle */
  DWORD affinitymask, sysaffinitymask;

  if (!GetProcessAffinityMask(myproc, &affinitymask, &sysaffinitymask)) {
    /* count length of affinity list */
    int affinitycount=0;
    int i;
    for (i=0; i<31; i++) {
      affinitycount += (affinitymask >> i) & 0x1;
    }
  
    /* build affinity list */
    if (affinitycount > 0) {
      affinitylist = (int *) malloc(affinitycount * sizeof(int));
      if (affinitylist == NULL)
        return NULL;

      int curcount = 0;
      for (i=0; i<CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &affinitymask)) {
          affinitylist[curcount] = i;
          curcount++;
        }
      }
    }

    *cpuaffinitycount = affinitycount; /* return final affinity list */
  }
#endif

/* Linux process affinity mask query */
#if defined(__linux)

/* protect ourselves from some older Linux distros */
#if defined(CPU_SETSIZE)
  int i;
  cpu_set_t affinitymask;
  int affinitycount=0;

#if defined(VMDTHR_COMPAT_GLIBC232)
  typedef int(*pt2Func)(pid_t,cpu_set_t*);
  /* PID 0 refers to the current process */
  if (glib232compat_getaffinity(0, sizeof(affinitymask), &affinitymask,(pt2Func)&sched_getaffinity) < 0) {
#else
  /* PID 0 refers to the current process */
  if (sched_getaffinity(0, sizeof(affinitymask), &affinitymask) < 0) {
#endif
    perror("vmd_cpu_affinitylist: sched_getaffinity");
    return NULL;
  }

  /* count length of affinity list */
  for (i=0; i<CPU_SETSIZE; i++) {
    affinitycount += CPU_ISSET(i, &affinitymask);
  }

  /* build affinity list */
  if (affinitycount > 0) {
    affinitylist = (int *) malloc(affinitycount * sizeof(int));
    if (affinitylist == NULL)
      return NULL;

    int curcount = 0;
    for (i=0; i<CPU_SETSIZE; i++) {
      if (CPU_ISSET(i, &affinitymask)) {
        affinitylist[curcount] = i;
        curcount++;
      }
    }
  }

  *cpuaffinitycount = affinitycount; /* return final affinity list */
#endif
#endif

  /* MacOS X 10.5.x has a CPU affinity query/set capability finally      */
  /* http://developer.apple.com/releasenotes/Performance/RN-AffinityAPI/ */

  /* Solaris and HP-UX use pset_bind() and related functions, and they   */
  /* don't use the single-level mask-based scheduling mechanism that     */
  /* the others, use.  Instead, they use a hierarchical tree of          */
  /* processor sets and processes float within those, or are tied to one */
  /* processor that's a member of a particular set.                      */

  return affinitylist;
}


int vmd_thread_set_self_cpuaffinity(int cpu) {
  int status=-1; /* unsupported by default */

#ifdef VMDTHREADS

#if defined(__linux)
#if 0
  /* XXX this code is too new even for RHEL4, though it runs on Fedora 7 */
  /* and other newer revs.                                               */
  /* NPTL systems can assign per-thread affinities this way              */
  cpu_set_t affinitymask;
  CPU_ZERO(&affinitymask); 
  CPU_SET(cpu, &affinitymask);
  status = pthread_setaffinity_np(pthread_self(), sizeof(affinitymask), &affinitymask);
#else
  /* non-NPTL systems based on the clone() API must use this method      */
  cpu_set_t affinitymask;
  CPU_ZERO(&affinitymask); 
  CPU_SET(cpu, &affinitymask);

#if defined(VMDTHR_COMPAT_GLIBC232)
  typedef int(*pt2Func)(pid_t,const cpu_set_t*);
  /* PID 0 refers to the current process */
  if ((status=glib232compat_setaffinity(0,sizeof(affinitymask),&affinitymask,(pt2Func)&sched_setaffinity)) < 0) {
#else
  /* PID 0 refers to the current process */
  if ((status=sched_setaffinity(0, sizeof(affinitymask), &affinitymask)) < 0) {
#endif
    perror("vmd_thread_set_self_cpuaffinitylist: sched_setaffinity");
    return status;
  }
#endif

  /* call sched_yield() so new affinity mask takes effect immediately */
  sched_yield();
#endif /* linux */

  /* MacOS X 10.5.x has a CPU affinity query/set capability finally      */
  /* http://developer.apple.com/releasenotes/Performance/RN-AffinityAPI/ */

  /* Solaris and HP-UX use pset_bind() and related functions, and they   */
  /* don't use the single-level mask-based scheduling mechanism that     */
  /* the others, use.  Instead, they use a hierarchical tree of          */
  /* processor sets and processes float within those, or are tied to one */
  /* processor that's a member of a particular set.                      */
#endif

  return status;
}


int vmd_thread_setconcurrency(int nthr) {
  int status=0;

#ifdef VMDTHREADS
#if defined(__sun) 
#ifdef USEPOSIXTHREADS 
  status = pthread_setconcurrency(nthr);
#else
  status = thr_setconcurrency(nthr);
#endif
#endif /* SunOS */

#if defined(__irix) || defined(_AIX)
  status = pthread_setconcurrency(nthr);
#endif
#endif /* VMDTHREADS */

  return status;
}



/* Typedef to eliminate compiler warning caused by C/C++ linkage conflict. */
#ifdef __cplusplus
extern "C" {
#endif
  typedef void * (*VMDTHREAD_START_ROUTINE)(void *);
#ifdef __cplusplus
}
#endif

int vmd_thread_create(vmd_thread_t * thr, void * fctn(void *), void * arg) {
  int status=0;

#ifdef VMDTHREADS 
#ifdef _MSC_VER
  DWORD tid; /* thread id, msvc only */
  *thr = CreateThread(NULL, 8192, (LPTHREAD_START_ROUTINE) fctn, arg, 0, &tid);
  if (*thr == NULL) {
    status = -1;
  }
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS 
#if defined(_AIX)
  {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    status = pthread_create(thr, &attr, fctn, arg);
    pthread_attr_destroy(&attr);
  }
#else   
  status = pthread_create(thr, NULL, (VMDTHREAD_START_ROUTINE)fctn, arg);
#endif 
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */
 
  return status;
}


int vmd_thread_join(vmd_thread_t thr, void ** stat) {
  int status=0;  

#ifdef VMDTHREADS
#ifdef _MSC_VER
  DWORD wstatus = 0;
 
  wstatus = WAIT_TIMEOUT;
 
  while (wstatus != WAIT_OBJECT_0) {
    wstatus = WaitForSingleObject(thr, INFINITE);
  }
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_join(thr, stat);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}  


int vmd_mutex_init(vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS
#ifdef _MSC_VER
  InitializeCriticalSection(mp);
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_mutex_init(mp, 0);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}


int vmd_mutex_lock(vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS
#ifdef _MSC_VER
  EnterCriticalSection(mp);
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_mutex_lock(mp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}


int vmd_mutex_unlock(vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS  
#ifdef _MSC_VER
  LeaveCriticalSection(mp);
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_mutex_unlock(mp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}


int vmd_mutex_destroy(vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS
#ifdef _MSC_VER
  DeleteCriticalSection(mp);
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_mutex_destroy(mp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}



int vmd_cond_init(vmd_cond_t * cvp) {
  int status=0;

#ifdef VMDTHREADS
#ifdef _MSC_VER
#if defined(VMDUSEWIN2008CONDVARS)
  InitializeConditionVariable(cvp);
#else
  /* XXX not implemented */
  cvp->waiters = 0;

  /* Create an auto-reset event. */
  cvp->events[VMD_COND_SIGNAL] = CreateEvent(NULL,  /* no security */
                                             FALSE, /* auto-reset event */
                                             FALSE, /* non-signaled initially */
                                             NULL); /* unnamed */

  // Create a manual-reset event.
  cvp->events[VMD_COND_BROADCAST] = CreateEvent(NULL,  /* no security */
                                                TRUE,  /* manual-reset */
                                                FALSE, /* non-signaled initially*/
                                                NULL); /* unnamed */
#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_init(cvp, NULL);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}

int vmd_cond_destroy(vmd_cond_t * cvp) {
  int status=0;

#ifdef VMDTHREADS
#ifdef _MSC_VER
#if defined(VMDUSEWIN2008CONDVARS)
  /* XXX not implemented */
#else
  CloseHandle(cvp->events[VMD_COND_SIGNAL]);
  CloseHandle(cvp->events[VMD_COND_BROADCAST]);
#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_destroy(cvp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}

int vmd_cond_wait(vmd_cond_t * cvp, vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS
#ifdef _MSC_VER
#if defined(VMDUSEWIN2008CONDVARS)
  SleepConditionVariableCS(cvp, mp, INFINITE)
#else
#if !defined(VMDUSEINTERLOCKEDATOMICOPS)
  EnterCriticalSection(&cvp->waiters_lock);
  cvp->waiters++;
  LeaveCriticalSection(&cvp->waiters_lock);
#else
  InterlockedIncrement(&cvp->waiters);
#endif

  LeaveCriticalSection(mp); /* SetEvent() maintains state, avoids lost wakeup */

  /* Wait either a single or broadcast even to become signalled */
  int result = WaitForMultipleObjects(2, cvp->events, FALSE, INFINITE);

#if !defined(VMDUSEINTERLOCKEDATOMICOPS)
  EnterCriticalSection (&cvp->waiters_lock);
  cvp->waiters--;
  LONG last_waiter = 
    ((result == (WAIT_OBJECT_0 + VMD_COND_BROADCAST)) && cvp->waiters == 0);
  LeaveCriticalSection (&cvp->waiters_lock);
#else
  LONG my_waiter = InterlockedDecrement(&cvp->waiters);
  LONG last_waiter = 
    ((result == (WAIT_OBJECT_0 + VMD_COND_BROADCAST)) && my_waiter == 0);
#endif

  /* Some thread called cond_broadcast() */
  if (last_waiter)
    /* We're the last waiter to be notified or to stop waiting, so */
    /* reset the manual event.                                     */
    ResetEvent(cvp->events[VMD_COND_BROADCAST]); 

  EnterCriticalSection(mp);
#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_wait(cvp, mp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}

int vmd_cond_signal(vmd_cond_t * cvp) {
  int status=0;

#ifdef VMDTHREADS
#ifdef _MSC_VER
#if defined(VMDUSEWIN2008CONDVARS)
  WakeConditionVariable(cvp);
#else
#if !defined(VMDUSEINTERLOCKEDATOMICOPS)
  EnterCriticalSection(&cvp->waiters_lock);
  int have_waiters = (cvp->waiters > 0);
  LeaveCriticalSection(&cvp->waiters_lock);
  if (have_waiters)
    SetEvent (cvp->events[VMD_COND_SIGNAL]);
#else
  if (InterlockedExchangeAdd(&cvp->waiters, 0) > 0)
    SetEvent(cvp->events[VMD_COND_SIGNAL]);
#endif
#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_signal(cvp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}

int vmd_cond_broadcast(vmd_cond_t * cvp) {
  int status=0;

#ifdef VMDTHREADS
#ifdef _MSC_VER
#if defined(VMDUSEWIN2008CONDVARS)
  WakeAllConditionVariable(cvp);
#else
#if !defined(VMDUSEINTERLOCKEDATOMICOPS)
  EnterCriticalSection(&cvp->waiters_lock);
  int have_waiters = (cvp->waiters > 0);
  LeaveCriticalSection(&cvp->waiters_lock);
  if (have_waiters)
    SetEvent(cvp->events[VMD_COND_BROADCAST]);
#else
  if (InterlockedExchangeAdd(&cvp->waiters, 0) > 0)
    SetEvent(cvp->events[VMD_COND_BROADCAST]);
#endif

#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_broadcast(cvp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}
  


#if !defined(VMDTHREADS)

int vmd_thread_barrier_init_proc_shared(vmd_barrier_t *barrier, int n_clients) {
  return 0;
}

void vmd_thread_barrier_destroy(vmd_barrier_t *barrier) {
}

int vmd_thread_barrier(vmd_barrier_t *barrier, int increment) {
  return 0;
}

#else 

#ifdef USEPOSIXTHREADS

/* When rendering in the CAVE we use a special synchronization    */
/* mode so that shared memory mutexes and condition variables     */
/* will work correctly when accessed from multiple processes.     */
/* Inter-process synchronization involves the kernel to a greater */
/* degree, so these barriers are substantially more costly to use */
/* than the ones designed for use within a single-process.        */
int vmd_thread_barrier_init_proc_shared(vmd_barrier_t *barrier, int n_clients) {
  if (barrier != NULL) {
    barrier->n_clients = n_clients;
    barrier->n_waiting = 0;
    barrier->phase = 0;
    barrier->sum = 0;

    pthread_mutexattr_t mattr;
    pthread_condattr_t  cattr;

    printf("Setting barriers to have system scope...\n");

    pthread_mutexattr_init(&mattr);
    if (pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED) != 0) {
      printf("WARNING: could not set mutex to process shared scope\n");
    }

    pthread_condattr_init(&cattr);
    if (pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED) != 0) {
      printf("WARNING: could not set mutex to process shared scope\n");
    }

    pthread_mutex_init(&barrier->lock, &mattr);
    pthread_cond_init(&barrier->wait_cv, &cattr);

    pthread_condattr_destroy(&cattr);
    pthread_mutexattr_destroy(&mattr);
  }

  return 0;
}

void vmd_thread_barrier_destroy(vmd_barrier_t *barrier) {
  pthread_mutex_destroy(&barrier->lock);
  pthread_cond_destroy(&barrier->wait_cv);
}

int vmd_thread_barrier(vmd_barrier_t *barrier, int increment) {
  int my_phase;
  int my_result;

  pthread_mutex_lock(&barrier->lock);
  my_phase = barrier->phase;
  barrier->sum += increment;
  barrier->n_waiting++;

  if (barrier->n_waiting == barrier->n_clients) {
    barrier->result = barrier->sum;
    barrier->sum = 0;
    barrier->n_waiting = 0;
    barrier->phase = 1 - my_phase;
    pthread_cond_broadcast(&barrier->wait_cv);
  }

  while (barrier->phase == my_phase) {
    pthread_cond_wait(&barrier->wait_cv, &barrier->lock);
  }

  my_result = barrier->result;

  pthread_mutex_unlock(&barrier->lock);

  return my_result;
}

#endif

#endif /* VMDTHREADS */

#ifdef __cplusplus
}
#endif

#endif
