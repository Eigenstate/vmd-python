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
 *      $RCSfile: VMDThreads.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.54 $       $Date: 2010/12/16 04:08:47 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Remaining VMD-specific threading code used by CAVE/FreeVR builds.
 *   Derived from Tachyon threads code, but simplified for use in VMD.
 ***************************************************************************/

#ifndef VMD_THREADS_INC
#define VMD_THREADS_INC 1

/* POSIX Threads */
#if defined(__hpux) || defined(__irix) || defined(__linux) || defined(_CRAY) || defined(__osf__) || defined(_AIX) || defined(__APPLE__) || defined(__sun)
#if !defined(USEPOSIXTHREADS)
#define USEPOSIXTHREADS
#endif
#endif

#ifdef VMDTHREADS
#ifdef USEPOSIXTHREADS
#include <pthread.h>

typedef pthread_t        vmd_thread_t;
typedef pthread_mutex_t   vmd_mutex_t;
typedef pthread_cond_t     vmd_cond_t;

typedef struct vmd_barrier_struct {
  int padding1[8]; /* Padding bytes to avoid false sharing and cache aliasing */
  pthread_mutex_t lock;   /**< Mutex lock for the structure */
  int n_clients;          /**< Number of threads to wait for at barrier */
  int n_waiting;          /**< Number of currently waiting threads */
  int phase;              /**< Flag to separate waiters from fast workers */
  int sum;                /**< Sum of arguments passed to barrier wait */
  int result;             /**< Answer to be returned by barrier wait */
  pthread_cond_t wait_cv; /**< Clients wait on condition variable to proceed */
  int padding2[8]; /* Padding bytes to avoid false sharing and cache aliasing */
} vmd_barrier_t;

#endif



#ifdef _MSC_VER
#include <windows.h>
typedef HANDLE vmd_thread_t;
typedef CRITICAL_SECTION vmd_mutex_t;

#if 0 && (NTDDI_VERSION >= NTDDI_WS08 || _WIN32_WINNT > 0x0600) 
/* Use native condition variables only with Windows Server 2008 and newer... */
#define VMDUSEWIN2008CONDVARS 1
typedef  CONDITION_VARIABLE vmd_cond_t;
#else
/* Every version of Windows prior to Vista/WS2008 must emulate */
/* variables using manually resettable events or other schemes */ 

/* For higher performance, use interlocked memory operations   */
/* rather than locking/unlocking mutexes when manipulating     */
/* internal state.                                             */
#if 1
#define VMDUSEINTERLOCKEDATOMICOPS 1
#endif 
#define VMD_COND_SIGNAL    0
#define VMD_COND_BROADCAST 1
typedef struct {
  LONG waiters;     /**< XXX this _MUST_ be 32-bit aligned for correct */
                    /**< operation with the InterlockedXXX() APIs      */
  CRITICAL_SECTION waiters_lock;
  HANDLE events[2]; /**< Signal and broadcast event HANDLEs. */
} vmd_cond_t;
#endif

typedef HANDLE vmd_barrier_t; /**< Not implemented for Windows */
#endif
#endif


#ifndef VMDTHREADS
typedef int vmd_thread_t;
typedef int vmd_mutex_t;
typedef int vmd_cond_t;
typedef int vmd_barrier_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Mutexes
 */

/* initialize a mutex */
int vmd_mutex_init(vmd_mutex_t *);

/* lock a mutex */
int vmd_mutex_lock(vmd_mutex_t *);

/* unlock a mutex */
int vmd_mutex_unlock(vmd_mutex_t *);

/* destroy a mutex */
int vmd_mutex_destroy(vmd_mutex_t *);


/*
 * Condition variables
 */
/* initialize a condition variable */
int vmd_cond_init(vmd_cond_t *);

/* destroy a condition variable */
int vmd_cond_destroy(vmd_cond_t *);

/* wait on a condition variable */
int vmd_cond_wait(vmd_cond_t *, vmd_mutex_t *);

/* signal a condition variable, waking at least one thread */
int vmd_cond_signal(vmd_cond_t *);

/* signal a condition variable, waking all threads */
int vmd_cond_broadcast(vmd_cond_t *);

#endif

/* 
 * When rendering in the CAVE we use a special synchronization
 * mode so that shared memory mutexes and condition variables 
 * will work correctly when accessed from multiple processes.
 * Inter-process synchronization involves the kernel to a greater
 * degree, so these barriers are substantially more costly to use 
 * than the ones designed for use within a single-process.
 */
int vmd_thread_barrier_init_proc_shared(vmd_barrier_t *, int n_clients);

/* destroy a thread barrier */
void vmd_thread_barrier_destroy(vmd_barrier_t *barrier);

/* 
 * Synchronize on a thread barrier, returning the sum of all 
 * of the "increment" parameters from participating threads 
 */
int vmd_thread_barrier(vmd_barrier_t *barrier, int increment);

#ifdef __cplusplus
}
#endif
