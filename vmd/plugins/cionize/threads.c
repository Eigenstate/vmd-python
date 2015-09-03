/*
 * threads.c - code for spawning threads on various platforms.
 *
 *  $Id: threads.c,v 1.1 2006/11/14 19:35:14 petefred Exp $
 */ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "threads.h"

#ifdef _MSC_VER
#include <windows.h> /* main Win32 APIs and types */
#include <winbase.h> /* system services headers */
#endif

#if defined(SunOS) || defined(Irix) || defined(__linux) || defined(_CRAY) || defined(__osf__) || defined(AIX)
#include<unistd.h>  /* sysconf() headers, used by most systems */
#endif

#if defined(__APPLE__) && defined(THR)
#include <Carbon/Carbon.h> /* Carbon APIs for Multiprocessing */
#endif

#if defined(HPUX)
#include <sys/mpctl.h> /* HP-UX Multiprocessing headers */
#endif


int rt_thread_numprocessors(void) {
  int a=1;

#ifdef THR
#if defined(__APPLE__)
  a = MPProcessorsScheduled(); /* Number of active/running CPUs */
#endif

#ifdef _MSC_VER
  struct _SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  a = sysinfo.dwNumberOfProcessors; /* total number of CPUs */
#endif /* _MSC_VER */

#if defined(__PARAGON__) 
  a=2; /* Threads-capable Paragons have 2 CPUs for computation */
#endif /* __PARAGON__ */ 

#if defined(_CRAY)
  a = sysconf(_SC_CRAY_NCPU); /* total number of CPUs */
#endif

#if defined(SunOS) || defined(__linux) || defined(__osf__) || defined(AIX)
  a = sysconf(_SC_NPROCESSORS_ONLN); /* Number of active/running CPUs */
#endif /* SunOS */

#ifdef Irix
  a = sysconf(_SC_NPROC_ONLN); /* Number of active/running CPUs */
#endif /* IRIX */

#ifdef HPUX
  a = mpctl(MPC_GETNUMSPUS, 0, 0); /* total number of CPUs */
#endif /* HPUX */
#endif /* THR */

  return a;
}


int rt_thread_setconcurrency(int nthr) {
  int status=0;

#ifdef THR
#ifdef SunOS
  status = thr_setconcurrency(nthr);
#endif /* SunOS */

#if defined(Irix) || defined(AIX)
  status = pthread_setconcurrency(nthr);
#endif
#endif /* THR */

  return status;
}

int rt_thread_create(rt_thread_t * thr, void * routine(void *), void * arg) {
  int status=0;

#ifdef THR
#ifdef _MSC_VER
  int tid; /* thread id, msvc only */
 
  *thr = CreateThread(NULL, 8192,
                    (LPTHREAD_START_ROUTINE) routine, arg, 0, &tid);
 
  if (*thr == NULL) {
    status = -1;
  }
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS 
#if defined(AIX)
  /* AIX schedule threads in system scope by default, have to ask explicitly */
  {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM); 
    status = pthread_create(thr, &attr, routine, arg);
    pthread_attr_destroy(&attr);
  } 
#elif defined(__PARAGON__)
  status = pthread_create(thr, pthread_attr_default, routine, arg);
#else   
  status = pthread_create(thr, NULL, routine, arg);
#endif 
#endif /* USEPOSIXTHREADS */

#ifdef USEUITHREADS 
  status = thr_create(NULL, 0, routine, arg, 0, thr); 
#endif /* USEUITHREADS */
#endif /* THR */
 
  return status;
}


int rt_thread_join(rt_thread_t thr, void ** stat) {
  int status=0;  

#ifdef THR
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

#ifdef USEUITHREADS
  status = thr_join(thr, NULL, stat);
#endif /* USEPOSIXTHREADS */
#endif /* THR */

  return status;
}  


int rt_mutex_init(rt_mutex_t * mp) {
  int status=0;

#ifdef THR
#ifdef USEPOSIXTHREADS
  status = pthread_mutex_init(mp, 0);
#endif /* USEPOSIXTHREADS */

#ifdef USEUITHREADS 
  status = mutex_init(mp, USYNC_THREAD, NULL);
#endif /* USEUITHREADS */
#endif /* THR */

  return status;
}


int rt_mutex_lock(rt_mutex_t * mp) {
  int status=0;

#ifdef THR
#ifdef USEPOSIXTHREADS
  status = pthread_mutex_lock(mp);
#endif /* USEPOSIXTHREADS */

#ifdef USEUITHREADS
  status = mutex_lock(mp);
#endif /* USEUITHREADS */
#endif /* THR */

  return status;
}


int rt_mutex_unlock(rt_mutex_t * mp) {
  int status=0;

#ifdef THR  
#ifdef USEPOSIXTHREADS
  status = pthread_mutex_unlock(mp);
#endif /* USEPOSIXTHREADS */

#ifdef USEUITHREADS
  status = mutex_unlock(mp);
#endif /* USEUITHREADS */
#endif /* THR */

  return status;
}

/*
 * Reader/Writer locks -- slower than mutexes but good for some purposes
 */
int rt_rwlock_init(rt_rwlock_t * rwp) {
  int status=0;

#ifdef THR  
#ifdef USEPOSIXTHREADS
  pthread_mutex_init(&rwp->lock, NULL);
  pthread_cond_init(&rwp->rdrs_ok, NULL);
  pthread_cond_init(&rwp->wrtr_ok, NULL);
  rwp->rwlock = 0;
  rwp->waiting_writers = 0;
#endif /* USEPOSIXTHREADS */

#ifdef USEUITHREADS
  status = rwlock_init(rwp, USYNC_THREAD, NULL);
#endif /* USEUITHREADS */
#endif /* THR */

  return status;
}

int rt_rwlock_readlock(rt_rwlock_t * rwp) {
  int status=0;

#ifdef THR  
#ifdef USEPOSIXTHREADS
  pthread_mutex_lock(&rwp->lock);
  while (rwp->rwlock < 0 || rwp->waiting_writers) 
    pthread_cond_wait(&rwp->rdrs_ok, &rwp->lock);   
  rwp->rwlock++;   /* increment number of readers holding the lock */
  pthread_mutex_unlock(&rwp->lock);
#endif /* USEPOSIXTHREADS */

#ifdef USEUITHREADS
  status = rw_rdlock(rwp);
#endif /* USEUITHREADS */
#endif /* THR */

  return status;
}

int rt_rwlock_writelock(rt_rwlock_t * rwp) {
  int status=0;

#ifdef THR  
#ifdef USEPOSIXTHREADS
  pthread_mutex_lock(&rwp->lock);
  while (rwp->rwlock != 0) {
    rwp->waiting_writers++;
    pthread_cond_wait(&rwp->wrtr_ok, &rwp->lock);
    rwp->waiting_writers--;
  }
  rwp->rwlock=-1;
  pthread_mutex_unlock(&rwp->lock);
#endif /* USEPOSIXTHREADS */

#ifdef USEUITHREADS
  status = rw_wrlock(rwp);
#endif /* USEUITHREADS */
#endif /* THR */

  return status;
}

int rt_rwlock_unlock(rt_rwlock_t * rwp) {
  int status=0;

#ifdef THR  
#ifdef USEPOSIXTHREADS
  int ww, wr;
  pthread_mutex_lock(&rwp->lock);
  if (rwp->rwlock > 0) {
    rwp->rwlock--;
  } else {
    rwp->rwlock = 0;
  } 
  ww = (rwp->waiting_writers && rwp->rwlock == 0);
  wr = (rwp->waiting_writers == 0);
  pthread_mutex_unlock(&rwp->lock);
  if (ww) 
    pthread_cond_signal(&rwp->wrtr_ok);
  else if (wr)
    pthread_cond_signal(&rwp->rdrs_ok);
#endif /* USEPOSIXTHREADS */

#ifdef USEUITHREADS
  status = rw_unlock(rwp);
#endif /* USEUITHREADS */
#endif /* THR */

  return status;
}


#if !defined(THR)

rt_barrier_t * rt_thread_barrier_init(int n_clients) {
  return NULL;
}

void rt_thread_barrier_destroy(rt_barrier_t *barrier) {
}

int rt_thread_barrier(rt_barrier_t *barrier, int increment) {
  return 0;
}

#else 

#ifdef USEPOSIXTHREADS
rt_barrier_t * rt_thread_barrier_init(int n_clients) {
  rt_barrier_t *barrier = (rt_barrier_t *) malloc(sizeof(rt_barrier_t));

  if (barrier != NULL) {
    barrier->n_clients = n_clients;
    barrier->n_waiting = 0;
    barrier->phase = 0;
    barrier->sum = 0;
    pthread_mutex_init(&barrier->lock, NULL);
    pthread_cond_init(&barrier->wait_cv, NULL);
  }

  return barrier;
}

void rt_thread_barrier_destroy(rt_barrier_t *barrier) {
  pthread_mutex_destroy(&barrier->lock);
  pthread_cond_destroy(&barrier->wait_cv);
  free(barrier);
}

int rt_thread_barrier(rt_barrier_t *barrier, int increment) {
  int my_phase;

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

  pthread_mutex_unlock(&barrier->lock);

  return (barrier->result);
}
#endif


#ifdef USEUITHREADS

rt_barrier_t * rt_thread_barrier_init(int n_clients) {
  rt_barrier_t *barrier = (rt_barrier_t *) malloc(sizeof(rt_barrier_t));

  if (barrier != NULL) {
    barrier->n_clients = n_clients;
    barrier->n_waiting = 0;
    barrier->phase = 0;
    barrier->sum = 0;
    mutex_init(&barrier->lock, USYNC_THREAD, NULL);
    cond_init(&barrier->wait_cv, USYNC_THREAD, NULL);
  }

  return barrier;
}

void rt_thread_barrier_destroy(rt_barrier_t *barrier) {
  mutex_destroy(&barrier->lock);
  cond_destroy(&barrier->wait_cv);
  free(barrier);
}

int rt_thread_barrier(rt_barrier_t *barrier, int increment) {
  int my_phase;

  mutex_lock(&barrier->lock);
  my_phase = barrier->phase;
  barrier->sum += increment;
  barrier->n_waiting++;

  if (barrier->n_waiting == barrier->n_clients) {
    barrier->result = barrier->sum;
    barrier->sum = 0;
    barrier->n_waiting = 0;
    barrier->phase = 1 - my_phase;
    cond_broadcast(&barrier->wait_cv);
  }

  while (barrier->phase == my_phase) {
    cond_wait(&barrier->wait_cv, &barrier->lock);
  }

  mutex_unlock(&barrier->lock);

  return (barrier->result);
}

#endif /* USEUITHREADS */


#endif /* THR */



