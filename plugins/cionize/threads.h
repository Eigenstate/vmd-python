/*
 * threads.h - code for spawning threads on various platforms.
 *
 *  $Id: threads.h,v 1.1 2006/11/14 19:37:10 petefred Exp $
 */ 

#ifndef RT_THREADS_INC
#define RT_THREADS_INC 1

/* define which thread calls to use */
#if defined(USEPOSIXTHREADS) && defined(USEUITHREADS)
#error You may only define USEPOSIXTHREADS or USEUITHREADS, but not both
#endif

/* POSIX Threads */
#if defined(HPUX) || defined(__PARAGON__) || defined(Irix) || defined(__linux) ||     defined(_CRAY) || defined(__osf__) || defined(AIX) || defined(__APPLE__)
#if !defined(USEUITHREADS) && !defined(USEPOSIXTHREADS)
#define USEPOSIXTHREADS
#endif
#endif

/* Unix International Threads */
#if defined(SunOS)
#if !defined(USEPOSIXTHREADS) && !defined(USEUITHREADS)
#define USEUITHREADS
#endif
#endif


#ifdef THR
#ifdef USEPOSIXTHREADS
#include <pthread.h>

typedef pthread_t        rt_thread_t;
typedef pthread_mutex_t   rt_mutex_t;

typedef struct barrier_struct {
  int padding1[8]; /* Padding bytes to avoid false sharing and cache aliasing */
  pthread_mutex_t lock;   /* Mutex lock for the structure */
  int n_clients;          /* Number of threads to wait for at barrier */
  int n_waiting;          /* Number of currently waiting threads */
  int phase;              /* Flag to separate waiters from fast workers */
  int sum;                /* Sum of arguments passed to barrier_wait */
  int result;             /* Answer to be returned by barrier_wait */
  pthread_cond_t wait_cv; /* Clients wait on condition variable to proceed */
  int padding2[8]; /* Padding bytes to avoid false sharing and cache aliasing */
} rt_barrier_t;

typedef struct rwlock_struct {
  pthread_mutex_t lock;         /* read/write monitor lock */
  int rwlock;                   /* >0 = #rdrs, <0 = wrtr, 0=none */
  pthread_cond_t  rdrs_ok;      /* start waiting readers */
  unsigned int waiting_writers; /* # of waiting writers  */
  pthread_cond_t  wrtr_ok;      /* start waiting writers */ 
} rt_rwlock_t;

#endif

#ifdef USEUITHREADS
#include <thread.h>

typedef thread_t  rt_thread_t;
typedef mutex_t   rt_mutex_t;
typedef rwlock_t  rt_rwlock_t;

typedef struct barrier_struct {
  int padding1[8]; /* Padding bytes to avoid false sharing and cache aliasing */
  mutex_t lock;           /* Mutex lock for the structure */
  int n_clients;          /* Number of threads to wait for at barrier */
  int n_waiting;          /* Number of currently waiting threads */
  int phase;              /* Flag to separate waiters from fast workers */
  int sum;                /* Sum of arguments passed to barrier_wait */
  int result;             /* Answer to be returned by barrier_wait */
  cond_t wait_cv;         /* Clients wait on condition variable to proceed */
  int padding2[8]; /* Padding bytes to avoid false sharing and cache aliasing */
} rt_barrier_t;

#endif



#ifdef _MSC_VER
#include <windows.h>
typedef HANDLE rt_thread_t;
typedef HANDLE rt_mutex_t;
typedef HANDLE cond;
#endif
#endif



#ifndef THR
typedef int rt_thread_t;
typedef int rt_mutex_t;
typedef int rt_barrier_t;
typedef int rt_rwlock_t;
#endif



int rt_thread_numprocessors(void);
int rt_thread_setconcurrency(int);
int rt_thread_create(rt_thread_t *, void * routine(void *), void *);
int rt_thread_join(rt_thread_t, void **);

int rt_mutex_init(rt_mutex_t *);
int rt_mutex_lock(rt_mutex_t *);
int rt_mutex_unlock(rt_mutex_t *);

int rt_rwlock_init(rt_rwlock_t *);
int rt_rwlock_readlock(rt_rwlock_t *);
int rt_rwlock_writelock(rt_rwlock_t *);
int rt_rwlock_unlock(rt_rwlock_t *);

rt_barrier_t * rt_thread_barrier_init(int n_clients);
void rt_thread_barrier_destroy(rt_barrier_t *barrier);
int rt_thread_barrier(rt_barrier_t *barrier, int increment);

#endif
