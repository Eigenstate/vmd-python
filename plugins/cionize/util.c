/*
 * util.c - Contains all of the timing functions for various platforms.
 *
 *  $Id: util.c,v 1.3 2008/10/16 21:02:44 johns Exp $ 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "util.h"

#if defined(__PARAGON__) || defined(__IPSC__)
#if defined(__IPSC__)
#include <cube.h>
#endif     /* iPSC/860 specific */

#if defined(__PARAGON__)
#include <nx.h>
#endif     /* Paragon XP/S specific */

#include <estat.h>
#endif /* iPSC/860 and Paragon specific items */

/* most platforms will use the regular time function gettimeofday() */
#if !defined(__IPSC__) && !defined(__PARAGON__) && !defined(NEXT)
#define STDTIME
#endif

#if defined(NEXT) 
#include <time.h>
#undef STDTIME
#define OLDUNIXTIME
#endif

#if defined(_MSC_VER) || defined(WIN32)
#include <windows.h>
#undef STDTIME
#define WIN32GETTICKCOUNT
#endif

#if defined(__linux) || defined(Bsd) || defined(AIX) || defined(SunOS) || defined(HPUX) || defined(_CRAYT3E) || defined(_CRAY) || defined(_CRAYC) || defined(__osf__) || defined(__BEOS__) || defined(__APPLE__) || defined(__irix)
#include <sys/time.h>
#endif

#if defined(MCOS) || defined(VXWORKS)
#define POSIXTIME
#endif


#if defined(WIN32GETTICKCOUNT)
typedef struct {
  DWORD starttime;
  DWORD endtime;
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  t->starttime = GetTickCount();
}

void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  t->endtime = GetTickCount();
}

float rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  flt ttime;

  ttime = (double) (t->endtime - t->starttime) / 1000.0;

  return (float) ttime;
}
#endif


#if defined(POSIXTIME)
#undef STDTIME
#include <time.h>

typedef struct {
  struct timespec starttime;
  struct timespec endtime;
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  clock_gettime(CLOCK_REALTIME, &t->starttime);
}

void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  clock_gettime(CLOCK_REALTIME, &t->endtime);
}

float rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  flt ttime, start, end;

  start = (t->starttime.tv_sec + 1.0 * t->starttime.tv_nsec / 1000000000.0);
    end = (t->endtime.tv_sec + 1.0 * t->endtime.tv_nsec / 1000000000.0);
  ttime = end - start;

  return (float) ttime;
}
#endif



/* if we're running on a Paragon or iPSC/860, use mclock() hi res timers */
#if defined(__IPSC__) || defined(__PARAGON__)

typedef struct {
  long starttime;
  long stoptime;
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  t->starttime=mclock(); 
}

void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  t->stoptime=mclock();
}

float rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  flt a;
  a = t->stoptime - t->starttime;
  a = ( a / 1000.0 );
  return (float) a;
}
#endif



/* if we're on a Unix with gettimeofday() we'll use newer timers */
#ifdef STDTIME 
typedef struct {
  struct timeval starttime, endtime;
#if !defined(VMS) && !defined(__STRICT_ANSI__)
  struct timezone tz;
#endif
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
#if defined(VMS) || defined(__STRICT_ANSI__)
  gettimeofday(&t->starttime, NULL);
#else
  gettimeofday(&t->starttime, &t->tz);
#endif
} 
  
void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
#if defined(VMS) || defined(__STRICT_ANSI__)
  gettimeofday(&t->endtime, NULL);
#else
  gettimeofday(&t->endtime, &t->tz);
#endif
} 
  
float rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  double ttime, start, end;

  start = (t->starttime.tv_sec + 1.0 * t->starttime.tv_usec / 1000000.0);
    end = (t->endtime.tv_sec + 1.0 * t->endtime.tv_usec / 1000000.0);
  ttime = end - start;

  return (float) ttime;
}  
#endif



/* use the old fashioned Unix time functions */
#ifdef OLDUNIXTIME
typedef struct {
  time_t starttime;
  time_t stoptime;
} rt_timer;

void rt_timer_start(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  time(&t->starttime);
}

void rt_timer_stop(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  time(&t->stoptime);
}

float rt_timer_time(rt_timerhandle v) {
  rt_timer * t = (rt_timer *) v;
  flt a;
  a = difftime(t->stoptime, t->starttime);
  return (float) a;
}
#endif


/* 
 * system independent routines to create and destroy timers 
 */
rt_timerhandle rt_timer_create(void) {
  rt_timer * t;  
  t = (rt_timer *) malloc(sizeof(rt_timer));
  memset(t, 0, sizeof(rt_timer));
  return t;
}

void rt_timer_destroy(rt_timerhandle v) {
  free(v);
}

float rt_timer_timenow(rt_timerhandle v) {
  rt_timer_stop(v);
  return rt_timer_time(v);
}



/*
 * Code for machines with deficient libc's etc.
 */

#if defined(__IPSC__) && !defined(__PARAGON__) 

/* the iPSC/860 libc is *missing* strstr(), so here it is.. */
char * strstr(const char *s, const char *find) {
  register char c, sc;
  register size_t len;

  if ((c = *find++) != 0) {
    len = strlen(find);
    do {
      do {
        if ((sc = *s++) == 0)
          return (NULL);
      } while (sc != c);
    } while (strncmp(s, find, len) != 0);
    s--;
  }
  return ((char *)s);
}
#endif

/* the Mercury libc is *missing* isascii(), so here it is.. */
#if defined(MCOS)
   int isascii(int c) {
     return (!((c) & ~0177));
   }
#endif

/*
 * Thread Safe Random Number Generators
 * (no internal static data storage)
 * 
 * Note: According to numerical recipes, one should not use
 *       random numbers in any way similar to rand() % number,
 *       as the greatest degree of randomness tends to be found
 *       in the upper bits of the random number, rather than the
 *       lower bits.  
 */

/*
 * Simple 32-bit random number generator suggested in
 * numerical recipes in C book.
 *
 * This random number generator has been fixed to work on 
 * machines that have "int" types which are larger than 32 bits.
 *
 * The rt_rand() API is similar to the reentrant "rand_r" version
 * found in some libc implementations.
 */
unsigned int rt_rand(unsigned int * idum) {
  *idum = ((1664525 * (*idum)) + 1013904223) & ((unsigned int) 0xffffffff); 
  return *idum;
}


