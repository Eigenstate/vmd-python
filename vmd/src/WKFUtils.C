/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2009 John E. Stone
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: WKFUtils.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.1 $       $Date: 2009/10/26 14:59:44 $
 *
 ***************************************************************************/
/*
 * Copyright (c) 1994-2009 John E. Stone
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "WKFUtils.h"

#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#include <sys/time.h>
#include <errno.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#if defined(__irix)
#include <bstring.h>
#endif

#if defined(__hpux)
#include <time.h>
#endif // HPUX
#endif // _MSC_VER


#ifdef __cplusplus
extern "C" {
#endif


#if defined(_MSC_VER)
typedef struct {
  DWORD starttime;
  DWORD endtime;
} wkf_timer;

void wkf_timer_start(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  t->starttime = GetTickCount();
}

void wkf_timer_stop(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  t->endtime = GetTickCount();
}

double wkf_timer_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;

  ttime = ((double) (t->endtime - t->starttime)) / 1000.0;

  return ttime;
}

double wkf_timer_start_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) (t->starttime)) / 1000.0;
  return ttime;
}

double wkf_timer_stop_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) (t->endtime)) / 1000.0;
  return ttime;
}

#else

// Unix with gettimeofday()
typedef struct {
  struct timeval starttime, endtime;
  struct timezone tz;
} wkf_timer;

void wkf_timer_start(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  gettimeofday(&t->starttime, &t->tz);
}

void wkf_timer_stop(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  gettimeofday(&t->endtime, &t->tz);
}

double wkf_timer_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) (t->endtime.tv_sec - t->starttime.tv_sec)) +
          ((double) (t->endtime.tv_usec - t->starttime.tv_usec)) / 1000000.0;
  return ttime;
}

double wkf_timer_start_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) t->starttime.tv_sec) +
          ((double) t->starttime.tv_usec) / 1000000.0;
  return ttime;
}

double wkf_timer_stop_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) t->endtime.tv_sec) +
          ((double) t->endtime.tv_usec) / 1000000.0;
  return ttime;
}

#endif

// system independent routines to create and destroy timers
wkf_timerhandle wkf_timer_create(void) {
  wkf_timer * t;
  t = (wkf_timer *) malloc(sizeof(wkf_timer));
  memset(t, 0, sizeof(wkf_timer));
  return t;
}

void wkf_timer_destroy(wkf_timerhandle v) {
  free(v);
}

double wkf_timer_timenow(wkf_timerhandle v) {
  wkf_timer_stop(v);
  return wkf_timer_time(v);
}

/// initialize status message timer
wkfmsgtimer * wkf_msg_timer_create(double updatetime) {
  wkfmsgtimer *mt;
  mt = (wkfmsgtimer *) malloc(sizeof(wkfmsgtimer));
  if (mt != NULL) {
    mt->timer = wkf_timer_create();
    mt->updatetime = updatetime;
    wkf_timer_start(mt->timer);
  }
  return mt;
}

/// return true if it's time to print a status update message
int wkf_msg_timer_timeout(wkfmsgtimer *mt) {
  double elapsed = wkf_timer_timenow(mt->timer);
  if (elapsed > mt->updatetime) {
    // reset the clock and return true that our timer expired
    wkf_timer_start(mt->timer);
    return 1;
  } else if (elapsed < 0) {
    // time went backwards, best reset our clock!
    wkf_timer_start(mt->timer);
  }
  return 0;
}

/// destroy message timer
void wkf_msg_timer_destroy(wkfmsgtimer * mt) {
  wkf_timer_destroy(mt->timer);
  free(mt);
}

#ifdef __cplusplus
}
#endif

