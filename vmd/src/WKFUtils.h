/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2009 John E. Stone
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: WKFUtils.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.1 $       $Date: 2009/10/26 14:59:45 $
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

#ifndef WKF_UTILS_INC
#define WKF_UTILS_INC 1

#ifdef __cplusplus
extern "C" {
#endif

typedef void * wkf_timerhandle;            ///< a timer handle
wkf_timerhandle wkf_timer_create(void);    ///< create a timer (clears timer)
void wkf_timer_destroy(wkf_timerhandle);   ///< create a timer (clears timer)
void wkf_timer_start(wkf_timerhandle);     ///< start a timer  (clears timer)
void wkf_timer_stop(wkf_timerhandle);      ///< stop a timer
double wkf_timer_time(wkf_timerhandle);    ///< report elapsed time in seconds
double wkf_timer_timenow(wkf_timerhandle); ///< report elapsed time in seconds
double wkf_timer_start_time(wkf_timerhandle); ///< report wall starting time
double wkf_timer_stop_time(wkf_timerhandle); ///< report wall stopping time

typedef struct {
  wkf_timerhandle timer;
  double updatetime;
} wkfmsgtimer;

/// initialize periodic status message timer
extern wkfmsgtimer * wkf_msg_timer_create(double updatetime);

/// return true if it's time to print a status update message
extern int wkf_msg_timer_timeout(wkfmsgtimer *time);

/// destroy message timer
void wkf_msg_timer_destroy(wkfmsgtimer * mt);

#ifdef __cplusplus
}
#endif

#endif
