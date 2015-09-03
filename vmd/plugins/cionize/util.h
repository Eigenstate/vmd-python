/* 
 * util.h - This file contains defines for the timer functions...
 *
 *  $Id: util.h,v 1.1 2006/11/14 19:37:11 petefred Exp $
 */

#if !defined(RT_UTIL_H) 
#define RT_UTIL_H 1

typedef void * rt_timerhandle;          /* a timer handle */
rt_timerhandle rt_timer_create(void);   /* create a timer (clears timer)  */
void rt_timer_destroy(rt_timerhandle);  /* create a timer (clears timer)  */
void rt_timer_start(rt_timerhandle);    /* start a timer  (clears timer)  */
void rt_timer_stop(rt_timerhandle);     /* stop a timer                   */
float rt_timer_time(rt_timerhandle);    /* report elapsed time in seconds */
float rt_timer_timenow(rt_timerhandle); /* report elapsed time in seconds */

#define RT_RAND_MAX 4294967296.0       /* Maximum random value from rt_rand */
unsigned int rt_rand(unsigned int *);  /* thread-safe 32-bit random numbers */

#endif
