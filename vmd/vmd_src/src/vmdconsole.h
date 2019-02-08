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
 *      $RCSfile: vmdconsole.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.10 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * vmd console redirector 
 * (c) 2006-2009 Axel Kohlmeyer <akohlmey@cmm.chem.upenn.edu>
 * 
 ***************************************************************************/

/* common definitions for the vmdconsole plugin */
#ifndef _VMDCONSOLE_H
#define _VMDCONSOLE_H

#ifdef __cplusplus
extern "C" {
#endif

/* list of vmd status types */
#define VMDCON_UNDEF    -1
#define VMDCON_NONE      0
#define VMDCON_WIDGET    1
#define VMDCON_TEXT      2

/* list of vmd console 'urgencies' */
#define VMDCON_ALL       0      /**< "print all messages" log level   */
#define VMDCON_INFO      1      /**< informational messages log level */
#define VMDCON_WARN      2      /**< warning messages" log level      */
#define VMDCON_ERROR     3      /**< error messages log level         */
#define VMDCON_ALWAYS    4      /**< print always log level           */
#define VMDCON_LOG       5      /**< store only in syslog log level   */

/* initialize vmd console */
extern void vmdcon_init(void);

/* report current vmdcon status */
extern int vmdcon_get_status(void);

/* set current vmdcon status */
extern void vmdcon_set_status(int, void *interp);

/* set current vmdcon log level */
void vmdcon_set_loglvl(int lvl);

/* set current vmdcon log level */
int vmdcon_get_loglvl(void);

/* turn on text mode processing */
extern void vmdcon_use_text(void *interp);

/* turn on tk text widget mode processing */
extern void vmdcon_use_widget(void *interp);

/* (de-)register a widget to be the console window */
extern int vmdcon_register(const char *w_path, const char *mark, void *interp);

/* append a string of up to 'length' characters to console message queue
 * at log level 'level'.
 * 'length' can be -1 to autodetect length with \0 terminated strings.
 * a 'length' of 0 means, do not add to message queue.  */
extern int vmdcon_append(int level, const char *text, int length);

/* purge message queue into registered text widget. */
extern int vmdcon_purge(void);

/* insert log message buffer into console destination. */
extern int vmdcon_showlog(void);

/* print to the current vmd console, printf style.*/
extern int vmdcon_printf(const int lvl, const char *format, ...);

/* print to the current vmd console, fputs style.*/
extern int vmdcon_fputs(const int lvl, const char *text);

/* insert text into an existing text widget.
 * returns NULL on success, or an error message on failure. */
extern const char *tcl_vmdcon_insert(void *interp, const char *w_path, 
                                     const char *mark, const char *text);

/* synchronize tcl variable for vmdcon status */
extern void tcl_vmdcon_set_status_var(void *interp, int status);

#ifdef __cplusplus
}
#endif

#endif
