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
 *      $RCSfile: vmdconsole.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.13 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * vmd console redirector
 * (c) 2006-2009 Axel Kohlmeyer <akohlmey@cmm.chem.upenn.edu>
 * 
 ***************************************************************************/

#if defined(VMDTKCON)

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>

#include "vmdconsole.h"
#include "WKFThreads.h"

#ifdef __cplusplus
extern "C" {
#endif

/* structure for linked list of pending messages. */
struct vmdcon_msg 
{
    char *txt;
    int  lvl;
    struct vmdcon_msg *next;
};

static struct vmdcon_msg *vmdcon_pending=NULL;
static struct vmdcon_msg *vmdcon_lastmsg=NULL;

static struct vmdcon_msg *vmdcon_logmsgs=NULL;
static struct vmdcon_msg *vmdcon_lastlog=NULL;
static int vmdcon_max_loglen=2000;
static int vmdcon_loglen=0;
    
/* static buffer size for vmdcon_printf */
static const int vmdcon_bufsz=4092;

/* path to console text widget */
static char *vmdcon_wpath=NULL;

/* current insertion mark */
static char *vmdcon_mark=NULL;

/* opaque pointer to the current tcl interpreter */
static void *vmdcon_interp=NULL;

/* output destination status. */
static int vmdcon_status=VMDCON_UNDEF;

/* output loglevel. default to print all.*/
static int vmdcon_loglvl=VMDCON_ALL;

/* mutex to lock access to status variables */
static wkf_mutex_t vmdcon_status_lock;
static wkf_mutex_t vmdcon_output_lock;

/* initialize vmdcon */
void vmdcon_init(void) 
{
    wkf_mutex_init(&vmdcon_status_lock);
    wkf_mutex_init(&vmdcon_output_lock);
    vmdcon_set_status(VMDCON_NONE, NULL);
}

/* report current vmdcon status */
int vmdcon_get_status(void)
{
    return vmdcon_status;
}
 
/* set current vmdcon status */
void vmdcon_set_status(int status, void *interp)
{
    wkf_mutex_lock(&vmdcon_status_lock);
    if (interp != NULL) vmdcon_interp=interp;
    vmdcon_status=status;
    tcl_vmdcon_set_status_var(vmdcon_interp, status);
    wkf_mutex_unlock(&vmdcon_status_lock);
}

/* set current vmdcon log level */
void vmdcon_set_loglvl(int lvl)
{
    wkf_mutex_lock(&vmdcon_status_lock);
    vmdcon_loglvl=lvl;
    wkf_mutex_unlock(&vmdcon_status_lock);
}

/* set current vmdcon log level */
int vmdcon_get_loglvl(void)
{
    return vmdcon_loglvl;
}

/* turn on text mode processing */
void vmdcon_use_text(void *interp) 
{
    vmdcon_set_status(VMDCON_TEXT, interp);
}

/* turn on tk text widget mode processing */
void vmdcon_use_widget(void *interp) 
{
    vmdcon_set_status(VMDCON_WIDGET, interp);
}

/* register a tk/tcl widget to be the console window.
 * we get the widget path directly from tcl, 
 * so we have to create a copy (and free it later).
 * we also need a pointer to the tcl interpreter.
 */
int vmdcon_register(const char *w_path, const char *mark, void *interp)
{
    wkf_mutex_lock(&vmdcon_status_lock);
    vmdcon_interp=interp;

    /* unregister current console widget */
    if(w_path == NULL) {
        if (vmdcon_wpath != NULL) {
            free(vmdcon_wpath);
            free(vmdcon_mark);
        }
        vmdcon_wpath=NULL;
        vmdcon_mark=NULL;
        /* we have to indicate that no console is available */
        if (vmdcon_status == VMDCON_WIDGET) vmdcon_status=VMDCON_NONE;
    } else {
        int len;
        
        if (vmdcon_wpath != NULL) {
            free(vmdcon_wpath);
            free(vmdcon_mark);
        }
    
        len=strlen(w_path);
        vmdcon_wpath=(char*)malloc(len+1);
        strcpy(vmdcon_wpath, w_path);
        len=strlen(mark);
        vmdcon_mark=(char*)malloc(len+1);
        strcpy(vmdcon_mark, mark);
    }
    wkf_mutex_unlock(&vmdcon_status_lock);

    /* try to flush pending console log text. */
    return vmdcon_purge();
}

/* append text from to console log buffer to queue. */
int vmdcon_showlog(void)
{
    struct vmdcon_msg *log, *msg;

    wkf_mutex_lock(&vmdcon_output_lock);
    log=vmdcon_logmsgs;
    do {
        /* append to message queue. */
        msg=(struct vmdcon_msg *)malloc(sizeof(struct vmdcon_msg));
        msg->txt=(char *) malloc(strlen(log->txt)+1);
        msg->lvl=VMDCON_ALWAYS;
        strcpy(msg->txt,log->txt);
        msg->next=NULL;
    
        if (vmdcon_pending == NULL) {
            vmdcon_pending=msg;
            vmdcon_lastmsg=msg;
        } else {
            vmdcon_lastmsg->next=msg;
            vmdcon_lastmsg=msg;
        }
        log=log->next;
    } while (log->next != NULL);

    /* terminate the dmesg output with a newline */
    msg=(struct vmdcon_msg *)malloc(sizeof(struct vmdcon_msg));
    msg->txt=(char *) malloc(strlen("\n")+1);
    msg->lvl=VMDCON_ALWAYS;
    strcpy(msg->txt,"\n");
    msg->next=NULL;
    
    if (vmdcon_pending == NULL) {
        vmdcon_pending=msg;
        vmdcon_lastmsg=msg;
    } else {
        vmdcon_lastmsg->next=msg;
        vmdcon_lastmsg=msg;
    }
    log=log->next;

    wkf_mutex_unlock(&vmdcon_output_lock);
    return vmdcon_purge();
}

/* append text to console log queue.
 * we have to make copies as we might get handed 
 * a tcl object or a pointer to some larger buffer. */
int vmdcon_append(int level, const char *txt, int len) 
{
    struct vmdcon_msg *msg;
    char *buf;

    /* len=0: don't print. len=-1, autodetect. */
    if (len == 0 ) return 0;
    if (len < 0) {
        len=strlen(txt);
    }

    wkf_mutex_lock(&vmdcon_output_lock);
    
    /* append to message queue. */
    /* but don't print stuff below the current loglevel */
    if (level >= vmdcon_loglvl) {
      /* create copy of text. gets free'd after it has been 'printed'. */
      buf=(char *)calloc(len+1,1);
      strncpy(buf,txt,len);
    
      msg=(struct vmdcon_msg *)malloc(sizeof(struct vmdcon_msg));
      msg->txt=buf;
      msg->lvl=level;
      msg->next=NULL;
      
      if (vmdcon_pending == NULL) {
        vmdcon_pending=msg;
        vmdcon_lastmsg=msg;
      } else {
        vmdcon_lastmsg->next=msg;
        vmdcon_lastmsg=msg;
      }
    }
    
    /* messages are added to the log regardless of loglevel.
     * this way we can silence the log window and still retrieve
     * useful information with 'vmdcon -dmesg'. */
    buf=(char *)calloc(len+1,1);
    strncpy(buf,txt,len);

    /* append to log message list. */
    msg=(struct vmdcon_msg *)malloc(sizeof(struct vmdcon_msg));
    msg->txt=buf;
    msg->lvl=level;
    msg->next=NULL;
        
    if (vmdcon_logmsgs == NULL) {
        vmdcon_logmsgs=msg;
        vmdcon_lastlog=msg;
        ++vmdcon_loglen;
    } else {
        vmdcon_lastlog->next=msg;
        vmdcon_lastlog=msg;
        ++vmdcon_loglen;
    }
    
    /* remove message from the front of the queue
     * in case we have too long a list */
    while (vmdcon_loglen > vmdcon_max_loglen) {
        msg=vmdcon_logmsgs;
        vmdcon_logmsgs=msg->next;
        free(msg->txt);
        free(msg);
        --vmdcon_loglen;
    }
    
    wkf_mutex_unlock(&vmdcon_output_lock);

    return 0;
}

/* flush current message queue to a registered 
 * console widget, if such a thing exists.
 * since vmdcon_append() allocates the storage,
 * for everything, we have to free the msg structs
 * and the strings. */
int vmdcon_purge(void) 
{
    struct vmdcon_msg *msg;
    const char *res;

    wkf_mutex_lock(&vmdcon_status_lock);
    /* purge message queue only if we have a working console window */
    if ( ! ((vmdcon_status == VMDCON_UNDEF) || (vmdcon_status == VMDCON_NONE)
        || ((vmdcon_status == VMDCON_WIDGET) &&
            ((vmdcon_interp == NULL) || (vmdcon_wpath == NULL))) ) ) {

        wkf_mutex_lock(&vmdcon_output_lock);
        while (vmdcon_pending != NULL) {
            msg=vmdcon_pending;

            switch (vmdcon_status) {
              case VMDCON_TEXT:
                  fputs(msg->txt,stdout);
                  break;
                  
              case VMDCON_WIDGET: 
                  res = tcl_vmdcon_insert(vmdcon_interp, vmdcon_wpath, 
                                          vmdcon_mark, msg->txt);
                  /* handle errors writing to a tcl console window.
                   * unregister widget, don't free current message 
                   * and append error message into holding buffer. */
                  if (res) {
                      wkf_mutex_unlock(&vmdcon_status_lock);
                      vmdcon_register(NULL, NULL, vmdcon_interp);
                      wkf_mutex_unlock(&vmdcon_output_lock);
                      vmdcon_printf(VMDCON_ERROR,
                                    "Problem writing to text widget: %s\n", res);
                      return 1;
                  }
                  break;

              default:
                  /* unknown console type */
                  return 1;
            }
            free(msg->txt);
            vmdcon_pending=msg->next;
            free(msg);

        }
        if (vmdcon_status == VMDCON_TEXT) 
            fflush(stdout);

        wkf_mutex_unlock(&vmdcon_output_lock);
    }
    wkf_mutex_unlock(&vmdcon_status_lock);
    return 0;
}

/* emulate printf. unfortunately, we cannot rely on 
 * snprintf being available, so we have to write to
 * a very large buffer and then free it. :-( */
int vmdcon_printf(const int lvl, const char *fmt, ...) 
{
    va_list ap;
    char *buf;
    int len;

    /* expand formated output into a single string */
    buf = (char *)malloc(vmdcon_bufsz);
    va_start(ap, fmt);
    len = vsprintf(buf, fmt, ap);

    /* check result. we may get a segfault, but if not
     * let the user know that he/she is in trouble. */
    if (len >= vmdcon_bufsz) {
        fprintf(stderr,"WARNING! buffer overflow in vmdcon_printf. %d vs %d.\n",
                len, vmdcon_bufsz);
        free(buf);
        errno=ERANGE;
        return -1;
    }

    /* prefix message with info level... or not. */
    switch (lvl) {
      case VMDCON_INFO:
        vmdcon_append(lvl, "Info) ", 6);
        break;

      case VMDCON_WARN:
        vmdcon_append(lvl, "Warning) ", 9);
        break;

      case VMDCON_ERROR:
        vmdcon_append(lvl, "ERROR) ", 7);
        break;

      default:  
        break;
    }

    vmdcon_append(lvl, buf, len);
    vmdcon_purge();

    free(buf);
    return 0;    
}

/* emulate fputs for console. */
int vmdcon_fputs(const int lvl, const char *string) 
{
    /* prefix message with info level... or not. */
    switch (lvl) {
      case VMDCON_INFO:
        vmdcon_append(lvl, "Info) ", 6);
        break;

      case VMDCON_WARN:
        vmdcon_append(lvl, "Warning) ", 9);
        break;

      case VMDCON_ERROR:
        vmdcon_append(lvl, "ERROR) ", 7);
        break;

      default:  
        break;
    }

    vmdcon_append(lvl, string, -1);
    vmdcon_purge();

    return 0;    
}

#ifdef __cplusplus
}
#endif

#endif
