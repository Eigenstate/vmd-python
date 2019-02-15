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
 *      $RCSfile: VMDCollab.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.11 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************/

#include "VMDCollab.h"
#include "WKFThreads.h"
#include "vmdsock.h"
#include "Inform.h"
#include "utilities.h"
#include "TextEvent.h"

#if defined(VMDTKCON)
#include "vmdconsole.h"
#endif

#include <limits.h>
#include <errno.h>

// collab messages will be the following number of bytes
static const int VMDCOLLAB_MSGSIZE = 256;


#if ( INT_MAX == 2147483647 )
typedef int     int32;
#else
typedef short   int32;
#endif

static int32 imd_readn(void *s, char *ptr, int32 n) {
  int32 nleft;
  int32 nread;
 
  nleft = n;
  while (nleft > 0) {
    if ((nread = vmdsock_read(s, ptr, nleft)) < 0) {
      if (errno == EINTR)
        nread = 0;         /* and call read() again */
      else
        return -1;
    } else if (nread == 0)
      break;               /* EOF */
    nleft -= nread;
    ptr += nread;
  }
  return n-nleft;
}


static int32 imd_writen(void *s, const char *ptr, int32 n) {
  int32 nleft;
  int32 nwritten;

  nleft = n;
  while (nleft > 0) {
    if ((nwritten = vmdsock_write(s, ptr, nleft)) <= 0) {
      if (errno == EINTR)
        nwritten = 0;
      else
        return -1;
    }
    nleft -= nwritten;
    ptr += nwritten; }
  return n;
}

VMDCollab::VMDCollab(VMDApp *app) : UIObject(app) {
  clientsock = NULL;
  serversock = NULL;
  eval_in_progress = FALSE;
#if defined(VMDTKCON)
  cmdbufstr = new Inform("",VMDCON_ALWAYS);
#else
  cmdbufstr = new Inform("");
#endif
  for (int i=0; i<Command::TOTAL; i++) command_wanted(i);
}


VMDCollab::~VMDCollab() {
  stopserver();
  delete cmdbufstr;
}


// this is a static method that will be created in a new child thread
void *VMDCollab::serverproc(void *serversock) {
  ResizeArray<void *>clients;
  char buf[VMDCOLLAB_MSGSIZE];
  int i, j;
  
  while (1) {
    // if we have no clients, hang until someone connects
    // otherwise, just check for pending connections
    if (vmdsock_selread(serversock, 0) > 0) {
      msgInfo << "serversock became readable" << sendmsg;
      void *clientsock = vmdsock_accept(serversock);
      if (clientsock) {
        msgInfo << "VMDCollab accepting connection" << sendmsg;
        clients.append(clientsock);
      }
    } else if (vmdsock_selwrite(serversock, 0)) {
      msgInfo << "serversock became writable; exiting..." << sendmsg;
      break;
    }

    // Loop through one socket at a time.  If incoming data is found,
    // drain it before moving on, on the assumption that we only want
    // commands from one VMD at a time to be propagated to the other
    // clients.
    for (i=0; i<clients.num(); i++) {
      void *client = clients[i];
      while (vmdsock_selread(client, 0) > 0) {
        memset(buf, 0, VMDCOLLAB_MSGSIZE);
        if (imd_readn(client, buf, VMDCOLLAB_MSGSIZE) != VMDCOLLAB_MSGSIZE) {
          msgInfo << "client sent incomplete message, shutting it down"
                  << sendmsg;
          vmdsock_shutdown(client);
          vmdsock_destroy(client);
          clients.remove(clients.find(client));
          break;
        }
        // send to all other clients
        for (j=0; j<clients.num(); j++) {
          void *dest = clients[j];
          if (dest != client) {
            imd_writen(clients[j], buf, VMDCOLLAB_MSGSIZE);
          }
        } // loop over clients other than sender
      } // while client is readable
    } // loop over clients
    vmd_msleep(10);
  }

  // if here, then the serversock got shut down, indicating that it's
  // time to die.
  msgInfo << "VMDCollab shutting down server" << sendmsg;
  for (i=0; i<clients.num(); i++) {
    void *client = clients[i];
    strcpy(buf, "exit");
    imd_writen(client, buf, VMDCOLLAB_MSGSIZE);
    vmdsock_shutdown(client);
    vmdsock_destroy(client);
  }
  vmdsock_destroy(serversock);
  return NULL;
}


int VMDCollab::startserver(int port) {
  if (serversock) {
    msgErr << "Already running a server on port " <<  port << sendmsg;
    return FALSE;
  }
  serversock = vmdsock_create();
  if (!serversock) {
    msgErr << "Could not create socket." << sendmsg;
    return FALSE;
  }
  if (vmdsock_bind(serversock, port)) {
    msgErr << "Could not bind vmdcollab server to port " << port 
           << sendmsg;
    vmdsock_destroy(serversock);
    return FALSE;
  }
  vmdsock_listen(serversock);

  wkf_thread_t serverthread;
  if (wkf_thread_create(&serverthread,
                        serverproc,    // my thread routine
                        serversock     // context for thread
  )) {
    msgErr << "VMDCollab: unable to create server thread" << sendmsg;
  } else {
    msgInfo << "Starting VMDCollab bounce server." << sendmsg;
  }

  return TRUE;
}


void VMDCollab::stopserver() {
  if (!serversock) return;
  vmdsock_shutdown(serversock);
  // don't destroy; let the server thread do that
  serversock = NULL;
}


int VMDCollab::connect(const char *host, int port) {
  if (clientsock) {
    msgErr << "Already connected to another vmdcollab server" << sendmsg;
    return FALSE;
  }
  if (!(clientsock = vmdsock_create())) {
    msgErr << "Could not create socket." << sendmsg;
    return FALSE;
  }
  int numTries = 3;
  for (int i=0; i<numTries; i++) {
    if (vmdsock_connect(clientsock, host, port)) {
      msgErr << "Could not connect to vmdcollab server at " << host << ":" << port << sendmsg;
      msgErr << "Error: " << strerror(errno) << sendmsg;
    } else {
      // success
      return TRUE;
    }
    // sleep for a second; maybe the server just hasn't started yet
    vmd_sleep(1);
  }
  // failed
  msgErr << "VMDCollab giving up after " << numTries << " seconds." << sendmsg;
  vmdsock_destroy(clientsock);
  clientsock = NULL;
  return FALSE;
}


void VMDCollab::disconnect() {
  if (!clientsock) return;
  vmdsock_shutdown(clientsock);
  vmdsock_destroy(clientsock);
  clientsock = NULL;
}


int VMDCollab::check_event() {
  if (!clientsock) return FALSE;
  eval_in_progress = TRUE;
  char buf[VMDCOLLAB_MSGSIZE];
  while (vmdsock_selread(clientsock, 0) > 0) {
    if (imd_readn(clientsock, buf, VMDCOLLAB_MSGSIZE) != VMDCOLLAB_MSGSIZE) {
      vmdsock_shutdown(clientsock);
      vmdsock_destroy(clientsock);
      clientsock = NULL;
      break;
    }
    runcommand(new TclEvalEvent(buf));
  }
  eval_in_progress = FALSE;
  return TRUE;
}


int VMDCollab::act_on_command(int, Command *cmd) {
  if (!clientsock) return FALSE;
  if (eval_in_progress) return FALSE;
  if (!cmd->has_text(cmdbufstr)) return TRUE;

  const char *txtcmd = cmdbufstr->text();
  int len = strlen(txtcmd);
  if (len >= VMDCOLLAB_MSGSIZE) {
    msgWarn << "VMDCollab: command too long: " << txtcmd << sendmsg;
    return FALSE;
  }

  char buf[VMDCOLLAB_MSGSIZE];
  strcpy(buf, txtcmd);
  cmdbufstr->reset();

  imd_writen(clientsock, buf, VMDCOLLAB_MSGSIZE);
  // give the server thread a chance to propagate events before continuing
  vmd_msleep(1);
  return TRUE;
}

