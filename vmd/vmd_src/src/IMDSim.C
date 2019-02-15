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
 *      $RCSfile: IMDSim.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.32 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Routines to manage the low-level setup and teardown of Interactive MD 
 *   simulations.
 ***************************************************************************/

#include <stdlib.h>
#include "IMDSim.h"
#include "vmdsock.h"
#include "IMDMgr.h"
#include "Inform.h"

IMDSim::IMDSim(const char *host, int port) {
  
  new_coords_ready = 0;
  numcoords = 0;
  simstate = IMDOFFLINE;
  
  vmdsock_init(); // make sure Winsock interfaces are initialized
  sock = vmdsock_create();
  if (sock == NULL) {
    msgErr << "Error connecting: could not create socket" << sendmsg;
    return;
  }
  int rc = vmdsock_connect(sock, host, port);
  if (rc < 0) {
    msgErr << "Error connecting to " << host << " on port "<< port <<sendmsg;
    vmdsock_destroy(sock);
    sock = 0;
    return;
  }
  handshake();
  simstate = IMDRUNNING;
}

void IMDSim::disconnect() {
  simstate = IMDOFFLINE;
  if (sock) {
    imd_disconnect(sock);
    vmdsock_shutdown(sock);
    vmdsock_destroy(sock);
    sock = 0;
  }
}
 
IMDSim::~IMDSim() {
  disconnect();
}

// Handshake: currently this is a 'one-way' handshake: after VMD connects,
// NAMD sends to VMD an integer 1 in the length field of the header, without
// converting network byte orer.

void IMDSim::handshake() {

  need2flip = imd_recv_handshake(sock);
  switch (need2flip) {
    case 0:
      msgInfo << "Connected to same-endian machine" << sendmsg;
      break;
    case 1:
      msgInfo << "Connected to opposite-endian machine" << sendmsg;
      break;
    default:
      msgErr << "Unable to ascertain relative endianness of remote machine"
             << sendmsg;
      disconnect();
  }
}

/* Only works with aligned 4-byte quantities, will cause a bus error */
/* on some platforms if used on unaligned data.                      */
void IMDSim::swap4_aligned(void *v, long ndata) {
  int *data = (int *) v;
  long i;
  int *N;
  for (i=0; i<ndata; i++) {
    N = data + i;
    *N=(((*N>>24)&0xff) | ((*N&0xff)<<24) |
        ((*N>>8)&0xff00) | ((*N&0xff00)<<8));
  }
}

