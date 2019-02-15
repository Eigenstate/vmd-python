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
 *      $RCSfile: IMDSimBlocking.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.18 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  A single-threaded implementation of the interactive MD 
 *  coordinate/force update communication loop 
 ***************************************************************************/

#include <string.h>
#include "vmdsock.h"
#include "IMDMgr.h"
#include "IMDSimBlocking.h"
#include "Inform.h"

IMDSimBlocking::IMDSimBlocking(const char *host, int port) 
: IMDSim(host, port) { 
  curpos = NULL;
}

IMDSimBlocking::~IMDSimBlocking() {
  delete [] curpos;
}

void IMDSimBlocking::update() {
  if (!isConnected()) return;
  IMDType type;
  int32 length;
  while (isConnected() && vmdsock_selread(sock,0)) {
    type = imd_recv_header(sock, &length);
    switch (type) {
      case IMD_FCOORDS: process_coordinates(length); break;
      case IMD_ENERGIES: process_energies(length);   break; 
      case IMD_MDCOMM: process_mdcomm(length);       break;
      case IMD_IOERROR: disconnect(); break;
      default: break;  // Don't need to read data 
    }
  }
}

void IMDSimBlocking::process_coordinates(int32 length) {
  if (numcoords < length) { // Need to resize
    delete [] curpos;
    curpos = new float[3L*length];
  }
  numcoords = length;
  if (imd_recv_fcoords(sock, numcoords, curpos)) {
    msgErr << "Error reading remote coordinates!" << sendmsg;
    disconnect();
  } else {
    new_coords_ready = 1;
  }
}

void IMDSimBlocking::process_energies(int32 /* length */) {
  if (imd_recv_energies(sock, &imdEnergies)) {
    msgErr << "Error reading energies!" << sendmsg;
    disconnect();   
  }
  else {
    if (need2flip) swap4_aligned(&imdEnergies, sizeof(imdEnergies) / 4);
  }
}

// This should never happen, but I'll handle it in case it does
void IMDSimBlocking::process_mdcomm(int32 length) {
  int32 *ind = new int32[length];
  float *f = new float[3L*length];
  if (imd_recv_mdcomm(sock, length, ind, f)) {
    msgErr << "Error reading MDComm-style forces!" << sendmsg;
    disconnect();
  }
  delete [] ind;
  delete [] f;
}

void IMDSimBlocking::get_next_ts(float *pos, IMDEnergies *buf) {
  new_coords_ready = 0;
  memcpy(pos, curpos, 3L*numcoords*sizeof(float));
  memcpy(buf, &imdEnergies, sizeof(IMDEnergies));
  if (need2flip) swap4_aligned(pos, 3L*numcoords);
}

void IMDSimBlocking::send_forces(int num, int *ind, float *forces) {
  // Total data sent will be one int and three floats for each atom 
  if (!isConnected()) return;
  if (need2flip) {
    swap4_aligned(ind, num);
    swap4_aligned(forces, 3L*num);
  }
  if (imd_send_mdcomm(sock, num, ind, forces)) {
    msgErr << "Error sending MDComm indices+forces" << sendmsg;
    disconnect();
  }
}

void IMDSimBlocking::pause() {
  if (isConnected() && (getSimState() == IMDRUNNING)) {
    simstate = IMDOFFLINE;
    imd_pause(sock);
  }
}

void IMDSimBlocking::unpause() {
  if (isConnected() && (getSimState() == IMDPAUSED)) {
    simstate = IMDRUNNING;    
    imd_pause(sock);
  }
}

void IMDSimBlocking::detach() {
  if (isConnected()) {
    simstate = IMDOFFLINE;
    imd_disconnect(sock);
  }
  disconnect();
}

void IMDSimBlocking::kill() {
  if (isConnected()) {
    simstate = IMDOFFLINE;
    imd_kill(sock);
  }
  disconnect();
}

void IMDSimBlocking::set_transrate(int rate) {
  if (isConnected())
    imd_trate(sock, rate);
}

