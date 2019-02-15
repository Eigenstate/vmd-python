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
 *      $RCSfile: IMDSim.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.25 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  A class to handle the low-level setup and teardown of interactive MD
 *  simulations.
 ***************************************************************************/
#ifndef IMD_SIM_H
#define IMD_SIM_H

#include "imd.h"

/// handle the low-level setup and teardown of interactive MD simulations.
class IMDSim {
public:
  /// Currently, VMD stores the simulation's state internally, but ideally
  /// these states should be communicated by NAMD
  enum IMDStates {IMDOFFLINE,IMDSTARTING,IMDRUNNING,IMDPAUSED};
    
  /// initialize with host and port
  IMDSim(const char *, int);
  virtual ~IMDSim();

  int isConnected() const { return sock != 0; }
  int getSimState() const { return simstate; }
  int next_ts_available() const { return new_coords_ready; }

  /// Check for available data from the socket
  virtual void update() {}

  /// Fetch last received coordinates and energies.  Ask for both at the same
  /// time so that they at least have a chance of being in sync with each other.
  virtual void get_next_ts(float *, IMDEnergies *) = 0;
  virtual void send_forces(int, int *, float *) = 0;

  virtual void pause() {}
  virtual void unpause() {}
    
  virtual void detach() {}
  virtual void kill() {}
  virtual void set_transrate(int) {}
  
protected:
  void *sock;
  int new_coords_ready;
  int numcoords;
  int simstate;             ///< One of enum IMDStates
  int need2flip;            ///< need to convert endianism

  void disconnect();
  static void swap4_aligned(void *data, long ndata); ///< reverse endianism of 4 bytes
   
private:
  void handshake();
};
    
#endif
