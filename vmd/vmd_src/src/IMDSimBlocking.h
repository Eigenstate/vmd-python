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
 *      $RCSfile: IMDSimBlocking.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.15 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  A single-threaded implementation of the interactive MD 
 *  coordinate/force update communication loop.
 ***************************************************************************/

#ifndef IMD_SIM_BLOCKING_H__
#define IMD_SIM_BLOCKING_H__

#include "IMDSim.h"

/// A single-threaded implementation of the interactive MD
/// coordinate/force update communication loop.
class IMDSimBlocking : public IMDSim {
public:
  /// initialize with host and port
  IMDSimBlocking(const char *, int);
  virtual ~IMDSimBlocking();

  /// Check for available data from the socket
  virtual void update();

  virtual void get_next_ts(float *, IMDEnergies *);
  virtual void send_forces(int, int *, float *);

  virtual void pause();
  virtual void unpause();
    
  virtual void detach();
  virtual void kill();
  virtual void set_transrate(int);

private:
  float *curpos;       ///< last complete set of coordinates
  IMDEnergies imdEnergies;
 
  void process_coordinates(int32);
  void process_energies(int32);
  void process_mdcomm(int32);
};

#endif
 
