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
 *      $RCSfile: IMDSimThread.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  A multithreaded implementation of the interactive MD 
 *  coordinate/force communication update loop.
 ***************************************************************************/
#ifndef IMD_SIM_THREAD_H__
#define IMD_SIM_THREAD_H__

#include "IMDSim.h"
#include "WKFThreads.h" 

/// A multithreaded implementation of the interactive MD
/// coordinate/force communication update loop.
class IMDSimThread : public IMDSim {
public:
  /// initialize with host and port
  IMDSimThread(const char *, int);
  virtual ~IMDSimThread();

  virtual void get_next_ts(float *, IMDEnergies *);

  /// These methods obtain a lock on the socket (to prevent it from being 
  /// destroyed from the reader thread), check for a connection, then send
  /// their information before releasing the lock.
  virtual void send_forces(int, int *, float *);
  virtual void pause();
  virtual void unpause();
  virtual void detach();
  virtual void kill();
  virtual void set_transrate(int);

  /// reader method must be public to be callable from extern "C" thread proc
  void *reader(void *);

private:
  /// Posbuf1 and posbuf2 each hold space for coordinates.  curpos points to
  /// the buffer holding the last completed set, and curbuf points to the
  /// read buffer.  When a set of coordinates has been read, the reader thread
  /// swaps curpos and curbuf, after obtaining a lock on coordmutex.
  float *curpos, *curbuf;      
  float *posbuf1, *posbuf2;

  /// The reader thread locks energymutex when it's reading into this buffer.
  /// This is not a problem as long as reading energies doesn't take too long.
  IMDEnergies imdEnergies; 

  /// The connection is checked and read in an independent thread running
  /// the update function below.  update() checks if deadsocket has been
  /// set to 1; if it has, it breaks the loop, destroys the socket (after first 
  /// obtaining a lock), and exits.  This is the only way the socket is ever
  /// destroyed.  
  int deadsocket;
  void process_coordinates(int32);
  void process_energies(int32);
  void process_mdcomm(int32);

  wkf_thread_t readerthread;
  wkf_mutex_t sockmutex;   ///< guards sock when we disconnect and destroy
  wkf_mutex_t coordmutex;  ///< guards curpos, new_coords_ready, and energies

  /// The class destructor sets this variable to cause the reader thread to
  /// exit gracefully.  
  int time2die;
  
};

#endif
 
