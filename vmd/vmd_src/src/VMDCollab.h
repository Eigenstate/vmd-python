/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: VMDCollab.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.6 $      $Date: 2016/11/28 03:05:05 $
 *
 ***************************************************************************/

#ifndef VMDCOLLAB_H
#define VMDCOLLAB_H

#include "UIObject.h"
#include "ResizeArray.h"
class Inform;

class VMDCollab : public UIObject {

private:
  void *serversock;
  static void *serverproc(void *v);

  Inform *cmdbufstr;
  void *clientsock;
  bool eval_in_progress;

public:
  // normal constructor
  VMDCollab(VMDApp *);
  ~VMDCollab();

  // start a vmdcollab server on localhost using the given port.  
  // Return success.
  int startserver(int port);
  
  void stopserver();

  // connect to a vmdcollab server at the given host/port.  Fails if
  // already connected to another server.  Return success.
  int connect(const char *host, int port);

  // close client connection to collab server
  void disconnect();

  // virtual UIObject methods
  int check_event();
  int act_on_command(int, Command *);
};

#endif
