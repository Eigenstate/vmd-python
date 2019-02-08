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
 *      $RCSfile: IMDMgr.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.26 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  High level interactive MD simulation management and update routines.
 ***************************************************************************/
#ifndef IMD_MGR_H__
#define IMD_MGR_H__

#include "UIObject.h"
#include "imd.h"

class Molecule;
class MoleculeIMD;
class IMDSim;

/// High level interactive MD simulation management and update management
class IMDMgr : public UIObject {
public:
  IMDMgr(VMDApp *);
  ~IMDMgr();

  void set_trans_rate(int);
  int  get_trans_rate() const {return Trate; }

  void set_keep_rate(int);
  int  get_keep_rate() const { return keep_rate; }

  void set_copyunitcell(int);
  int  get_copyunitcell() const { return copy_unit_cell; }

  /// Get the name of the remote host, or NULL if none was found.
  const char *gethost() const {return host; }
  int getport() const {return port;}
  int connected() {return sim != 0; }

  /// XXX should be const Molecule *
  Molecule *get_imdmol() {return mol; }

  /// Connect to interactive simulation and return whether successful. 
  int connect(Molecule *, const char *, int);

  /// Send forces to remote simulation
  int send_forces(int, const int *, const float *);

  void pause();
  void unpause();
  void togglepause();
  void detach();
  void kill();

  int check_event();

  int act_on_command(int, Command *);

private:
  Molecule *mol;
  IMDSim *sim;
  IMDEnergies *energies;
  char *host;
  int port;
  int Trate;
  int keep_rate;
  int copy_unit_cell;
  int frames_received;
};

#endif

