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
 *      $RCSfile: CmdIMD.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.27 $       $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Commands for IMD simulation control
 ***************************************************************************/


#ifndef CMDIMD_H__
#define CMDIMD_H__

#include "Command.h"

/// Connect to a running IMD simulation
class CmdIMDConnect : public Command {

public:
  CmdIMDConnect(int id, const char *hostname, int port);
  virtual ~CmdIMDConnect();

  int molid;
  char *host;
  int port;

protected:
  virtual void create_text();
};


/// Change IMD connection status (pause, detach, kill)
class CmdIMDSim : public Command {
public:
  enum CmdIMDSimCommand {
    PAUSE_TOGGLE,
    PAUSE_ON,
    PAUSE_OFF,
    DETACH,
    KILL
  };

  CmdIMDSim(CmdIMDSimCommand);

protected: 
  virtual void create_text();

private:
  CmdIMDSimCommand cmd;
};

/// Set the IMD transfer rate and storage mode
class CmdIMDRate : public Command {
public:
  enum CmdIMDRateCommand { TRANSFER, KEEP };
  CmdIMDRate(CmdIMDRateCommand, int);

protected: 
  virtual void create_text();

private:
  CmdIMDRateCommand rate_type;
  int rate;
};

/// Set the IMD unit cell copy mode
class CmdIMDCopyUnitCell : public Command {
public:
  enum CmdIMDCopyUnitCellCommand { COPYCELL_OFF=0, COPYCELL_ON=1 };
  CmdIMDCopyUnitCell(CmdIMDCopyUnitCellCommand);

protected: 
  virtual void create_text();

private:
  CmdIMDCopyUnitCellCommand copy_mode;
};

#endif

