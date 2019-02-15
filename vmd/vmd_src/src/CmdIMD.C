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
 *      $RCSfile: CmdIMD.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.32 $       $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Commands for IMD simulation control
 ***************************************************************************/


#include "CmdIMD.h"
#include "utilities.h"

////////////////// IMDConnect

CmdIMDConnect::CmdIMDConnect(int id, const char *hostname, int portnum)
: Command(Command::IMD_ATTACH) {
  host = stringdup(hostname);
  port = portnum;
  molid = id; 
}

CmdIMDConnect::~CmdIMDConnect() {
  delete [] host;
}

void CmdIMDConnect::create_text() {
  *cmdText << "imd connect " << host << " " << port << ends;
}

//////////////// IMDSim
CmdIMDSim::CmdIMDSim(CmdIMDSimCommand newcmd)
: Command(Command::IMD_SIM), cmd(newcmd) {}

void CmdIMDSim::create_text() {
  const char *text;
  switch (cmd) {
    case PAUSE_TOGGLE:  text="pause toggle"; break;
    case PAUSE_ON:  text="pause on"; break;
    case PAUSE_OFF:  text="pause off"; break;
    case DETACH: text="detach"; break;
    case KILL:   text="kill"; break;
    default:     text=""; 
  }
  *cmdText << "imd sim " << text << ends;
}

//////////////// IMDRate
CmdIMDRate::CmdIMDRate(CmdIMDRateCommand type, int newrate)
: Command(Command::IMD_RATE), rate_type(type), rate(newrate) {}

void CmdIMDRate::create_text() {
  switch (rate_type) {
    case TRANSFER:
      *cmdText << "imd transrate " << rate << ends;
      break;
    case KEEP:
      *cmdText << "imd keep " << rate << ends;
      break;
    default: ;
  }
}

//////////////// IMDCopyUnitCell
CmdIMDCopyUnitCell::CmdIMDCopyUnitCell(CmdIMDCopyUnitCellCommand mode) : Command(Command::IMD_COPYUNITCELL), copy_mode(mode) {}

void CmdIMDCopyUnitCell::create_text() {
  *cmdText << "imd copyunitcell " << ((copy_mode == CmdIMDCopyUnitCell::COPYCELL_ON) ? "on" : "off") << ends;
}


