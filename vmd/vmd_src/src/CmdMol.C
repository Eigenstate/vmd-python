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
 *	$RCSfile: CmdMol.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.153 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Command objects for affecting molecules.
 *
 ***************************************************************************/

#include <stdlib.h>

#include "config.h"
#include "CmdMol.h"
#include "Inform.h"
#include "VMDDisplayList.h"

void CmdMolLoad::create_text(void) {
  *cmdText << "mol ";
  if (molid == -1) {
    *cmdText << "new {";
  } else {
    *cmdText << "addfile {";
  }

  *cmdText << name << "} type {" << type << "}" 
    << " first " << spec.first 
    << " last " << spec.last 
    << " step " << spec.stride 
    << " waitfor " << spec.waitfor;

  if (spec.autobonds == 0) 
    *cmdText << " autobonds " << spec.autobonds;

  if (spec.nvolsets > 0) {
    *cmdText << " volsets {";
    for (int i=0; i<spec.nvolsets; i++) {
      *cmdText << spec.setids[i] << " ";
    }
    *cmdText << "}";
  }

  if (molid != -1) {
    *cmdText << " " << molid;
  }
}

void CmdMolDelete::create_text(void) {
  *cmdText << "mol delete " << whichMol << ends;
}

void CmdMolCancel::create_text(void) {
  *cmdText << "mol cancel " << whichMol << ends;
}

void CmdMolActive::create_text(void) {
  *cmdText << "mol " << (yn ? "active " : "inactive ") << whichMol << ends;
}

void CmdMolFix::create_text(void) {
  *cmdText << "mol " << (yn ? "fix " : "free ") << whichMol << ends;
}

void CmdMolOn::create_text(void) {
  *cmdText << "mol " << (yn ? "on " : "off ") << whichMol << ends;
}

void CmdMolTop::create_text(void) {
  *cmdText << "mol top " << whichMol << ends;
}

void CmdMolSelect::create_text(void) {
  *cmdText << "mol selection ";
  if(sel)
    *cmdText << sel;
  *cmdText << ends;
}

void CmdMolRep::create_text(void) {
  *cmdText << "mol representation ";
  if(sel)
    *cmdText << sel;
  *cmdText << ends;
}

void CmdMolColor::create_text(void) {
  *cmdText << "mol color ";
  if(sel)
    *cmdText << sel;
  *cmdText << ends;
}

void CmdMolMaterial::create_text(void) {
  *cmdText << "mol material ";
  if (mat)
    *cmdText << mat;
  *cmdText << ends;
}

void CmdMolAddRep::create_text(void) {
  *cmdText << "mol addrep " << whichMol << ends;
}

void CmdMolChangeRep::create_text(void) {
  *cmdText << "mol modrep " << repn << " " << whichMol << ends;
}

void CmdMolChangeRepItem::create_text(void) {
  *cmdText << "mol mod";
  if (repData == COLOR)
    *cmdText << "color ";
  else if (repData == REP)
    *cmdText << "style ";
  else if (repData == SEL)
    *cmdText << "select ";
  else if (repData == MAT)
    *cmdText << "material ";
  *cmdText << repn << " " << whichMol << " " << str << ends;
}

void CmdMolRepSelUpdate::create_text() {
  *cmdText << "mol selupdate " << repn << " " << whichMol << " " << onoroff 
           << ends;
}

void CmdMolRepColorUpdate::create_text() {
  *cmdText << "mol colupdate " << repn << " " << whichMol << " " << onoroff 
           << ends;
}

void CmdMolDeleteRep::create_text(void) {
  *cmdText << "mol delrep " << repn << " " << whichMol << ends;
}

void CmdMolReanalyze::create_text(void) {
  *cmdText << "mol reanalyze " << whichMol << ends;
}

void CmdMolBondsRecalc::create_text(void) {
  *cmdText << "mol bondsrecalc " << whichMol << ends;
}

void CmdMolSSRecalc::create_text(void) {
  *cmdText << "mol ssrecalc " << whichMol << ends;
}

void CmdMolRename::create_text() {
  *cmdText << "mol rename " << whichMol << " {" << newname << "}" << ends;
}
CmdMolRename::CmdMolRename(int id, const char *nm)
: Command(MOL_RENAME), whichMol(id) {
  newname = strdup(nm);
}
CmdMolRename::~CmdMolRename() {
  free(newname);
}

void CmdMolShowPeriodic::create_text() {
  *cmdText << "mol showperiodic " << whichMol << " " << repn << " ";
  char buf[10];
  buf[0] = '\0';
  if (pbc & PBC_X) strcat(buf, "x");
  if (pbc & PBC_Y) strcat(buf, "y");
  if (pbc & PBC_Z) strcat(buf, "z");
  if (pbc & PBC_OPX) strcat(buf, "X");
  if (pbc & PBC_OPY) strcat(buf, "Y");
  if (pbc & PBC_OPZ) strcat(buf, "Z");
  if (pbc & PBC_NOSELF) strcat(buf, "n");
  *cmdText << buf << ends;
}

void CmdMolNumPeriodic::create_text() {
  *cmdText << "mol numperiodic " << whichMol << " " << repn << " " << nimages
           << ends;
}


void CmdMolShowInstances::create_text() {
  *cmdText << "mol showinstances " << whichMol << " " << repn << " ";
  char buf[10];
  buf[0] = '\0';
  if (instances == INSTANCE_NONE) strcat(buf, "none");
  else if (instances & INSTANCE_ALL) strcat(buf, "all");
  else if (instances & PBC_NOSELF) strcat(buf, "noself");
  *cmdText << buf << ends;
}


void CmdMolScaleMinmax::create_text() {
  *cmdText << "mol scaleminmax " << whichMol << " " << repn << " ";
  if (reset) {
    *cmdText << "auto";
  } else {
    *cmdText << scalemin << " " << scalemax;
  }
  *cmdText << ends;
}

void CmdMolDrawFrames::create_text() {
  *cmdText << "mol drawframes " << whichMol << " " << repn << " {" 
           << framespec << "}" << ends;
}

void CmdMolSmoothRep::create_text() {
  *cmdText << "mol smoothrep " << whichMol << " " << repn << " "
           << winsize << ends;
}

void CmdMolShowRep::create_text() {
  *cmdText << "mol showrep " << whichMol << " " << repn << " "
           << onoff << ends;
}

