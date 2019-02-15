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
 *      $RCSfile: PlainTextInterp.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.13 $       $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Last resort text interpreter if no other is available
 ***************************************************************************/

#include "PlainTextInterp.h"
#include "Inform.h"
#include "utilities.h"
#include <stdlib.h>

PlainTextInterp::PlainTextInterp() {
  msgInfo << "Starting VMD text interpreter..." << sendmsg; 
}

PlainTextInterp::~PlainTextInterp() {
  msgInfo << "Exiting VMD text interpreter." << sendmsg;
}

int PlainTextInterp::evalString(const char *s) {
  vmd_system(s);
  return 0;
}

void PlainTextInterp::appendString(const char *s) {
  msgInfo << s << sendmsg;
}

void PlainTextInterp::appendList(const char *s) {
  msgInfo << s << sendmsg;
}

 
