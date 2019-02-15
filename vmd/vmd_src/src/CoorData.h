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
 *      $RCSfile: CoorData.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.12 $       $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * CoorData
 * Abstract base class representing objects that periodically read or 
 * write new timesteps.  Next best thing to multithreading!
 *
 ***************************************************************************/
#ifndef COOR_DATA_H
#define COOR_DATA_H

#include <stdlib.h>
#include <string.h>

class Molecule;

/// Abstract base class for objects that periodically read/write timesteps
class CoorData {
public:
  char *name;

public:
  enum CoorDataState { DONE, NOTDONE };

  CoorData(const char *nm) {
    name = strdup(nm);
  }
  virtual ~CoorData() {
    free(name);
  }

  /// read/write next coordinate set.  Return state 
  virtual CoorDataState next(Molecule *m) = 0;
};

#endif 

