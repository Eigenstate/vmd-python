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
 *      $RCSfile: CoorPluginData.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $       $Date: 2016/11/28 03:04:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  CoorPluginData: Uses a MolFilePlugin to load a coordinate file. 
 ***************************************************************************/
#ifndef COOR_PLUGIN_DATA_H
#define COOR_PLUGIN_DATA_H

#include "utilities.h"
#include "WKFUtils.h"
#include "CoorData.h"

class Molecule;
class MolFilePlugin;

/// CoorPluginData: Uses a MolFilePlugin to load a coordinate file
class CoorPluginData : public CoorData {
protected:
  MolFilePlugin *plugin;
  int is_input;
  int begFrame, frameSkip, endFrame, recentFrame;
  wkf_timerhandle tm;
  long kbytesperframe, totalframes;
  int *selection; ///< If non-NULL, an array of atom indices to be written

public:
  CoorPluginData(const char *nm, Molecule *m, MolFilePlugin *,
    int is_input, int firstframe=-1, int framestride=-1, int lastframe=-1,
    const int *sel = NULL);
  ~CoorPluginData();

  // read/write next coordinate set.  Return state 
  virtual CoorDataState next(Molecule *m);
};

#endif

