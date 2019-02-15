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
 *	$RCSfile: VMDTitle.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.33 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * A flashy title object which is displayed when the program starts up,
 * until a molecule is loaded.
 *
 ***************************************************************************/
#ifndef VMDTITLE_H
#define VMDTITLE_H

#include "Displayable.h"
#include "DispCmds.h"

/// Displayable subclass for a flashy title object displayed when VMD starts up,
/// until a molecule is loaded.
class VMDTitle : public Displayable {
private:
  DisplayDevice *disp;
  DispCmdColorIndex color;
  double starttime;
  void redraw_list(void);
  int letterson;
  
public:
  VMDTitle(DisplayDevice *, Displayable *);
  virtual void prepare(); ///< prepare to draw
};

#endif

