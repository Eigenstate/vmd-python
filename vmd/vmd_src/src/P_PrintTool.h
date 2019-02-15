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
 *	$RCSfile: P_PrintTool.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.12 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************/

/// The print tool allows users to print tracker values on the fly
/** The print tool is the most basic of tools, as evidenced by the
 minimal amount of code it contains.  All it does is allow users
 to print the tracker coordinates using the functionality
 provided by UIVR. */

#include "P_Tool.h"
class PrintTool : public Tool {
 public:
  PrintTool(int id, VMDApp *, Displayable *);
  virtual void do_event();

  const char *type_name() const { return "print"; }
 private:
  int targetting;
};


