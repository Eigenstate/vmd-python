/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: ops.h,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.1 $	$Date: 2009/08/06 20:58:46 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *
 ***************************************************************************/

#ifndef _OPS_H_
#define _OPS_H_

typedef enum {Regular, PMF} Ops;


class Operation {
public:
  virtual char  *name() = 0;
  virtual bool   trivial() = 0;
  virtual double ConvertValue(double) = 0;
  virtual double ConvertAverage(double) = 0;
};



// Get an 
Operation *GetOps(Ops optype);

#endif
