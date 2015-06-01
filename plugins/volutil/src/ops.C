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
 *	$RCSfile: ops.C,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.1 $	$Date: 2009/08/06 20:58:46 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Operation provides a class to make abstraction of whether the data is a
 * density, or a PMF/energy map, etc. By translating every value on-the-fly, it
 * removes the need for writing two versions of each volmap transformation, 
 * and instead allows code to be multipurpose and reusable for different types
 * of data which require different types of averaging... 
 *
 ***************************************************************************/


#include <stdlib.h>
#include <math.h>

#include "ops.h"


static Operation *RegularStaticOps=NULL;
static Operation *PMFStaticOps=NULL;


class RegularOps : public Operation {
public:
  char  *name();
  bool   trivial()  {return true;}
  double ConvertValue(double val) {return val;}
  double ConvertAverage(double avg) {return avg;}
};

class PMFOps : public Operation {
public:
  char  *name();
  bool   trivial()  {return false;}
  double ConvertValue(double val);
  double ConvertAverage(double avg);
};


/* Regular Ops */

char *RegularOps::name() {
  return "regular";
}


/* PMF Ops */

char  *PMFOps::name() {
  return "PMF";
}


double PMFOps::ConvertValue(double val) {
  return exp(-val);
}


double PMFOps::ConvertAverage(double avg) {
  double val = -log(avg);
  if (val != val || val > 150.) val = 150.;
  return val;
}



Operation *GetOps(Ops optype) {
  if (optype == PMF) {
    if (!PMFStaticOps) PMFStaticOps = new PMFOps;
    return PMFStaticOps;
  }
  if (!RegularStaticOps) RegularStaticOps = new RegularOps;
  return RegularStaticOps;
}
