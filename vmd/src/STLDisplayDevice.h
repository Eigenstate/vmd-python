/***************************************************************************
 *cr
 *cr		(C) Copyright 1995-2011 The Board of Trustees of the
 *cr			    University of Illinois
 *cr			     All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: STLDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.23 $	$Date: 2010/12/16 04:08:38 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Use to make stereolithography files. See http://www.sdsc.edu/tmf/
 *
 ***************************************************************************/

#ifndef STLDISPLAYDEVICE
#define STLDISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to STL solid model format
class STLDisplayDevice : public FileRenderer {
protected:
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);

public:
  STLDisplayDevice(void);            // constructor
  virtual ~STLDisplayDevice(void);   // destructor
  void write_header (void);
  void write_trailer(void);
};

#endif

