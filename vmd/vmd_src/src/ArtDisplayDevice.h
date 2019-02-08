#ifndef ARTDISPLAYDEVICE_H
#define ARTDISPLAYDEVICE_H

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
 *	$RCSfile: ArtDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.27 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION: 
 *   Writes to the ART raytracer.  This is available from gondwana.ecr.mu.oz.au
 * as part of the vort package.
 *
 ***************************************************************************/


#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to ART ray tracer scene format
class ArtDisplayDevice : public FileRenderer {
private:
  int Initialized;    ///< was the output file created?

protected:
  // assorted graphics functions
  void comment(const char *);
  void cone(float *, float *, float, int); 
  void cylinder(float *, float *, float,int filled);
  void line(float *, float *);
  void point(float *);
  void sphere(float *);
  void square(float *, float *, float *, float *, float *);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);

public: 
  ArtDisplayDevice();
  virtual ~ArtDisplayDevice(void);
  void write_header(void);
  void write_trailer(void);
}; 

#endif

