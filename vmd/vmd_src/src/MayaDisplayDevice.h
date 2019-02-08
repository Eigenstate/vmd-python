/***************************************************************************
 *cr
 *cr		(C) Copyright 1995-2019 The Board of Trustees of the
 *cr			    University of Illinois
 *cr			     All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: MayaDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.5 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Use to make Maya ASCII scene files for direct import into Autodesk Maya.
 *
 ***************************************************************************/

#ifndef MayaDISPLAYDEVICE
#define MayaDISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to Maya ASCII format
class MayaDisplayDevice : public FileRenderer {
private:
  int objnameindex;
  int oldColorIndex;
  int oldMaterialIndex;
  int oldMaterialState;
  void write_material_block(void);        ///< write full material table
  void write_cindexmaterial(const char *, int, int); ///< write colors/materials
  void write_colormaterial(float *, int); ///< write colors, materials etc.

protected:
  void beginrepgeomgroup(const char *);
  void comment(const char *);
  void line(float *xyz1, float *xyz2);
  void point(float *xyz);
  void sphere(float *xyzr);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
#if 0
  void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float * c1,   const float * c2,   const float * c3);
#endif

public:
  MayaDisplayDevice(void);            // constructor
  virtual ~MayaDisplayDevice(void);   // destructor
  virtual int open_file(const char *filename);
  virtual void close_file(void);
  void write_header (void);
  void write_trailer(void);
};

#endif

