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
*      $RCSfile: LibGelatoDisplayDevice.h
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.8 $         $Date: 2019/01/17 21:20:59 $
*
***************************************************************************
* DESCRIPTION:
*
* FileRenderer type for the Gelato interface.
*
***************************************************************************/

#ifndef LIBGELATODISPLAYDEVICE
#define LIBGELATODISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"

/// forward declaration of Gelato API handle
class GelatoAPI;

/// FileRenderer subclass to exports VMD scenes to Gelato PYG scene format
class LibGelatoDisplayDevice: public FileRenderer {
private:
  GelatoAPI *gapi;

  /// keep track of what the last written material properties are,
  /// that way we can avoid writing redundant definitions.
  float old_color[3];
  float old_opacity;
  float old_ambient;
  float old_diffuse;
  float old_specular;

  void reset_vars(void); ///< reset internal state variables
  void write_materials(int write_color);

protected:
  void line(float *xyz1, float *xyz2);
  void point(float *xyz);
  void sphere(float *xyzr);
  void square(float *, float *, float *, float *, float *);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float *c1,    const float *c2,    const float *c3);
  virtual void trimesh_c4n3v3(int numverts, float * cnv, 
                              int numfacets, int * facets);
  virtual void tristrip(int numverts, const float * cnv,
                        int numstrips, const int *vertsperstrip,
                        const int *facets);

public: 
  LibGelatoDisplayDevice(void);
  virtual ~LibGelatoDisplayDevice(void);
  void write_header(void); 
  void write_trailer(void);
}; 

#endif

