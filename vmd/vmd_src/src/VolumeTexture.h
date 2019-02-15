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
 *      $RCSfile: VolumeTexture.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.9 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class for managing volumetric texture maps for use by the various
 *   DrawMolItem representation methods.
 ***************************************************************************/

#ifndef VOLUME_TEXTURE_H_
#define VOLUME_TEXTURE_H_

class VolumetricData;
class Scene;

class VolumeTexture {
public:
  // constructor - initialize values, no memory allocation.
  VolumeTexture();
  // Destructor
  ~VolumeTexture();
  
  // Assign reference to grid data.  Caller must ensure that the
  // lifetime of the volumetric data is at least as long as the VolumeTexture
  // instance; for VMD this should be fine since we currently never delete
  // VolumetricData instances.  This invalidates any previously generated
  // texture maps.
  void setGridData(VolumetricData *);

  //
  // routines to generate texture maps from the grid data.
  //
  
  /// color voxels by their position  
  void generatePosTexture();

  /// color voxels by their index
  void generateIndexTexture();

  // a charge-oriented texturing method
  void generateChargeTexture(float datamin, float datamax);

  // HSV color ramp
  void generateHSVTexture(float datamin, float datamax);

  // VMD color scale color ramp
  void generateColorScaleTexture(float datamin, float datamax, const Scene *);

  void generateContourLineTexture(float densityperline, float linewidth);

  // Get an ID for the current texture; this gets incremented whenever the
  // texture changes.
  unsigned long getTextureID() const { return texid; }

  // Get the size of the current texture along x/y/z axes.
  const int *getTextureSize() const { return size; }

  // Return a pointer to the texture map.  This data is allocated using
  // vmd_alloc and will exist for the lifetime of the VolumeTexture instance.
  unsigned char *getTextureMap() const { return texmap; }

  // Calculate texgen plane equations for the current texture.
  void calculateTexgenPlanes(float v0[4], float v1[4], float v2[4], float v3[4]) const;

private:
  VolumetricData *v;
  unsigned char *texmap;
  int size[3];
  unsigned long texid;

  // copy and operator= disallowed
  VolumeTexture(VolumeTexture &) {}
  VolumeTexture &operator=(VolumeTexture &) { return *this; }

  // compute texel count from current size[] 
  long num_texels(void) { return long(size[0])*long(size[1])*long(size[2]); }

  // allocate texture memory for n texels (3n bytes), and update texid.  
  // Return success.
  int allocateTextureMap(long ntexels);
};

#endif

