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
 *	$RCSfile: VolumetricData.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.34 $	$Date: 2016/11/28 03:05:06 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Base class for storing volumetric data and associated gradient data
 *
 ***************************************************************************/

#ifndef VOLUMETRICDATA_H
#define VOLUMETRICDATA_H

/// Volumetric data class for potential maps, electron density maps, etc
class VolumetricData {
public:
  double origin[3];        ///< origin of volume (x=0, y=0, z=0 corner)
  double xaxis[3];         ///< direction and length for X axis (non-unit)
  double yaxis[3];         ///< direction and length for Y axis (non-unit)
  double zaxis[3];         ///< direction and length for Z axis (non-unit)
  int xsize, ysize, zsize; ///< number of samples along each axis
  char *name;              ///< human-readable volume dataset identifier
  float *data;             ///< raw data, total of xsize*ysize*zsize voxels
  float *gradient;         ///< negated normalized volume gradient map
  float datamin, datamax;  ///< min and max data values 

  /// constructor
  VolumetricData(const char *name, const float *origin, 
                 const float *xaxis, const float *yaxis, const float *zaxis,
                 int xs, int ys, int zs, float *dataptr);

  VolumetricData(const char *name, const double *origin, 
                 const double *xaxis, const double *yaxis, const double *zaxis,
                 int xs, int ys, int zs, float *dataptr);

  /// destructor
  ~VolumetricData();

  /// return total number of gridpoints
  int gridsize() const { return xsize*ysize*zsize; }

  /// Sets data name to an internal copy of the provided string
  void set_name(const char* name);

  /// return cell side lengths
  void cell_lengths(float *xl, float *yl, float *zl) const;

  /// return cell axes
  void cell_axes(float *xax, float *yax, float *zax) const;

  /// return cell axes directions
  void cell_dirs(float *xax, float *yax, float *zax) const;

  /// return volumetric coordinate from cartesian coordinate
  void voxel_coord_from_cartesian_coord(const float *carcoord, float *voxcoord, int shiftflag) const;

  /// return index of the voxel nearest to a cartesian coordinate
  long voxel_index_from_coord(float xpos, float ypos, float zpos) const;

  /// return voxel at requested index, no safety checks
  inline float voxel_value(int x, int y, int z) const {
    return data[z*xsize*ysize + y*xsize + x];
  }

  /// return voxel, after safely clamping index to valid range
  float voxel_value_safe(int x, int y, int z) const;

  /// return interpolated value from 8 nearest neighbor voxels
  float voxel_value_interpolate(float xv, float yv, float zv) const;

  /// return voxel value based on cartesian coordinates
  float voxel_value_from_coord(float xpos, float ypos, float zpos) const;
  float voxel_value_interpolate_from_coord(float xpos, float ypos, float zpos) const;


  /// (re)compute the volume gradient
  void compute_volume_gradient(void);

  /// provide the volume gradient
  void set_volume_gradient(float *gradient);

  /// return gradient at requested index, no safety checks
  void voxel_gradient_fast(int x, int y, int z, float *grad) const {
    long index = (z*xsize*ysize + y*xsize + x) * 3;
    grad[0] = gradient[index    ];
    grad[1] = gradient[index + 1];
    grad[2] = gradient[index + 2];
  }

  /// return gradient, after safely clamping index to valid range
  void voxel_gradient_safe(int x, int y, int z, float *grad) const;

  /// interpolate the gradient between the eight neighboring voxels
  void voxel_gradient_interpolate(const float *voxcoord, float *gradient) const;

  /// return voxel gradient based on cartesian coordinates
  void voxel_gradient_from_coord(const float *coord, float *gradient) const;
  void voxel_gradient_interpolate_from_coord(const float *coord, float *gradient) const;

};


//
// Fast and loose accessor macros, don't use unless you have to
// 

/// fast but unsafe macro for querying volume gradients
#define VOXEL_GRADIENT_FAST_IDX(v, index, grad) \
  { (grad)[0] = v->gradient[index    ]; \
    (grad)[1] = v->gradient[index + 1]; \
    (grad)[2] = v->gradient[index + 2]; \
  }

/// fast but unsafe macro for querying volume gradients
#define VOXEL_GRADIENT_FAST(v, x, y, z, grad) \
  { long index = ((z)*v->xsize*v->ysize + (y)*v->xsize + (x)) * 3; \
    (grad)[0] = v->gradient[index    ]; \
    (grad)[1] = v->gradient[index + 1]; \
    (grad)[2] = v->gradient[index + 2]; \
  }

#endif // VOLUMETRICDATA_H
