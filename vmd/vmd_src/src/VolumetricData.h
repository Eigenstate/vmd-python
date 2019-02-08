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
 *	$RCSfile: VolumetricData.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.47 $	$Date: 2019/01/17 21:21:02 $
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
                           ///< stored x varying fastest, then y, then z.

  /// constructors for both single- and double-precision axes
  VolumetricData(const char *name, const float *origin, 
                 const float *xaxis, const float *yaxis, const float *zaxis,
                 int xs, int ys, int zs, float *dataptr);

  VolumetricData(const char *name, const double *origin, 
                 const double *xaxis, const double *yaxis, const double *zaxis,
                 int xs, int ys, int zs, float *dataptr);

  /// destructor
  ~VolumetricData();

  /// return total number of gridpoints
  long gridsize() const { return long(xsize)*long(ysize)*long(zsize); }

  /// return min/max data values
  void datarange(float &min, float &max);

  /// Sets data name to an internal copy of the provided string
  void set_name(const char* name);

  /// return the mean value of the density map, 
  /// implemented via O(1) access to a cached value
  float mean();

  /// return the standard deviation (sigma) of the density map, 
  /// implemented via O(1) access to a cached value
  float sigma();

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
  /// return voxel value based on cartesian coordinates. safe versions
  /// return zero if coordinates are outside the map.
  float voxel_value_from_coord_safe(float xpos, float ypos, float zpos) const;
  float voxel_value_interpolate_from_coord_safe(float xpos, float ypos, float zpos) const;


  /// get read-only access to the gradient
  const float *access_volume_gradient();

  /// provide the volume gradient
  void set_volume_gradient(float *gradient);

  /// (re)compute the volume gradient
  void compute_volume_gradient(void);


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

  /// get the cartesian coordinate of a voxel given its x,y,z indices
  inline void voxel_coord(int x, int y, int z, 
                          float &gx, float &gy, float &gz) const {
    float xdelta[3], ydelta[3], zdelta[3];
    cell_axes(xdelta, ydelta, zdelta);
    
    gx = origin[0] + (x * xdelta[0]) + (y * ydelta[0]) + (z * zdelta[0]);
    gy = origin[1] + (x * xdelta[1]) + (y * ydelta[1]) + (z * zdelta[1]);
    gz = origin[2] + (x * xdelta[2]) + (y * ydelta[2]) + (z * zdelta[2]);
  }

  /// get the cartesian coordinate of a voxel given its 1-D index 
  inline void voxel_coord(long i, float &x, float &y, float &z) const {
    float xdelta[3], ydelta[3], zdelta[3];
    cell_axes(xdelta, ydelta, zdelta);
    
    long gz = i / (ysize*xsize);
    long gy = (i / xsize) % ysize;
    long gx = i % xsize;

    x = origin[0] + (gx * xdelta[0]) + (gy * ydelta[0]) + (gz * zdelta[0]);
    y = origin[1] + (gx * xdelta[1]) + (gy * ydelta[1]) + (gz * zdelta[1]);
    z = origin[2] + (gx * xdelta[2]) + (gy * ydelta[2]) + (gz * zdelta[2]);
  }

  //
  //--unary ops--
  // 
  
  /// add or remove voxels in the given axis directions
  void pad(int padxm, int padxp, int padym, int padyp, int padzm, int padzp);
  
  /// crop a volumetric data to a given set of cartesian coordinates
  void crop(double crop_minx, double crop_miny, double crop_minz, double crop_maxx, double crop_maxy, double crop_maxz);

  /// clamp out of range voxel values
  void clamp(float min_value, float max_value);

  /// scales voxel data by given amount
  void scale_by(float ff);

  /// add scalar value to to all voxels
  void scalar_add(float ff);

  /// rescale voxel data to a given range
  void rescale_voxel_value_range(float min_value, float max_value);

  /// decimate/dowmnsample voxels by 2 in each dimension (x8 total reduction)
  void downsample();

  /// refine/supersample voxels by 2 in each dimension (x8 total increase)
  void supersample();

  /// Transform map to a sigma scale, so that isovalues in VMD correspond
  /// to number of sigmas above the mean
  void sigma_scale();

  /// Make a binary mask out of a map, i.e. map values > threshold 
  /// are set to 1, and all others are set to 0.
  void binmask(float threshold=0.0f);
  
  /// Guassian blur by sigma
  void gaussian_blur(double sigma);
  
  /// Create a potential for use with MDFF
  void mdff_potential(double threshold);

private:
  float *gradient;            ///< negated normalized volume gradient map
  bool gradient_isvalid;      ///< gradient map is current/valid
  void invalidate_gradient(); ///< invalidate the cached min/max value

  bool minmax_isvalid;        ///< cached min/max values are current/valid
  float cached_min;           ///< cached min voxel values
  float cached_max;           ///< cached max voxel values 

  bool mean_isvalid;          ///< cached mean value is current/valid
  float cached_mean;          ///< cached mean value
 
  bool sigma_isvalid;         ///< cached sigma value is current/valid
  float cached_sigma;         ///< cached sigma value

  void compute_minmaxmean();  ///< compute min/max/mean, encache results
  void invalidate_minmax();   ///< invalidate the cached min/max value
  void compute_minmax();      ///< compute and encache min/max voxel values

  void invalidate_mean();     ///< invalidate the cached mean value
  void compute_mean();        ///< compute and encache the mean voxel value

  void invalidate_sigma();    ///< invalidate the cached sigma value
  void compute_sigma();       ///< compute and encache sigma for voxel values


  /// Cubic interpolation used by supersample
  inline float cubic_interp(float y0, float y1, float y2, float y3, float mu) const {
    float mu2 = mu*mu;
    float a0 = y3 - y2 - y0 + y1;
    float a1 = y0 - y1 - a0;
    float a2 = y2 - y0;
    float a3 = y1;

    return (a0*mu*mu2+a1*mu2+a2*mu+a3);
  }
  
};


//
// Fast and loose accessor macros, don't use unless you have to
// 

/// fast but unsafe macro for querying volume gradients
#define VOXEL_GRADIENT_FAST_IDX(gradientmap, index, newgrad) \
  { (newgrad)[0] = (gradientmap)[index    ]; \
    (newgrad)[1] = (gradientmap)[index + 1]; \
    (newgrad)[2] = (gradientmap)[index + 2]; \
  }

/// fast but unsafe macro for querying volume gradients
#define VOXEL_GRADIENT_FAST(gradientmap, planesz, rowsz, x, y, z, newgrad) \
  { long index = ((z)*(planesz) + (y)*(rowsz) + (x)) * 3L; \
    (newgrad)[0] = (gradientmap)[index    ]; \
    (newgrad)[1] = (gradientmap)[index + 1]; \
    (newgrad)[2] = (gradientmap)[index + 2]; \
  }

#endif // VOLUMETRICDATA_H
