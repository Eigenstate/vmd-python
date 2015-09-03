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
 *	$RCSfile: volmap.h,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.6 $	$Date: 2009/11/08 20:49:22 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *
 ***************************************************************************/

#ifndef _VOLMAP_H
#define _VOLMAP_H

#include <math.h>
#include "ops.h"

extern const float kNAN;
#if defined(_MSC_VER)
#include <float.h>
#define ISNAN(X) _isnan(X)
#else
#define ISNAN(X) isnan(X)
#endif

#define MIN(X,Y) (((X)<(Y))? (X) : (Y))
#define MAX(X,Y) (((X)>(Y))? (X) : (Y))

enum {USE_PADDING=1, USE_UNION=2, USE_INTERP=4, USE_SAFE=8};

class VolMap {
private:
  enum {REQUIRE_ORDERED=1, REQUIRE_ORTHO=2, REQUIRE_UNIFORM=4, REQUIRE_NONSINGULAR=8};
  int  conditionbits;  
  char *refname;           /// name to call this map in stdout
  char *dataname;          /// dataset name XXX UNUSED
  
public:    
//  From VolumetricData.h :
//
//  char *name;              ///< human-readable volume dataset identifier
  double origin[3];        ///< origin of volume (x=0, y=0, z=0 corner)
  double xaxis[3];         ///< direction and length for X axis (non-unit)
  double yaxis[3];         ///< direction and length for Y axis (non-unit)
  double zaxis[3];         ///< direction and length for Z axis (non-unit)
  int xsize, ysize, zsize; ///< number of samples along each axis
  float *data;             ///< raw data, total of xsize*ysize*zsize voxels
//  float *gradient;         ///< negated normalized volume gradient map
//  float datamin, datamax;  ///< min and max data values 



  double xdelta[3];        /// x axis of unit cell
  double ydelta[3];        /// y axis of unit cell
  double zdelta[3];        /// z axis of unit cell


  double weight;
  double temperature;

  /* In volmap.C */   

private:
  void init();

public:       
  VolMap();
  VolMap(const VolMap *src);
  VolMap(const char *filename);
  ~VolMap();
  
  void clone(const VolMap *src);
  void zero();
  void dirty(); // call this when data/cell has been changed
  
  void set_dataname(const char *);
  void set_refname(const char *);
  const char *get_refname() const;

  bool condition(int cond);  
  void require(const char *funcname, int cond);
  
  void print_stats();

  
/* In unary_ops.C */
  
  void convert_pmf_to_density();
  void convert_density_to_pmf();
  
  void pad(int padxm, int parxp, int padym, int padyp, int padzm, int padzp);
  void crop(double crop_minx, double crop_miny, double crop_minz, double crop_maxx, double crop_maxy, double crop_maxz);
  void downsample(Ops ops=Regular);
  void supersample(Ops ops=Regular);
  void collapse_onto_z(Ops ops=Regular);
  
  void average_over_rotations(Ops ops=Regular);
  
  void total_occupancy();

  void clamp(float min_value, float max_value);
  void scale_by(float factor);
  void scalar_add(float x);
  void fit_to_range(float min_value, float max_value);

  void dock_grid(float max_value);
  void apply_threshold(float min_value);
  void apply_threshold_sigmas(double sigmas);
  void sigma_scale();
  void histogram(int nbins);
  void binmask();
  void invmask();

  void calc_volume();
  
  /* In convolutions.C */
  
  void convolution_gauss1d(double radius, unsigned int flagsbits, Ops ops=Regular);
  void convolution_gauss3d(double radius, unsigned int flagsbits, Ops ops=Regular);
  
  
  /* In binary_ops.C */
  
  void init_from_intersection(VolMap *mapA, VolMap *mapB);
  void init_from_union(VolMap *mapA, VolMap *mapB);
  void init_from_identity(VolMap *mapA);
  void chop_B_from_A(VolMap *mapA, VolMap *mapB);
      
  void add(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops ops=Regular);
  void multiply(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops ops=Regular);
  void subtract(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops ops=Regular);
  void average(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops ops=Regular);

  void correlate(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops ops=Regular);
  void correlate_map(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops ops=Regular);
  void compare(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops ops=Regular);

  void perform_recursively(char **files, int numfiles, unsigned int flagsbits, void (VolMap::*func)(VolMap*, VolMap*, unsigned int, Ops), Ops optype=Regular);
  
  
  
  /* In voxels.C */   
  
  /// return index of voxel in map, from x/y/z indices
  inline int voxel_index(int gx, int gy, int gz) const {
    return (gx + gy*xsize + gz*ysize*xsize);
  }
    
  /// return voxel at requested index, no safety checks
  inline float voxel_value(int gx, int gy, int gz) const {
    return data[gx + gy*xsize + gz*ysize*xsize];
  }
  
  inline void voxel_coord(int gx, int gy, int gz, float &x, float &y, float &z) const {
    x = origin[0] + (gx)*xdelta[0]; 
    y = origin[1] + (gy)*ydelta[1];
    z = origin[2] + (gz)*zdelta[2];
  }
  
  /// return voxel, after safely clamping index to valid range
  float voxel_value_safe(int x, int y, int z) const;
  float voxel_value_safe(int x, int y, int z, int myxsize, int myysize, int myzsize, float *mydata) const;

  /// return interpolated value from 8 nearest neighbor voxels
  float voxel_value_interpolate(float xv, float yv, float zv) const;
  float voxel_value_interpolate_pmf_exp(float xv, float yv, float zv) const;
    
  int   coord_to_index(float x, float y, float z) const;
  void  index_to_coord(int index, float &x, float &y, float &z) const;
  void  index_to_grid(int index, int &x, int &y, int &z) const;
  float voxel_value_from_coord(float xpos, float ypos, float zpos) const;
  float voxel_value_from_coord_safe(float xpos, float ypos, float zpos) const;
  
  float voxel_value_interpolate_from_coord(float xpos, float ypos, float zpos, Ops ops=Regular) const;
  float voxel_value_interpolate_from_coord_safe(float xpos, float ypos, float zpos, Ops ops=Regular) const;
  
  
  
  /* In io.C or molfile.C (depending on whether we are using the molfile lib) */
  int load (const char *filename);
  int write_old (const char *filename) const;
  int write (const char *filename) const;
  
};


#endif
