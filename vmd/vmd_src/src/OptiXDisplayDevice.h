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
*      $RCSfile: OptiXDisplayDevice.h
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.25 $         $Date: 2019/01/17 21:21:00 $
*
***************************************************************************
* DESCRIPTION:
*   FileRenderer type for the OptiX interface.
*
* This work is described in:
*  "GPU-Accelerated Molecular Visualization on
*   Petascale Supercomputing Platforms"
*   John E. Stone, Kirby L. Vandivort, and Klaus Schulten.
*   UltraVis'13: Proceedings of the 8th International Workshop on
*   Ultrascale Visualization, pp. 6:1-6:8, 2013.
*   http://dx.doi.org/10.1145/2535571.2535595
*
*  "Atomic Detail Visualization of Photosynthetic Membranes with
*   GPU-Accelerated Ray Tracing"
*   John E. Stone, Melih Sener, Kirby L. Vandivort, Angela Barragan,
*   Abhishek Singharoy, Ivan Teo, João V. Ribeiro, Barry Isralewitz,
*   Bo Liu, Boon Chong Goh, James C. Phillips, Craig MacGregor-Chatwin,
*   Matthew P. Johnson, Lena F. Kourkoutis, C. Neil Hunter, and Klaus Schulten
*   J. Parallel Computing, 55:17-27, 2016.
*   http://dx.doi.org/10.1016/j.parco.2015.10.015
*
*  "Immersive Molecular Visualization with Omnidirectional
*   Stereoscopic Ray Tracing and Remote Rendering"
*   John E. Stone, William R. Sherman, and Klaus Schulten.
*   High Performance Data Analysis and Visualization Workshop,
*   2016 IEEE International Parallel and Distributed Processing
*   Symposium Workshops (IPDPSW), pp. 1048-1057, 2016.
*   http://dx.doi.org/10.1109/IPDPSW.2016.121
*
*  "Omnidirectional Stereoscopic Projections for VR"
*   John E. Stone.
*   In, William R. Sherman, editor, 
*   VR Developer Gems, Taylor and Francis / CRC Press, Chapter 24, 2019.
*
*  "Interactive Ray Tracing Techniques for
*   High-Fidelity Scientific Visualization"
*   John E. Stone.
*   In, Eric Haines and Tomas Akenine-Möller, editors,
*   Ray Tracing Gems, Apress, 2019.
*
*  "A Planetarium Dome Master Camera"
*   John E. Stone.
*   In, Eric Haines and Tomas Akenine-Möller, editors,
*   Ray Tracing Gems, Apress, 2019.
*
* Portions of this code are derived from Tachyon:
*   "An Efficient Library for Parallel Ray Tracing and Animation"
*   John E. Stone.  Master's Thesis, University of Missouri-Rolla,
*   Department of Computer Science, April 1998
*
*   "Rendering of Numerical Flow Simulations Using MPI"
*   John Stone and Mark Underwood.
*   Second MPI Developers Conference, pages 138-141, 1996.
*   http://dx.doi.org/10.1109/MPIDC.1996.534105
*
***************************************************************************/

#ifndef LIBOPTIXDISPLAYDEVICE
#define LIBOPTIXDISPLAYDEVICE

#include <stdio.h>
#include "Matrix4.h"
#include "FileRenderer.h"
#include "WKFUtils.h" // timers

/// forward declarations of OptiX API types 
class OptiXRenderer;

#if !defined(LIBOPTIXRENDERER)
/// remote device access
typedef void * RTRDev;
#endif

/// FileRenderer subclass to exports VMD scenes to OptiX
class OptiXDisplayDevice: public FileRenderer {
private:
  int isinteractive;
  RTRDev remote_device;
  OptiXRenderer * ort;
  wkf_timerhandle ort_timer;

  void reset_vars(void); ///< reset internal state variables
  void write_lights(void);
  void write_materials(void);
  void add_material(void);

#if 0
  // state tracking for volumetric texturing
  int involtex;               ///< volume texturing is enabled
  int voltexID;               ///< current volume texturing ID
  float xplaneeq[4];          ///< volumetric texture plane equations
  float yplaneeq[4];
  float zplaneeq[4];

  // state tracking for user-defined clipping planes
  int inclipgroup;            ///< whether a clipping group is currently active
#endif

  // storage and state variables needed to aggregate lone cylinders
  // with common color and material state into a larger buffer for
  // transmission to OptiX
  int cylinder_matindex;
  Matrix4 *cylinder_xform;
  float cylinder_radius_scalefactor;
  ResizeArray<float> cylinder_vert_buffer;
  ResizeArray<float> cylinder_radii_buffer;
  ResizeArray<float> cylinder_color_buffer;
  // cylinder end caps, made from rings
  ResizeArray<float> cylcap_vert_buffer;
  ResizeArray<float> cylcap_norm_buffer;
  ResizeArray<float> cylcap_radii_buffer;
  ResizeArray<float> cylcap_color_buffer;
  

  /// reset cylinder buffer to empty state
  void reset_cylinder_buffer() {
    cylinder_matindex = -1; 
    cylinder_xform = NULL;
    cylinder_radius_scalefactor=1.0f; 
    cylinder_vert_buffer.clear();
    cylinder_radii_buffer.clear();
    cylinder_color_buffer.clear();

    cylcap_vert_buffer.clear();
    cylcap_norm_buffer.clear();
    cylcap_radii_buffer.clear();
    cylcap_color_buffer.clear();
  };


  // storage and state variables needed to aggregate lone triangles
  // with common color and material state into a larger buffer for
  // transmission to OptiX
  int triangle_cindex;
  int triangle_matindex;
  Matrix4 *triangle_xform;
  ResizeArray<float> triangle_vert_buffer;
  ResizeArray<float> triangle_norm_buffer;

  /// reset triangle buffer to empty state
  void reset_triangle_buffer() {
    triangle_cindex = -1;   
    triangle_matindex = -1; 
    triangle_xform = NULL;
    triangle_vert_buffer.clear();
    triangle_norm_buffer.clear();
  };

protected:
  void send_cylinder_buffer(void);
  void cylinder(float *, float *, float rad, int filled);
  void sphere_array(int num, int res, float *centers, 
                    float *radii, float *colors);
  void text(float *pos, float size, float thickness, const char *str);
  void send_triangle_buffer(void);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float * c1,   const float * c2,   const float * c3);
  void trimesh_c4u_n3b_v3f(unsigned char *c, char *n, float *v, int numfacets);
  void trimesh_c4u_n3f_v3f(unsigned char *c, float *n, float *v, int numfacets);
  void trimesh_c4n3v3(int numverts, float * cnv, int numfacets, int * facets);
  void trimesh_n3b_v3f(char *n, float *v, int numfacets);
  void trimesh_n3f_v3f(float *n, float *v, int numfacets);
  void trimesh_n3fopt_v3f(float *n, float *v, int numfacets);
  void tristrip(int numverts, const float * cnv,
                int numstrips, const int *vertsperstrip,
                const int *facets);

public: 
  OptiXDisplayDevice(VMDApp *, int interactive, void *cluster_dev);
  virtual ~OptiXDisplayDevice(void);
  void write_header(void); 
  void write_trailer(void);
}; 

#endif

