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
 *	$RCSfile: QuickSurf.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.21 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Fast gaussian surface representation
 ***************************************************************************/
#ifndef QUICKSURF_H
#define QUICKSURF_H

#include "AtomSel.h"
#include "ResizeArray.h"
#include "Isosurface.h"
#include "WKFUtils.h"
#include "VolumetricData.h"
#include "VMDDisplayList.h"

class CUDAQuickSurf;

class QuickSurf {
private:
  float *volmap;           ///< Density map
  float *voltexmap;        ///< Volumetric texture map in RGB format
  IsoSurface s;            ///< Isosurface computed on the CPU
  float isovalue;          ///< Isovalue of the surface to extract
  float solidcolor[3];     ///< RGB color to use when not using per-atom colors
  int numvoxels[3];        ///< Number of voxels in each dimension
  float origin[3];         ///< Origin of the volumetric map
  float xaxis[3];          ///< X-axis of the volumetric map
  float yaxis[3];          ///< Y-axis of the volumetric map
  float zaxis[3];          ///< Z-axis of the volumetric map

  int force_cpuonly;       ///< If we need CPU-fallback for MDFF, we set this flag in the constructor
  CUDAQuickSurf *cudaqs;   ///< Pointer to CUDAQuickSurf object if it exists

  wkf_timerhandle timer;   ///< Internal timer for performance instrumentation
  double pretime;          ///< Internal timer for performance instrumentation
  double voltime;          ///< Internal timer for performance instrumentation
  double gradtime;         ///< Internal timer for performance instrumentation
  double mctime;           ///< Internal timer for performance instrumentation
  double mcverttime;       ///< Internal timer for performance instrumentation
  double reptime;          ///< Internal timer for performance instrumentation

public:
  QuickSurf(int forcecpuonly=0);
  ~QuickSurf(void);

  void free_gpu_memory(void); ///< experimental mechanism to free GPU memory
                              ///< when needed for other things like 
                              ///< OptiX ray tracing 

  int calc_surf(AtomSel * atomSel, DrawMolecule *mymol, 
                const float *atompos, const float *atomradii,
                int quality, float radscale, float gridspacing, float isovalue,
                const int *colidx, const float *cmap, VMDDisplayList *cmdList); 

  VolumetricData * calc_density_map(AtomSel * atomSel, DrawMolecule *mymol, 
                                    const float *atompos, 
                                    const float *atomradii,
                                    int quality, float radscale, 
                                    float gridspacing);

private:
  int get_trimesh(int &numverts, float *&v3fv, float *&n3fv, float *&c3fv,
                  int &numfacets, int *&fiv);

  int draw_trimesh(VMDDisplayList *cmdList);

};

#endif

