/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: CUDAQuickSurf.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.4 $	$Date: 2012/09/06 04:02:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Fast gaussian surface representation
 ***************************************************************************/
#ifndef CUDAQUICKSURF_H
#define CUDAQUICKSURF_H

class VMDDisplayList;

class CUDAQuickSurf {
  void *voidgpu; ///< pointer to structs containing private per-GPU pointers 

public:
  enum VolTexFormat { RGB3F, RGB4U }; ///< which texture map format to use
 
  CUDAQuickSurf(void);
  ~CUDAQuickSurf(void);

  int calc_surf(long int natoms, const float *xyzr, const float *colors,
                int colorperatom, float *origin, int* numvoxels, float maxrad,
                float radscale, float gridspacing,
                float isovalue, float gausslim,
                VMDDisplayList *cmdList);

private:
  int free_bufs(void);

  int check_bufs(long int natoms, int colorperatom,
                 int gx, int gy, int gz);

  int alloc_bufs(long int natoms, int colorperatom, 
                 VolTexFormat vtexformat,
                 int gx, int gy, int gz);

  int get_chunk_bufs(int testexisting,
                     long int natoms, int colorperatom, 
                     VolTexFormat vtexformat,
                     int gx, int gy, int gz,
                     int &cx, int &cy, int &cz,
                     int &sx, int &sy, int &sz);


};

#endif

