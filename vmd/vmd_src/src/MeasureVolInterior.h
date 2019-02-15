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
 *      $RCSfile: MeasureVolInterior.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Method for measuring the interior volume in a vesicle or capsid based    
 *   on an externally-provided simulated density map (e.g., from QuickSurf)
 *   and a threshold isovalue for inside/outside tests based on density.
 *   The approach computes and counts interior/exterior voxels based on 
 *   a parallel ray casting approach on an assumed orthorhombic grid.
 *   Juan R. Perilla - 2018
 *
 ***************************************************************************/
VolumetricData* CreateEmptyGrid(const VolumetricData *);  
void VolInterior_CleanGrid(VolumetricData *);
long RaycastGrid(const VolumetricData *, VolumetricData *, float, float *);  
long volin_threaded(const VolumetricData *, VolumetricData *, float, float *);
long countIsoGrids(const VolumetricData *, const float);
long markIsoGrid(const VolumetricData *, VolumetricData *, const float);

