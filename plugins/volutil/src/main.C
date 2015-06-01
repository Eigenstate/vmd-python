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
 *	$RCSfile: main.C,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.2 $	$Date: 2009/11/08 20:49:22 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Main program for volutil.
 *
 ***************************************************************************/

// Written by Jordi Cohen, 2005-2007
// Theoretical and Computational Biophysics Group
// University of Illinois, Urbana, IL


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "params.h"
#include "volmap.h"

Parameters* params=NULL;

int main(int argc, char** argv) {
  int err;

  // Test if ISNAN macro works in this platform
  if (ISNAN(kNAN) == 0) {
    fprintf(stderr, "ERROR: ISNAN macro failed.\n");
    return 1;
  }
      
  params = new Parameters(argc, argv);

  VolMap *pot = new VolMap();
  pot->set_refname("MAP");
  
  /* Do binary operations */
  
  if (params->do_add)
    pot->perform_recursively(params->filelist, params->numinputfiles, params->flagsbits, &VolMap::add);
  else if (params->do_compare)
    pot->perform_recursively(params->filelist, params->numinputfiles, params->flagsbits, &VolMap::compare);
  else if (params->do_multiply)
    pot->perform_recursively(params->filelist, params->numinputfiles, params->flagsbits, &VolMap::multiply);
  else if (params->do_subtract)
    pot->perform_recursively(params->filelist, params->numinputfiles, params->flagsbits, &VolMap::subtract);
  else if (params->do_average)
    pot->perform_recursively(params->filelist, params->numinputfiles, params->flagsbits, &VolMap::average);
  else if (params->do_combinepmf)
    pot->perform_recursively(params->filelist, params->numinputfiles, params->flagsbits, &VolMap::average, PMF);
  else if (params->do_correlate)
    pot->perform_recursively(params->filelist, params->numinputfiles, params->flagsbits, &VolMap::correlate);
  else if (params->do_correlate_map)
    pot->perform_recursively(params->filelist, params->numinputfiles, params->flagsbits, &VolMap::correlate_map);
  else {
    err = pot->load(params->inputname1);
    if (err) exit(1);
    printf("%s <- \"%s\"\n", pot->get_refname(), params->inputname1);
  }
  
  
  /* Do unary operations */
  
  if (params->do_rotate_avg) pot->average_over_rotations();
  else if (params->do_rotate_avg_pmf) pot->average_over_rotations(PMF);

  if (params->do_downsample) pot->downsample();
  else if (params->do_downsample_pmf) pot->downsample(PMF);

  if (params->do_supersample) pot->supersample();

  if (params->do_histogram) pot->histogram(params->histogram_nbins);

  if (params->smooth_radius) pot->convolution_gauss1d(params->smooth_radius, params->flagsbits);
  else if (params->smooth_radius_pmf) pot->convolution_gauss1d(params->smooth_radius_pmf, params->flagsbits, PMF);

  if (params->do_trim) pot->pad(-params->trimxm, -params->trimxp, -params->trimym, -params->trimyp, -params->trimzm, -params->trimzp);

  if (params->do_crop) pot->crop(params->crop_minx, params->crop_miny, params->crop_minz, params->crop_maxx, params->crop_maxy, params->crop_maxz);
    
  if (params->scale_by != 1.) pot->scale_by(params->scale_by);
  if (params->scalar_add != 0.) pot->scalar_add(params->scalar_add);
      
  if (params->do_exp) pot->convert_pmf_to_density();
  if (params->do_log) pot->convert_density_to_pmf();
  
  if (params->do_occup) pot->total_occupancy();

  if (params->clamp_min > -FLT_MAX || params->clamp_max < FLT_MAX) 
    pot->clamp(params->clamp_min, params->clamp_max);

  if (!ISNAN(params->fit_to_range_min) && !ISNAN(params->fit_to_range_max))
    pot->fit_to_range(params->fit_to_range_min, params->fit_to_range_max);

  if (params->dock_grid_max < FLT_MAX) 
    pot->dock_grid(params->dock_grid_max);

  if (params->threshold_sigmas > -FLT_MAX)
    pot->apply_threshold_sigmas(params->threshold_sigmas);

  if (params->volume_threshold > -FLT_MAX) {
    pot->apply_threshold_sigmas(params->volume_threshold);
    pot->calc_volume();
  }

  if (params->do_sigma_scale)
    pot->sigma_scale();
 
  if (params->do_invmask)
    pot->invmask();

  if (params->do_binmask)
        pot->binmask();

  /* Output */

  if (params->do_collapse) pot->collapse_onto_z();
  if (params->do_collapse_pmf) pot->collapse_onto_z(PMF);
    
  if (params->outputname)
    pot->write(params->outputname);
  else 
    printf("No output produced.\n");

  pot->print_stats();
    
  
  delete pot;
  
  return 0;
}
