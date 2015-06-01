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
 *	$RCSfile: params.h,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.2 $	$Date: 2009/11/08 20:49:22 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *
 ***************************************************************************/

#ifndef _PARAMS_H
#define _PARAMS_H

class Parameters {
public:
 bool do_exp;
 bool do_log;

 bool do_trim;
 bool do_crop;
 bool do_downsample;
 bool do_downsample_pmf;
 bool do_supersample;
 bool do_collapse;
 bool do_collapse_pmf;

 bool do_rotate_avg;
 bool do_rotate_avg_pmf;

 bool do_occup;

 bool do_histogram;
 int histogram_nbins;

 bool do_multiply;
 bool do_add;
 bool do_subtract;
 bool do_average;
 bool do_combinepmf;

 bool do_correlate;
 bool do_correlate_map;
 bool do_compare;
  
 bool do_sigma_scale;
 double threshold_sigmas;
 double volume_threshold;

 bool do_binmask;
 bool do_invmask;

 double smooth_radius;
 double smooth_radius_pmf;

 double scale_by; 
 double scalar_add;

 int trimxp;  // trim positive x side
 int trimyp;
 int trimzp;
 int trimxm;  // trim negative x side
 int trimym;
 int trimzm;

 int padxp;   // pad positive x size
 int padyp;
 int padzp;
 int padxm;   // pad negative x size
 int padym;
 int padzm;

 double crop_minx;
 double crop_miny;
 double crop_minz;
 double crop_maxx;
 double crop_maxy;
 double crop_maxz;

 char *inputname1, *inputname2, *outputname;

 char **filelist;
 int numinputfiles;

 bool do_custom1;
 bool do_custom2;

 double clamp_min;
 double clamp_max;

 double fit_to_range_min;
 double fit_to_range_max;

 double dock_grid_max;

 bool use_padding;
 bool use_union;
 bool use_nointerp;
 bool use_safe;

 unsigned int flagsbits;
 

private:
  void init();
    
public:
  // Read from command-line
  Parameters(int argc, char** argv); 
};

#endif
