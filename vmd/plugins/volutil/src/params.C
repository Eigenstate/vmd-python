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
 *	$RCSfile: params.C,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.3 $	$Date: 2009/11/08 20:49:22 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "params.h"
#include "volmap.h"
#include "getplugins.h"

// From sysexits.h
#ifndef EX_SUCCESS
#define EX_SUCCESS 0
#endif
#ifndef EX_USAGE
#define EX_USAGE 64
#endif

// Initialize to default values
void Parameters::init() {
  do_exp = false;
  do_log = false;

  do_trim = false;
  do_crop = false;
  do_downsample = false;
  do_downsample_pmf = false;
  do_supersample = false;

  do_collapse = false;
  do_collapse_pmf = false;

  do_rotate_avg = false;
  do_rotate_avg_pmf = false;

  do_occup = false;

  do_multiply = false;
  do_add = false;
  do_subtract = false;
  do_average = false;
  do_combinepmf = false;
  do_compare = false;

  do_correlate = false;
  do_correlate_map = false;

  do_sigma_scale = false;
  threshold_sigmas = -FLT_MAX;
  volume_threshold = -FLT_MAX;

  do_custom1 = false;
  do_custom2 = false;

  use_padding = false;
  use_union = false;
  use_nointerp = false; 
  use_safe = false;

  do_histogram = false;

  histogram_nbins = 10;

  do_binmask = false;
  do_invmask = false;

  clamp_min = -FLT_MAX;
  clamp_max = FLT_MAX;

  fit_to_range_min = kNAN;
  fit_to_range_max = kNAN;

  dock_grid_max = FLT_MAX;

  smooth_radius = 0.;
  smooth_radius_pmf = 0.;

  scale_by = 1.;
  scalar_add = 0.;

  trimxp = 0;
  trimyp = 0;
  trimzp = 0;
  trimxm = 0;
  trimym = 0;
  trimzm = 0;

  crop_minx = FLT_MAX;
  crop_miny = FLT_MAX;
  crop_minz = FLT_MAX;
  crop_maxx = FLT_MAX;
  crop_maxy = FLT_MAX;
  crop_maxz = FLT_MAX;

  outputname = NULL;
  inputname1 = NULL;
  inputname2 = NULL;

  filelist = NULL;
  numinputfiles = 0;

  flagsbits = 0;

}

#define SETPARAM_ERR(X) fprintf(stderr, "ERROR: Failed to set parameter for option -%s\n", X); exit(EX_USAGE);

#define SETPARAM_INT(X,var) else if (!strcmp(argv[i]+1, (X))) { \
  if (argv[++i] == NULL || sscanf(argv[i], "%d", &var) != 1) { \
    SETPARAM_ERR(X) \
  } else continue; \
}

#define SETPARAM_DOUBLE(X,var) else if (!strcmp(argv[i]+1, (X))) { \
  if (argv[++i] == NULL || sscanf(argv[i], "%lf", &var) != 1) { \
    SETPARAM_ERR(X) \
  } else continue; \
}

#define SETPARAM_6_DOUBLE(X,var1,var2,var3,var4,var5,var6) else if (!strcmp(argv[i]+1, (X))) { \
  if (argv[++i] == NULL || sscanf(argv[i], "%lf", &var1) != 1 || \
      argv[++i] == NULL || sscanf(argv[i], "%lf", &var2) != 1 || \
      argv[++i] == NULL || sscanf(argv[i], "%lf", &var3) != 1 || \
      argv[++i] == NULL || sscanf(argv[i], "%lf", &var4) != 1 || \
      argv[++i] == NULL || sscanf(argv[i], "%lf", &var5) != 1 || \
      argv[++i] == NULL || sscanf(argv[i], "%lf", &var6) != 1) { \
    SETPARAM_ERR(X) \
  } else continue; \
} 

#define SETPARAM_DOUBLE_RANGE(X,var1,var2) else if (!strcmp(argv[i]+1, (X))) { \
  if (argv[++i] == NULL || \
     (sscanf(argv[i], "%lf:%lf", &var1, &var2) != 2 && \
      sscanf(argv[i], "%lf: ", &var1) != 1 && \
      sscanf(argv[i], " :%lf", &var2) != 1)) { \
    SETPARAM_ERR(X) \
  } else continue; \
} \

#define SETPARAM_FLAG(SET,UNSET,var) \
  else if (!strcmp(argv[i]+1, (SET)) && strcmp("", (SET))) \
  var = 1; \
  else if (!strcmp(argv[i]+1, (UNSET)) && strcmp("", (UNSET))) \
  var = 0;
  
#define SETPARAM_STRING(X,var) else if (!strcmp(argv[i]+1, (X))) { \
  if (argv[++i] == NULL) { \
    SETPARAM_ERR(X) \
  } else var = argv[i]; continue; \
} 

//char *infostring=NULL;

/* Parse command line
*/	
Parameters::Parameters(int argc, char** argv) {
  bool err_too_many_inputs = false;
  bool need_two_inputs = false;
  
  init();
  
  bool print_usage = false;
  int rc = 0;

  if (argc == 1) {
    print_usage = true;
    rc = EX_USAGE;
  } else if (!strcmp(argv[1],"--help")) {
    print_usage = true;
    rc = EX_SUCCESS;
  }

  if (print_usage) {
    fprintf(stderr, "volutil 1.3\n");
    fprintf(stderr, "usage: volutil [options] <map1> [map2]\n");
    fprintf(stderr, "map operations:\n");
    fprintf(stderr, "  -downsample|-ds   downsample by x2 (x8 total reduction)\n");
    fprintf(stderr, "  -supersample      supersample by x2 (x8 total increase)\n");
    fprintf(stderr, "  -trim <trim>      trim grid by specified amount in x, y and z\n");
    fprintf(stderr, "  -trimx <x>        trim grid by specified amount in x (or y,z)\n");
    fprintf(stderr, "  -trimxp <x>       trim one side of grid by specified amount in x (or y,z)\n");
    fprintf(stderr, "  -trimxm <x>       trim other side of grid by specified amount in x (or y,z)\n");
    fprintf(stderr, "  -crop <minx miny minz maxx maxy maxz>\n");
    fprintf(stderr, "  -exp              convert pmf to density\n");
    fprintf(stderr, "  -log              convert density to pmf\n");
    fprintf(stderr, "  -smooth <radius>  3D gaussian blur\n");
    fprintf(stderr, "  -collapse         project maps onto z axis\n");
    fprintf(stderr, "  -clamp <min:max>  clamp densities out of range\n");
    fprintf(stderr, "  -smult <x>        multiply every voxel by x\n");
    fprintf(stderr, "  -sadd <x>         add a scalar <x> to every voxel\n");
    fprintf(stderr, "  -range <min:max>  fit data to the given range\n");
    fprintf(stderr, "  -binmask          make a binary mask of the map\n");
    
    // XXX - The option -invmask should be commented out. It is here to
    // avoid three consecutive calls to volutil when making an edge smoothing
    // mask. Need to extend further -EV
    //     // fprintf(stderr, "  -invmask          make an inverse binary mask of the map\n");
    
    // XXX - The option -dockgrid should be kept commented out! It is 
    // here solely to avoid three consecutive calls to volutil when
    // setting up a flexible fitting simulations, but the user of volutil
    // doesn't need to know about it. -LT
    //
    // fprintf(stderr, "  -dockgrid <max>   create a map for flexible fitting with NAMD\n");

    // XXX - The option -threshold should be kept commented out! It
    // is used as an intermediate step in the calculation of local
    // cross-correlation coefficients between atomic structures and
    // electron microscopy maps. The user of volutil doesn't need to
    // know about it.
    //
    //fprintf(stderr, "  -threshold <sigmas> set voxels below threshold to NAN\n");
    
    fprintf(stderr, "  -sigma            transform map to sigma scale\n");
    fprintf(stderr, "binary operations:\n");
    fprintf(stderr, "  -average          average the input maps into one\n");
    fprintf(stderr, "  -add              add: map1 + map2\n");
    fprintf(stderr, "  -diff             subtract: map1 - map2\n");
    fprintf(stderr, "  -mult             multiply: map1 * map2\n");
    //fprintf(stderr, "  -cmp              compare map1 with map2\n");
    fprintf(stderr, "  -corr             calculate correlation coefficient\n");
    //fprintf(stderr, "  -coormap          calculate a correlation map\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -o <filename>     write output map to file\n");
    fprintf(stderr, "  -union            use union in binary operations\n");
    fprintf(stderr, "  -nointerp         do not use interpolation in binary operations\n");
    fprintf(stderr, "  -pad              use padding with -smooth\n");
    //fprintf(stderr, "  -safe            use safe version of -corr\n");
    fprintf(stderr, "operations with a PMF version:\n");
    fprintf(stderr, "  -dspmf, smoothpmf, collapsepmf, combinepmf\n");

    fprintf(stderr, "\nAllowed input file types: %s\n", plugins_read_volumetric_data());
    fprintf(stderr, "\nAllowed output file types: %s\n", plugins_write_volumetric_data());

    exit(rc);
  }

  // XXX - I don't like to have this hardcoded limit here -LT
  #define MAXFILES 200
  numinputfiles = 0;
  filelist = (char**) malloc(MAXFILES*sizeof(char*));
  memset(filelist, 0, MAXFILES*sizeof(char*));
  
  int trimx = 0;
  int trimy = 0;
  int trimz = 0;
  int trim = 0;

  // PARSE OPTIONS AND FILES
  int i;
  for (i=1; i<argc; i++)
    if (argv[i][0] == '-'){

      if (0); //NECESSARY
      
      SETPARAM_FLAG("exp", "", do_exp)
      SETPARAM_FLAG("log", "", do_log)
                
      SETPARAM_FLAG("downsample", "", do_downsample)
      SETPARAM_FLAG("dspmf", "", do_downsample_pmf)
      SETPARAM_FLAG("ds", "", do_downsample)
      SETPARAM_FLAG("supersample", "", do_supersample)
          
      SETPARAM_FLAG("collapsepmf", "", do_collapse_pmf)
      SETPARAM_FLAG("collapse", "", do_collapse)
      
      SETPARAM_FLAG("avgrotpmf", "", do_rotate_avg_pmf)
      SETPARAM_FLAG("avgrot", "", do_rotate_avg)
            
      SETPARAM_FLAG("combinepmf", "", do_combinepmf)
      SETPARAM_FLAG("average", "", do_average)
      SETPARAM_FLAG("multiply", "", do_multiply)
      SETPARAM_FLAG("mult", "", do_multiply)
      SETPARAM_FLAG("add", "", do_add)
      SETPARAM_FLAG("subtract", "", do_subtract)
      SETPARAM_FLAG("diff", "", do_subtract)
      SETPARAM_FLAG("cmp", "", do_compare)

      SETPARAM_FLAG("corr", "", do_correlate)
      SETPARAM_FLAG("coormap", "", do_correlate_map)

      SETPARAM_FLAG("count", "", do_occup)

      SETPARAM_FLAG("hist", "", do_histogram)


      SETPARAM_INT("trimxp", trimxp)
      SETPARAM_INT("trimyp", trimyp)
      SETPARAM_INT("trimzp", trimzp)
      SETPARAM_INT("trimxm", trimxm)
      SETPARAM_INT("trimym", trimym)
      SETPARAM_INT("trimzm", trimzm)

      SETPARAM_6_DOUBLE("crop", crop_minx, crop_miny, crop_minz, crop_maxx, crop_maxy, crop_maxz)
                
      SETPARAM_INT("trimx", trimx)
      SETPARAM_INT("trimy", trimy)
      SETPARAM_INT("trimz", trimz)
      SETPARAM_INT("trim", trim)

      SETPARAM_FLAG("custom1", "", do_custom1)
      SETPARAM_FLAG("custom2", "", do_custom2)

      SETPARAM_FLAG("pad", "", use_padding)
      SETPARAM_FLAG("union", "", use_union)
      SETPARAM_FLAG("nointerp", "", use_nointerp)
      SETPARAM_FLAG("safe", "", use_safe)

      SETPARAM_FLAG("sigma", "", do_sigma_scale)

      SETPARAM_FLAG("binmask", "", do_binmask)
      SETPARAM_FLAG("invmask", "", do_invmask)

      SETPARAM_DOUBLE_RANGE("clamp", clamp_min, clamp_max)
      SETPARAM_DOUBLE_RANGE("range", fit_to_range_min, fit_to_range_max)

      SETPARAM_DOUBLE("dockgrid", dock_grid_max)
      SETPARAM_DOUBLE("threshold", threshold_sigmas)
      SETPARAM_DOUBLE("volume", volume_threshold)

      SETPARAM_DOUBLE("smoothpmf", smooth_radius_pmf)
      SETPARAM_DOUBLE("smooth", smooth_radius)

      SETPARAM_DOUBLE("smult", scale_by)
      SETPARAM_DOUBLE("sadd", scalar_add)


      SETPARAM_INT("histnbins", histogram_nbins)
                  
      SETPARAM_STRING("o", outputname)
            		
      else {
        fprintf(stderr, "ERROR: Unknown parameter: %s\n", argv[i]);
        exit(EX_USAGE);
      }
    }
    else {
      if (numinputfiles < MAXFILES) {
        filelist[numinputfiles] = argv[i];
        numinputfiles++;
      } 
      else err_too_many_inputs = true;
      
      if (!inputname1) inputname1 = argv[i];
      else if (!inputname2) inputname2 = argv[i];
    }

  //post-processing
  if (!inputname1) {
    fprintf(stderr, "ERROR: No input files given!\n");
    exit(EX_USAGE);
  }	else {
  //  printf("INPUT 1: %s\n", inputname1);
  }

  if (histogram_nbins < 0) {
    fprintf(stderr, "ERROR: Number of histogram bins need to be positive\n");
    exit(EX_USAGE);
  }

  if (!ISNAN(fit_to_range_min) || !ISNAN(fit_to_range_max)) {
    if (ISNAN(fit_to_range_min) || ISNAN(fit_to_range_max) ||
      !(fit_to_range_min > -FLT_MAX && fit_to_range_min < FLT_MAX &&
        fit_to_range_max > -FLT_MAX && fit_to_range_max < FLT_MAX &&
        fit_to_range_min <= fit_to_range_max)) {
      fprintf(stderr, "ERROR: Invalid range to option -range\n");
      exit(EX_USAGE);
    }
  }

  if (clamp_min > clamp_max) {
    fprintf(stderr, "ERROR: Invalid range to option -clamp\n");
    exit(EX_USAGE);
  }

  if (smooth_radius < 0.0f || smooth_radius_pmf < 0.0f) {
    fprintf(stderr, "ERROR: Radius of Gaussian kernel must be positive\n");
    exit(EX_USAGE);
  }

  if (do_correlate && use_union) {
    fprintf(stderr, "ERROR: Calculation of correlation using union is not supported\n");
    exit(EX_USAGE);
  }
 
  if (do_subtract) need_two_inputs = true;
  if (need_two_inputs && !inputname2) {
    fprintf(stderr, "ERROR: The operations requested require 2 input files!\n");
    exit(EX_USAGE);
  }
  else if (need_two_inputs) {
  //  printf("INPUT 2: %s\n", inputname2);
  }
    
//  if (err_too_many_inputs || (numinputfiles > 2 && !do_combinepmf && !do_combine)) {
//    fprintf(stderr, "Too many input files given (max allowed: 2)!\n");
//    exit(EX_USAGE);
//  }
  
  if (trimx == 0) trimx = trim;
  if (trimy == 0) trimy = trim;
  if (trimz == 0) trimz = trim;
  if (trimxp == 0) trimxp = trimx;
  if (trimyp == 0) trimyp = trimy;
  if (trimzp == 0) trimzp = trimz;
  if (trimxm == 0) trimxm = trimx;
  if (trimym == 0) trimym = trimy;
  if (trimzm == 0) trimzm = trimz;
  if (trimxp || trimxm || trimyp || trimym || trimzp || trimzm) do_trim = true;

  if (crop_minx*crop_minx < FLT_MAX &&
      crop_miny*crop_miny < FLT_MAX &&
      crop_minz*crop_minz < FLT_MAX &&
      crop_maxx*crop_maxx < FLT_MAX &&
      crop_maxy*crop_maxy < FLT_MAX &&
      crop_maxz*crop_maxz < FLT_MAX ) {
    do_crop = true;
  }


  /* Set flags */
  if (use_padding) flagsbits |= USE_PADDING;
  if (use_union) flagsbits |= USE_UNION;
  if (!use_nointerp) flagsbits |= USE_INTERP;
  if (use_safe) flagsbits |= USE_SAFE;


}


