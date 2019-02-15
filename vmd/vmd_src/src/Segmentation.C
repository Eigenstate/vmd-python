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
 *      $RCSfile: Segmentation.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.28 $        $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Scale-space variant of Watershed image segmentation intended for use  
 *   on 3-D cryo-EM density maps.
 ***************************************************************************/

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string.h>
#include "Segmentation.h"
#include "Watershed.h"
#include "CUDASegmentation.h"
#include "CUDAGaussianBlur.h"
#include "Inform.h"
#include "ProfileHooks.h"

#ifdef TIMER
#include "WKFUtils.h"
#endif

#define INST_GET_RESULTS(G_T) \
  template void Segmentation::get_results<G_T>(G_T* results);

INST_GET_RESULTS(unsigned short)
INST_GET_RESULTS(short)
INST_GET_RESULTS(unsigned int)
INST_GET_RESULTS(int)
INST_GET_RESULTS(unsigned long)
INST_GET_RESULTS(long)
INST_GET_RESULTS(float)


Segmentation::Segmentation(const VolumetricData *map, bool cuda) {
  PROFILE_PUSH_RANGE("Segmentation::Segmentation(MAP)", 6);

  init(map->xsize, map->ysize, map->zsize, cuda);

  //XXX in future, check VolData to see which type we should use here
  image_type = DT_FLOAT;
  gaussian_f = new GaussianBlur<float>(map->data, width, height, depth, cuda);

  /* Example
   * image_type = DT_UCHAR;
   * gaussian_uc = new GaussianBlur<unsigned char>(map->data_c, width, height, depth, cuda);
   * */

  PROFILE_POP_RANGE();
}


void Segmentation::init(unsigned long w,
                        unsigned long h,
                        unsigned long d,
                        bool cuda) {
  width     = w;
  height    = h;
  depth     = d;
  nVoxels   = h * w * d;
  nGroups   = nVoxels;
  use_cuda  = cuda;
  group_type = get_new_group_type();
  image_type = DT_ERROR;
  segments_us = NULL;
  segments_ui = NULL;
  segments_ul = NULL;
  gaussian_f = NULL;
  gaussian_us = NULL;
  gaussian_uc = NULL;

  switch (group_type) {
    case DT_ULONG:
      segments_ul  = allocate_array<unsigned long>(nVoxels);
      break;
    case DT_UINT:
      segments_ui  = allocate_array<unsigned int>(nVoxels);
      break;
    case DT_USHORT:
      segments_us  = allocate_array<unsigned short>(nVoxels);
      break;
    default:
      fprintf(stderr, "Error: invalid segment type.\n");
      return;
  }
}



Segmentation::Segmentation(float* data,
                           molfile_volumetric_t* metadata,
                           bool cuda) {
  PROFILE_PUSH_RANGE("Segmentation::Segmentation(float *data)", 6);

  init(metadata->xsize, metadata->ysize, metadata->zsize, cuda);
  image_type = DT_FLOAT;

  gaussian_f = new GaussianBlur<float>(data, width, height, depth, use_cuda);

  PROFILE_POP_RANGE();
}


Segmentation::Segmentation(unsigned short* data,
                           molfile_volumetric_t* metadata,
                           bool cuda) {
  PROFILE_PUSH_RANGE("Segmentation::Segmentation(float *data)", 6);

  init(metadata->xsize, metadata->ysize, metadata->zsize, cuda);
  image_type = DT_USHORT;

  gaussian_us = new GaussianBlur<unsigned short>(data, width, height, depth, use_cuda);

  PROFILE_POP_RANGE();
}

Segmentation::Segmentation(unsigned char* data,
                           molfile_volumetric_t* metadata,
                           bool cuda) {
  PROFILE_PUSH_RANGE("Segmentation::Segmentation(float *data)", 6);

  init(metadata->xsize, metadata->ysize, metadata->zsize, cuda);
  image_type = DT_UCHAR;

  gaussian_uc = new GaussianBlur<unsigned char>(data, width, height, depth, use_cuda);

  PROFILE_POP_RANGE();
}


Segmentation::~Segmentation() {
  if (gaussian_f != NULL) {
    delete gaussian_f;
  }
  if (gaussian_us != NULL) {
    delete gaussian_us;
  }
  if (gaussian_uc != NULL) {
    delete gaussian_uc;
  }
  if (segments_ul != NULL) {
    free_array(segments_ul);
  }
  if (segments_ui != NULL) {
    free_array(segments_ui);
  }
  if (segments_us != NULL) {
    free_array(segments_us);
  }
}



double Segmentation::segment(int   num_final_groups,
                             float watershed_blur_sigma,
                             float blur_initial_sigma,
                             float blur_multiple,
                             MERGE_POLICY policy,
                             const bool isverbose) {
  PROFILE_PUSH_RANGE("Segmentation::segment()", 6);

  double time_sec = 0.0;
#ifdef TIMER
  wkf_timerhandle timer = wkf_timer_create();    
  wkf_timer_start(timer);
#endif
  
  verbose = isverbose;

  // Perform watershed segmentation
  switch (group_type) {
    case DT_ULONG:
      if (verbose)
        printf("Using unsigned long representation.\n");
      get_watershed_segmentation<unsigned long>(watershed_blur_sigma, segments_ul);
      break;
    case DT_UINT:
      if (verbose)
        printf("Using unsigned int representation.\n");
      get_watershed_segmentation<unsigned int>(watershed_blur_sigma, segments_ui);
      break;
    case DT_USHORT:
      if (verbose)
        printf("Using unsigned short representation.\n");
      get_watershed_segmentation<unsigned short>(watershed_blur_sigma, segments_us);
      break;
    default:
      fprintf(stderr, "Error: invalid segmentation group type.\n");
      PROFILE_POP_RANGE(); // error return point
      return 0;
  }


  // this would switch to smaller datatype for scale-space filtering
  // currently not implemented
  update_type();

  // merge groups until we have <= num_final_groups
  switch (group_type) {
    case DT_ULONG:
      merge_groups<unsigned long>(num_final_groups, blur_initial_sigma, blur_multiple, policy, segments_ul);
      break;
    case DT_UINT:
      merge_groups<unsigned int>(num_final_groups, blur_initial_sigma, blur_multiple, policy, segments_ui);
      break;
    case DT_USHORT:
      merge_groups<unsigned short>(num_final_groups, blur_initial_sigma, blur_multiple, policy, segments_us);
      break;
    default:
      fprintf(stderr, "Error: invalid segment type.\n");
      PROFILE_POP_RANGE(); // error return point
      return 0;
  }

#ifdef TIMER
  wkf_timer_stop(timer);
  time_sec = wkf_timer_time(timer);
  wkf_timer_destroy(timer);
  if (verbose) {
    char msgbuf[2048];
    sprintf(msgbuf, "Total segmentation time: %.3f seconds.", time_sec);
    msgInfo << msgbuf << sendmsg;
  }
#endif

  PROFILE_POP_RANGE(); // final return point

  return time_sec;
}



template <typename GROUP_T>
void Segmentation::merge_groups(unsigned long num_final_groups,
                                float blur_initial_sigma,
                                float blur_multiple,
                                MERGE_POLICY policy,
                                GROUP_T* segments) {
  PROFILE_PUSH_RANGE("Segmentation::merge_groups()", 0);

  switch (image_type) {
    case DT_FLOAT:
      merge_groups_helper<GROUP_T, float>(num_final_groups, blur_initial_sigma,
                                          blur_multiple, segments, policy, gaussian_f);
      break;
    case DT_USHORT:
      merge_groups_helper<GROUP_T, unsigned short>(num_final_groups, blur_initial_sigma,
                                                   blur_multiple, segments, policy, gaussian_us);
      break;
    case DT_UCHAR:
      merge_groups_helper<GROUP_T, unsigned char>(num_final_groups, blur_initial_sigma,
                                                  blur_multiple, segments, policy, gaussian_uc);
      break;
    default:
      fprintf(stderr, "Error: invalid image type.\n");
      PROFILE_POP_RANGE(); // error return point
      break;
  }

  PROFILE_POP_RANGE();
}

template <typename GROUP_T, typename IMAGE_T>
void Segmentation::merge_groups_helper(unsigned long num_final_groups,
                                       float blur_initial_sigma,
                                       float blur_multiple,
                                       GROUP_T* segments,
                                       MERGE_POLICY policy,
                                       GaussianBlur<IMAGE_T>* g) {
  PROFILE_PUSH_RANGE("Segmentation::merge_groups_helper()", 0);

  if (verbose)
    printf("Num groups = %ld\n", nGroups);

  ScaleSpaceFilter<GROUP_T, IMAGE_T> filter(width, height, depth, 
                                   nGroups, blur_initial_sigma,
                                   blur_multiple, use_cuda);

  // XXX right now we have one policy for the entire merge.
  // It might give a better/faster segmentation to start out with MERGE_HILL_CLIMB
  // and transition to a MERGE_WATERSHED* policy for integer
  // image types once we stop being able to merge groups.
  // This might be more work than it is worth though.
  while (nGroups > num_final_groups) {
    nGroups = filter.merge(segments, g, policy);
    if (verbose)
      printf("Num groups = %ld\n", nGroups);
  }

  PROFILE_POP_RANGE();
}


void Segmentation::switch_to_cpu() {
  switch(group_type) {
    case DT_ULONG:
      {
      unsigned long* new_seg_array = new unsigned long[nVoxels];
      copy_and_convert_type<unsigned long>(segments_ul, new_seg_array, nVoxels, true);
      free_array(segments_ul);
      segments_ul = new_seg_array;
      break;
      }
    case DT_UINT:
      {
      unsigned int* new_seg_array = new unsigned int[nVoxels];
      copy_and_convert_type<unsigned int>(segments_ui, new_seg_array, nVoxels, true);
      free_array(segments_ui);
      segments_ui = new_seg_array;
      break;
      }
    case DT_USHORT:
      {
      unsigned short* new_seg_array = new unsigned short[nVoxels];
      copy_and_convert_type<unsigned short>(segments_us, new_seg_array, nVoxels, true);
      free_array(segments_us);
      segments_us = new_seg_array;
      break;
      }

    // this should really never happen but xlC complains if we don't
    // include all enums in the switch block
    case DT_ERROR:
    default:
      printf("Segmentation::switch_to_cpu(): Invalid voxel type was set\n");
      break;
  }

  //TODO switch gaussian_fblur to cpu
  use_cuda = false;
}




template <typename GROUP_T>
void Segmentation::get_watershed_segmentation(float watershed_blur_sigma,
                                              GROUP_T* segments) {
  switch (image_type) {
    case DT_FLOAT:
      {
        gaussian_f->blur(watershed_blur_sigma);
        int imageongpu=1;
        float *imageptr = gaussian_f->get_image_d();
        if (!imageptr) {
          imageptr = gaussian_f->get_image();
          imageongpu=0;
        }
        Watershed<GROUP_T, float> watershed(height, width, depth, use_cuda);
        watershed.watershed(imageptr, imageongpu, segments, verbose);
        break;
      }
    case DT_USHORT:
      {
        gaussian_us->blur(watershed_blur_sigma);
        int imageongpu=1;
        unsigned short *imageptr = gaussian_us->get_image_d();
        if (!imageptr) {
          imageptr = gaussian_us->get_image();
          imageongpu=0;
        }
        Watershed<GROUP_T, unsigned short> watershed(height, width, depth, use_cuda);
        watershed.watershed(imageptr, imageongpu, segments, verbose);
        break;
      }
    case DT_UCHAR:
      {
        gaussian_uc->blur(watershed_blur_sigma);
        int imageongpu=1;
        unsigned char *imageptr = gaussian_uc->get_image_d();
        if (!imageptr) {
          imageptr = gaussian_uc->get_image();
          imageongpu=0;
        }
        Watershed<GROUP_T, unsigned char> watershed(height, width, depth, use_cuda);
        watershed.watershed(imageptr, imageongpu, segments, verbose);
        break;
      }
    default:
      fprintf(stderr, "Invalid image type.\n");
      break;
  }

  GROUP_T* group_map = allocate_array<GROUP_T>(nGroups);
  sequentialize_groups<GROUP_T>(segments, group_map);
  free_array(group_map);
}



unsigned long Segmentation::get_num_groups() {
  return nGroups;
}



template <typename OUT_T>
void Segmentation::get_results(OUT_T* results) {
  PROFILE_PUSH_RANGE("Segmentation::get_results()", 1);

  switch (group_type) {
    case DT_ULONG:
      copy_and_convert_type<unsigned long, OUT_T>(segments_ul, results, nVoxels, use_cuda);
      break;
    case DT_UINT:
      copy_and_convert_type<unsigned int, OUT_T>(segments_ui, results, nVoxels, use_cuda);
      break;
    case DT_USHORT:
      copy_and_convert_type<unsigned short, OUT_T>(segments_us, results, nVoxels, use_cuda);
      break;
    default:
      fprintf(stderr, "Error: invalid segment type.\n");
      break;
  }

  PROFILE_POP_RANGE();
}



template <typename IN_T, typename OUT_T>
void Segmentation::copy_and_convert_type(IN_T* in, OUT_T* out, unsigned long num_elems, bool copy_from_gpu) {
#if defined(VMDCUDA)
  if (copy_from_gpu) {
    IN_T* local_arr = new IN_T[num_elems];
    copy_array_from_gpu(local_arr, in, num_elems * sizeof(IN_T));

#ifdef WATERSHED_COMPATABILITY_TEST
    // dirty hack because gpu and cpu sequentialization don't give same ordering
    IN_T* group_map = new IN_T[nGroups];
    sequentialize_groups_cpu<IN_T>(local_arr, group_map);
    delete [] group_map;
#endif

    for (unsigned long i = 0; i < num_elems; i++) {
      out[i] = (OUT_T)local_arr[i];
    }
    delete [] local_arr;
  } else 
#endif
  {
#if defined(VMDCUDA)
    if (use_cuda) {
      copy_and_convert_type_cuda<IN_T, OUT_T>(in, out, num_elems);
    } else 
#endif
    {
      for (unsigned long i = 0; i < num_elems; i++) {
        out[i] = (OUT_T)in[i];
      }
    }
  }
}



Segmentation::Datatype Segmentation::get_new_group_type() {
  Datatype type;
  if (nGroups < USHRT_MAX) {
    type = DT_USHORT;
  } else if (nGroups < UINT_MAX) {
    type = DT_UINT;
  } else if (nGroups < ULONG_MAX) {
    type = DT_ULONG;
  } else {
    fprintf(stderr, "Error: images this large are not currently supported.\n");
    fprintf(stderr, "Warning: Undefined behavior if you continue to use this object.\n");
    type = DT_ERROR;
  }
  return type;
}

// these next two functions are only needed once after
// we get the intitial segmentation from watershed.
// They are duplicated in the ScaleSpaceFilter class
// so would be nice to find a way to remove them.
// We need them before we create a ScaleSpaceFilter obj.
// so we know know the max group # and can use decalare
// the obj with appropriate datatype


template <typename GROUP_T>
void Segmentation::sequentialize_groups(GROUP_T* segments, GROUP_T* group_map) {
#if defined(VMDCUDA)
  if (use_cuda) {
    nGroups = sequentialize_groups_cuda<GROUP_T>(segments, group_map, nVoxels, nGroups);
  } else 
#endif
  {
    nGroups = sequentialize_groups_cpu<GROUP_T>(segments, group_map);
  }
}



template <typename GROUP_T>
unsigned long Segmentation::sequentialize_groups_cpu(GROUP_T* segments, GROUP_T* group_map) {
  unsigned long numGroups = 0; //TODO changed from 0, update CUDA code
  GROUP_T max_val = GROUP_T(0)-1;
  // GROUP_T is only unsigned integer types so this ^ is defined behavior of C++ standard
  memset(group_map, max_val, nGroups * sizeof(GROUP_T));

  for (unsigned long i = 0; i < nVoxels; ++i) {
    const GROUP_T curr_group = segments[i];
    if (group_map[curr_group] == max_val) {
      group_map[curr_group] = numGroups++;
    }
    segments[i] = group_map[curr_group];
  }
  return numGroups;
}


// TODO check if we can use a smaller type,
// and if so transition segments to the new type

void Segmentation::update_type() {
  Datatype new_type = get_new_group_type();
  if (new_type != group_type) {
    switch (group_type) {
      case DT_ULONG:
        if (new_type == DT_UINT) {
          if (verbose) {
            printf("Converting unsigned ulong to uint.\n");
          }
          segments_ui = allocate_array<unsigned int>(nVoxels);
          copy_and_convert_type<unsigned long, unsigned int>(segments_ul, segments_ui, nVoxels);
        } else {
          if (verbose) {
            printf("Converting unsigned ulong to ushort.\n");
          }
          segments_us = allocate_array<unsigned short>(nVoxels);
          copy_and_convert_type<unsigned long, unsigned short>(segments_ul, segments_us, nVoxels);
        }
        free_array(segments_ul);
        break;
      case DT_UINT:
        if (verbose) {
          printf("Converting uint to ushort.\n");
        }
        segments_us = allocate_array<unsigned short>(nVoxels);
        copy_and_convert_type<unsigned int, unsigned short>(segments_ui, segments_us, nVoxels);
        free_array(segments_ui);
        break;
      default:
        printf("Error in type conversion.\n");
        break;
    }
  }
  group_type = new_type;
}



template <typename GROUP_T>
GROUP_T* Segmentation::allocate_array(unsigned long num_elements) {
  GROUP_T* arr;
#if defined(VMDCUDA)
  if (use_cuda) {
    arr = (GROUP_T*) alloc_cuda_array(num_elements * sizeof(GROUP_T));
  } else 
#endif
  {
    arr = new GROUP_T[num_elements];
  }
  return arr;
}



template <typename GROUP_T>
void Segmentation::free_array(GROUP_T*& arr) {
#if defined(VMDCUDA)
  if (use_cuda) {
    free_cuda_array(arr);
  } else 
#endif
  {
    delete [] arr;
  }
  arr = NULL;
}

