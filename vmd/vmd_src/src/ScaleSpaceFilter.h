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
 *      $RCSfile: ScaleSpaceFilter.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.9 $        $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Perform scale-space filtering on 3-D volumes, e.g., cryo-EM density maps,
 *   for use in subsequent image segmentation algorithms.
 ***************************************************************************/

#ifndef SCALE_SPACE_FILTER_H
#define SCALE_SPACE_FILTER_H

#include "GaussianBlur.h"
#include "CUDASegmentation.h"

enum MERGE_POLICY {
  MERGE_HILL_CLIMB,
  MERGE_WATERSHED_HILL_CLIMB,
  MERGE_WATERSHED_OVERLAP
};

template <typename GROUP_T, typename IMAGE_T>
class ScaleSpaceFilter {
  public:

    ScaleSpaceFilter(int w,
                     int h, 
                     int d,
                     long nGroups,
                     float initial_blur_sigma,
                     float blur_multiple,
                     bool use_cuda);

    ~ScaleSpaceFilter();

    long merge(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian, MERGE_POLICY policy);

    long merge_with_watershed(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian);

  private:
    int   width;
    int   height;
    int   depth;
    long  heightWidth;
    long  nVoxels;
    long  nGroups;
    float blur_multiple;
    float current_blur;
    bool  use_cuda;
    gpuseg_temp_storage gpu_seq_tmp;
    gpuseg_temp_storage gpu_scanwork_tmp;
    gpuseg_temp_storage gpu_grpmaxidx_tmp;

    unsigned long* max_idx;
    GROUP_T* group_map;

    long find_local_maxima(long curr_idx, IMAGE_T* image);

    long sequentialize_group_nums(GROUP_T* segments, long max_group_num);

    long sequentialize_group_nums_cpu(GROUP_T* segments, long max_group_num);

    void find_groups_max_idx(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian);

    void find_groups_max_idx_cpu(GROUP_T* segments, IMAGE_T* image);

    void watershed_overlap_merge_cpu(GROUP_T* segments, GROUP_T* new_segments);

    void watershed_hill_climb_merge_cpu(GROUP_T* segments, GROUP_T* new_segments, IMAGE_T* image);

    void hill_climb_merge_cpu(GROUP_T* segments, IMAGE_T* image);

    void watershed_overlap_merge(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian);

    void watershed_hill_climb_merge(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian);

    void hill_climb_merge(GROUP_T* segments, GaussianBlur<IMAGE_T>* gaussian);

    template <typename ARRAY_T>
    ARRAY_T* allocate_array(long num_elements);

    template <typename ARRAY_T>
    void free_array(ARRAY_T*& arr);

};

#endif
