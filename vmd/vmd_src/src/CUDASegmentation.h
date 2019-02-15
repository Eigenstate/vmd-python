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
 *      $RCSfile: CUDASegmentation.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   GPU-accelerated scale-space variant of Watershed image segmentation 
 *   intended for use on 3-D cryo-EM density maps.
 ***************************************************************************/

#ifndef CUDA_SEGMENTATION
#define CUDA_SEGMENTATION

typedef struct {
  void *tmp_d; // GPU memory buffer for temporary storage
  unsigned long sz;   // size of the GPU memory buffer
} gpuseg_temp_storage;  

void free_gpu_temp_storage(gpuseg_temp_storage *tmp);

template <typename GROUP_T>
long sequentialize_groups_cuda(GROUP_T* groups_d, GROUP_T* group_map_d, 
                               unsigned long nVoxels, unsigned long nGroups, 
                               gpuseg_temp_storage *tmp = NULL,
                               gpuseg_temp_storage *scanwork = NULL);

template <typename GROUP_T, typename IMAGE_T>
void find_groups_max_idx_cuda(GROUP_T* groups_d, IMAGE_T* image_d, unsigned long* max_idx, unsigned long nVoxels, unsigned long nGroups, gpuseg_temp_storage *tmp = NULL);

template <typename GROUP_T, typename IMAGE_T>
void hill_climb_merge_cuda(GROUP_T* groups_d, IMAGE_T* image_d, unsigned long* max_idx_d, GROUP_T* group_map_d,
                             int height, int width, int depth, unsigned long nGroups);

template <typename IN_T, typename OUT_T>
void copy_and_convert_type_cuda(IN_T* in, OUT_T* out, long num_elements);

template <typename GROUP_T, typename IMAGE_T>
void watershed_hill_climb_merge_cuda(GROUP_T* segments_d, GROUP_T* new_segments_d, IMAGE_T* image_d,
                                     GROUP_T* group_map_d, unsigned long* max_idx_d,
                                     long height, long width, long depth, unsigned long nGroups);

#endif
