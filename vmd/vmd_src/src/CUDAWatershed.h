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
 *      $RCSfile: CUDAWatershed.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.9 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA-accelerated Watershed image segmentation
 ***************************************************************************/

#ifndef CUDA_WATERSHED_H
#define CUDA_WATERSHED_H

#include "Watershed.h"

template <typename GROUP_T, typename IMAGE_T>
bool init_gpu(state_t<GROUP_T, IMAGE_T>& state, int* eq_and_lower, watershed_gpu_state_t<GROUP_T, IMAGE_T>& gpu_state,
              unsigned int w, unsigned int h, unsigned int d);

template <class GROUP_T, typename IMAGE_T>
bool init_gpu_on_device(watershed_gpu_state_t<GROUP_T, IMAGE_T> &gpu_state, 
                        IMAGE_T* image, int imageongpu, 
                        unsigned int w, unsigned int h, unsigned int d);
 
template <typename GROUP_T, typename IMAGE_T>
void destroy_gpu(watershed_gpu_state_t<GROUP_T, IMAGE_T>& gpu_state);

template <typename GROUP_T, typename IMAGE_T>
void update_cuda(watershed_gpu_state_t<GROUP_T, IMAGE_T>& gpu_state, GROUP_T* segments_d);

void guassian3D_gpu();

#endif
