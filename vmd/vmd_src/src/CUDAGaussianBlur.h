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
 *      $RCSfile: CUDAGaussianBlur.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.5 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Perform Gaussian blur filtering on 3-D volumes. Primarily for use in
 *   scale-space filtering for 3-D image segmentation of Cryo-EM density maps.
 ***************************************************************************/

#ifndef CUDA_GAUSSIAN_BLUR
#define CUDA_GAUSSIAN_BLUR

void gaussian3D_cuda(float* kernel, int kernel_size);

template <typename IMAGE_T>
bool setup_cuda_filter(IMAGE_T* image, int w, int h, int d);

void copy_array_from_gpu(void* arr, void* arr_d, int bytes);

void copy_array_to_gpu(void* arr_d, void* arr, int bytes);

template <typename IMAGE_T>
void gaussian1D_x_cuda(IMAGE_T* src_d, IMAGE_T* dst_d, int kernel_size,
                       int width, int height, int depth);

template <typename IMAGE_T>
void gaussian1D_y_cuda(IMAGE_T* src_d, IMAGE_T* dst_d, int kernel_size,
                       int width, int height, int depth);

template <typename IMAGE_T>
void gaussian1D_z_cuda(IMAGE_T* src_d, IMAGE_T* dst_d, int kernel_size,
                       int width, int height, int depth);

void set_gaussian_1D_kernel_cuda(float* kernel, int kernel_size);

void* alloc_cuda_array(int bytes);

void free_cuda_array(void* arr);

#endif
