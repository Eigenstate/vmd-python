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
 *      $RCSfile: GaussianBlur.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.25 $        $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Perform Gaussian blur filtering on 3-D volumes. Primarily for use in 
 *   scale-space filtering for 3-D image segmentation of Cryo-EM density maps.
 ***************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <climits>
#include "GaussianBlur.h"
#include "CUDAGaussianBlur.h"
#include "ProfileHooks.h"

#define PI_F (3.14159265359f)

template <typename IMAGE_T>
GaussianBlur<IMAGE_T>::GaussianBlur(IMAGE_T* img, int w, int h, int d, bool use_cuda) {
  host_image_needs_update = 0;
  height = h;
  width = w;
  depth = d;
  image_d = NULL;
  image = NULL;
  scratch = NULL;
  heightWidth = long(height) * long(width);
  nVoxels = heightWidth * long(depth);
  cuda = use_cuda;
#ifndef VMDCUDA
  cuda = false;
#endif

  image = new IMAGE_T[nVoxels];
#ifdef VMDCUDA
  if (cuda) {
    image_d = (IMAGE_T*) alloc_cuda_array(nVoxels * sizeof(IMAGE_T));
    scratch_d = (IMAGE_T*) alloc_cuda_array(nVoxels * sizeof(IMAGE_T));
    if (image_d == NULL || scratch_d == NULL) {
      cuda = false;
      free_cuda_array(image_d);
      free_cuda_array(scratch_d);
    } else {
      copy_array_to_gpu(image_d, img, nVoxels * sizeof(IMAGE_T));
      copy_array_to_gpu(scratch_d, img, nVoxels * sizeof(IMAGE_T));
    }
  } 
  if (!cuda)
#endif
  {
    scratch = new IMAGE_T[nVoxels];
    memcpy(image, img, nVoxels * sizeof(IMAGE_T));
  }
}


template <typename IMAGE_T>
GaussianBlur<IMAGE_T>::~GaussianBlur() {
  delete [] image;
#ifdef VMDCUDA
  if (cuda) {
    free_cuda_array(image_d);
    free_cuda_array(scratch_d);
  } else
#endif
    delete [] scratch;
}


template <typename IMAGE_T>
IMAGE_T* GaussianBlur<IMAGE_T>::get_image() {
#ifdef VMDCUDA
  if (host_image_needs_update) {
    copy_array_from_gpu(image, image_d, nVoxels * sizeof(IMAGE_T));
    host_image_needs_update = 0;
  }
#endif

  return image;
}


template <typename IMAGE_T>
IMAGE_T* GaussianBlur<IMAGE_T>::get_image_d() {
  return image_d;
}

// This is an arbitarily  "max" size for the CPU, but
// is the max for the GPU. Additionally we probably never want
// to run a blur with a kernel_size this larger anyways as it is likely
// larger than the image. I think it makes sense to cap the sigma to a certian
// large value. That would have to be done before this function is called though.
#define MAX_KERNEL_SIZE 4096

template <typename IMAGE_T>
int GaussianBlur<IMAGE_T>::getKernelSizeForSigma(float sigma) {
  int size = int(ceilf(sigma * 6));
  if (size % 2 == 0) {
    ++size;
  }
  if (size < 3) {
    size = 3;
  }
  if (size >=  MAX_KERNEL_SIZE) {
    size = MAX_KERNEL_SIZE;
  }
  return size;
}


template <typename IMAGE_T>
void GaussianBlur<IMAGE_T>::fillGaussianBlurKernel3D(float sigma, int size, float* kernel) {
  float sigma2 = sigma * sigma;
  int middle = size / 2;
  for (int z = 0; z < size; z++) {
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        const int idx = z * size * size + y * size + x;
        float x_dist = middle - x;
        float y_dist = middle - y;
        float z_dist = middle - z;
        float distance2 = x_dist * x_dist + y_dist * y_dist + z_dist * z_dist;
        float s = 1.0f / (sigma * sqrt(2.0f * PI_F)) * expf(-distance2 / (2.0f * sigma2));
        kernel[idx] = s;
      }
    }
  }
}


template <typename IMAGE_T>
void GaussianBlur<IMAGE_T>::fillGaussianBlurKernel1D(float sigma, int size, float* kernel) {
  float sigma2 = sigma * sigma;
  int middle = size / 2;
  for (int i = 0; i < size; ++i) {
    float distance = middle - i;
    float distance2 = distance * distance;
    float s = 1.0f / (sigma * sqrtf(2.0f*PI_F)) * expf(-distance2 / (2.0f * sigma2));
    kernel[i] = s;
  }
}


#define SWAPT(a, b) {\
  IMAGE_T* t = a;\
  a = b;\
  b = t;\
}


template <typename IMAGE_T>
void GaussianBlur<IMAGE_T>::blur(float sigma) {
#ifdef VMDCUDA
  if (cuda) {
    blur_cuda(sigma);
  } else
#endif
    blur_cpu(sigma);
}


#if defined(VMDCUDA)
template <typename IMAGE_T>
void GaussianBlur<IMAGE_T>::blur_cuda(float sigma) {
  PROFILE_PUSH_RANGE("GaussianBlur<IMAGE_T>::blur_cuda()", 7);

  if (true) {  //TODO temp hack to test 3D gaussian kernel
    int size = getKernelSizeForSigma(sigma);
    float kernel[size];
    fillGaussianBlurKernel1D(sigma, size, kernel);
    set_gaussian_1D_kernel_cuda(kernel, size);

    gaussian1D_x_cuda(image_d, scratch_d, size, width, height, depth);
    SWAPT(image_d, scratch_d);

    gaussian1D_y_cuda(image_d, scratch_d, size, width, height, depth);
    SWAPT(image_d, scratch_d);

    gaussian1D_z_cuda(image_d, scratch_d, size, width, height, depth);
    SWAPT(image_d, scratch_d);

    // don't copy back to the host until we are absolutely forced to...
    host_image_needs_update = 1;
  } else {
    int size = getKernelSizeForSigma(sigma);
    float kernel[size * size * size];
    fillGaussianBlurKernel3D(sigma, size, kernel);
  }

  PROFILE_POP_RANGE();
}
#endif

#if 1
template <typename T>
static inline void assign_float_to_dest(float val, T* dest) {
  // truncation is not great for quantized integer representations,
  // but it is at least portable...
  dest[0] = val;
}
#else
template <typename T>
static inline void assign_float_to_dest(float val, T* dest) {
  dest[0] = roundf(val);
}

// this case causes problem with older compilers
template <> inline void assign_float_to_dest<float>(float val, float* dest) {
  dest[0] = val;
}
#endif

template <typename IMAGE_T>
void GaussianBlur<IMAGE_T>::blur_cpu(float sigma) {
  int x, y, z;
  int size = getKernelSizeForSigma(sigma);
  float *kernel = new float[size];
  fillGaussianBlurKernel1D(sigma, size, kernel);
  const int offset = size >> 1;
  SWAPT(scratch, image);

  /* X kernel */
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
  for (z = 0; z < depth; ++z) {
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        const long idx = z*heightWidth + y*long(width) + x;
        const int new_offset_neg = x - offset >= 0 ? -offset :  -x;
        const int new_offset_pos = x + offset < width ? offset : width - x - 1;
        float value = 0.0f;
        int i;
        for (i = -offset; i < new_offset_neg; ++i) {
          value += scratch[idx + new_offset_neg] * kernel[i+offset];
        }

        for (; i <= new_offset_pos; ++i) {
          value += scratch[idx + i] * kernel[i+offset];
        }

        for (; i <= offset; ++i) {
          value += scratch[idx + new_offset_pos] * kernel[i+offset];
        }

        assign_float_to_dest(value, &image[idx]);
      }
    }
  }
  SWAPT(scratch, image);

  /* Y kernel */
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
  for (z = 0; z < depth; ++z) {
    for (y = 0; y < height; ++y) {
      const int new_offset_neg = y - offset >= 0 ? -offset :  -y;
      const int new_offset_pos = y + offset < height ? offset : height - y - 1;
      for (x = 0; x < width; ++x) {
        const long idx = z*heightWidth + y*long(width) + x;
        float value = 0.0f;
        int i;

        for (i = -offset; i < new_offset_neg; ++i) {
          value += scratch[idx + new_offset_neg*width] * kernel[i+offset];
        }

        for (; i <= new_offset_pos; ++i) {
          value += scratch[idx + i*width] * kernel[i+offset];
        }

        for (; i <= offset; ++i) {
          value += scratch[idx + new_offset_pos*width] * kernel[i+offset];
        }

        assign_float_to_dest(value, &image[idx]);
      }
    }
  }
  SWAPT(scratch, image);

  /* Z kernel */
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
  for (z = 0; z < depth; ++z) {
    const int new_offset_neg = z - offset >= 0 ? -offset :  -z;
    const int new_offset_pos = z + offset < depth ? offset : depth - z - 1;
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        const long idx = z*heightWidth + y*long(width) + x;
        float value = 0.0f;
        int i;

        for (i = -offset; i < new_offset_neg; ++i) {
          value += scratch[idx + new_offset_neg*heightWidth] * kernel[i+offset];
        }

        for (; i <= new_offset_pos; ++i) {
            value += scratch[idx + i*heightWidth] * kernel[i+offset];
        }

        for (; i <= offset; ++i) {
          value += scratch[idx + new_offset_pos*heightWidth] * kernel[i+offset];
        }

        assign_float_to_dest(value, &image[idx]);
      }
    }
  }

  delete [] kernel;
}

template class GaussianBlur<float>;
template class GaussianBlur<unsigned short>;
template class GaussianBlur<unsigned char>;
