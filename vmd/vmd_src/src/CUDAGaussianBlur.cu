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
 *      $RCSfile: CUDAGaussianBlur.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Perform Gaussian blur filtering on 3-D volumes. Primarily for use in
 *   scale-space filtering for 3-D image segmentation of Cryo-EM density maps.
 ***************************************************************************/

#define WATERSHED_INTERNAL 1

#include <stdio.h>
#include "CUDAGaussianBlur.h"
#include "Watershed.h"
#include "ProfileHooks.h"
#include <cuda_fp16.h>

#if 0 
#define CUERR { cudaError_t err; \
  cudaDeviceSynchronize(); \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  }}
#else
#define CUERR
#endif

#define KERNEL_MAX_SIZE 4096

__constant__ float kernel_d_c[KERNEL_MAX_SIZE];

#define BLOCKDIM_X 128
#define BLOCKDIM_Y 1
#define BLOCKDIM_Z 1

// optimized for 21 kernel size
#define XCONV_BLOCKDIM_X 96
#define XCONV_BLOCKDIM_Y 1
#define XCONV_BLOCKDIM_Z 1

#define YCONV_BLOCKDIM_X 16
#define YCONV_BLOCKDIM_Y 8
#define YCONV_BLOCKDIM_Z 1

#define ZCONV_BLOCKDIM_X 4
#define ZCONV_BLOCKDIM_Y 1
#define ZCONV_BLOCKDIM_Z 32

#define X_PADDING 32

#define SWAPF(a, b) {\
  float* t = a;\
  a = b;\
  b = t;\
}

#define INST_GAUSSIAN_CUDA(I_T) \
template void gaussian1D_x_cuda<I_T>(I_T* src_d, I_T* dst_d, int kernel_size,\
                                       int width, int height, int depth);\
template void gaussian1D_y_cuda<I_T>(I_T* src_d, I_T* dst_d, int kernel_size,\
                                       int width, int height, int depth);\
template void gaussian1D_z_cuda<I_T>(I_T* src_d, I_T* dst_d, int kernel_size,\
                                       int width, int height, int depth);

INST_GAUSSIAN_CUDA(float)
//INST_GAUSSIAN_CUDA(half)
INST_GAUSSIAN_CUDA(unsigned short)
INST_GAUSSIAN_CUDA(unsigned char)

// Function that converts val to the appropriate type and then assigns it to arr[idx]
// This is necessary for coverting an real type to integer type so we can handle rounding
template <typename T>
__device__ __forceinline__ void convert_type_and_assign_val(T* arr, long idx, float val) {
  arr[idx] = roundf(val);
}

template <> __device__ __forceinline__ void convert_type_and_assign_val<float>(float* arr, long idx, float val) {
  arr[idx] = val;
}

#if CUDART_VERSION >= 9000
// half-precision only became mainstream with CUDA 9.x and later versions
template <> __device__ __forceinline__ void convert_type_and_assign_val<half>(half* arr, long idx, float val) {
  arr[idx] = val;
}
#endif

void copy_array_from_gpu(void* arr, void* arr_d, int bytes) {
  PROFILE_PUSH_RANGE("CUDAGaussianBlur::copy_array_from_gpu()", 5);
 
  cudaMemcpy(arr, arr_d, bytes, cudaMemcpyDeviceToHost);
  CUERR // check and clear any existing errors

  PROFILE_POP_RANGE();
}


void copy_array_to_gpu(void* arr_d, void* arr, int bytes) {
  cudaMemcpyAsync(arr_d, arr, bytes, cudaMemcpyHostToDevice);
  CUERR // check and clear any existing errors
}


void* alloc_cuda_array(int bytes) {
  void* t;
  cudaMalloc(&t, bytes);
  CUERR // check and clear any existing errors
  return t;
}


void free_cuda_array(void* arr) {
  cudaFree(arr);
}


void set_gaussian_1D_kernel_cuda(float* kernel, int kernel_size) {

  if (kernel_size > KERNEL_MAX_SIZE) {
    printf("Warning: exceeded maximum kernel size on GPU.\n");
    kernel_size = KERNEL_MAX_SIZE;
  }

  cudaMemcpyToSymbolAsync(kernel_d_c, kernel, kernel_size * sizeof(float));
  CUERR // check and clear any existing errors
}


template <typename IMAGE_T, const int offset>
__global__ void convolution_x_kernel_s(const IMAGE_T* __restrict__ src_d,
    IMAGE_T* __restrict__ dst_d, int width, int height, int depth) {
  __shared__ IMAGE_T src_s[XCONV_BLOCKDIM_Z * XCONV_BLOCKDIM_Y * (XCONV_BLOCKDIM_X + offset * 2)];
  const int x = blockIdx.x * XCONV_BLOCKDIM_X + threadIdx.x;
  const int y = blockIdx.y * XCONV_BLOCKDIM_Y + threadIdx.y;
  const int z = blockIdx.z * XCONV_BLOCKDIM_Z + threadIdx.z;

  if (y >= height || z >= depth) {
    return;
  }

  // global idx
  const long idx = z * long(height) * long(width) + y * long(width) + x;

  // idx of first entry in row (x = 0)
  const long g_row_base_idx = z * long(height) * long(width) + y * long(width);

  // idx of first entry in shared mem array row
  const int t_row_base_idx = threadIdx.z * XCONV_BLOCKDIM_Y * (XCONV_BLOCKDIM_X + offset * 2)
                 + threadIdx.y * (XCONV_BLOCKDIM_X + offset*2) + offset;

  // idx of first x coord in this block
  const int base_x = x - threadIdx.x;

  // Check if we need to deal with edge cases
  if (base_x - offset < 0 || base_x + XCONV_BLOCKDIM_X + offset >= width) {
    for (int i = threadIdx.x - offset; i < XCONV_BLOCKDIM_X + offset; i += XCONV_BLOCKDIM_X) {
        int x_offset = base_x + i;
        if (x_offset < 0) // left edge case
            x_offset = 0;
        else if (x_offset >= width) // right edge case
            x_offset = width-1;
        src_s[t_row_base_idx + i] = src_d[g_row_base_idx + x_offset];
    }
  } else {
    for (int i = threadIdx.x - offset; i < XCONV_BLOCKDIM_X + offset; i += XCONV_BLOCKDIM_X) {
        int x_offset = base_x + i;
        src_s[t_row_base_idx + i] = src_d[g_row_base_idx + x_offset];
    }
  }
  if (x >= width)
    return;

  // XXX is this safe to do with early returns on all GPUs?
  __syncthreads();

  const IMAGE_T* src_s_offset = src_s + t_row_base_idx + threadIdx.x;
  const float* kernel_offset = kernel_d_c + offset;
  float value = 0.0f;
#pragma unroll
  for (int i = -offset; i <= offset; ++i) {
    value += src_s_offset[i] * kernel_offset[i];
  }

  convert_type_and_assign_val(dst_d, idx, value);
}



template <typename IMAGE_T, const int offset>
__global__ void convolution_y_kernel_s(const IMAGE_T* __restrict__ src_d,
    IMAGE_T* __restrict__ dst_d, int width, int height, int depth) {
  __shared__ IMAGE_T src_s[YCONV_BLOCKDIM_Z * (YCONV_BLOCKDIM_Y + 2*offset) * YCONV_BLOCKDIM_X];
  const int x = blockIdx.x * YCONV_BLOCKDIM_X + threadIdx.x;
  const int y = blockIdx.y * YCONV_BLOCKDIM_Y + threadIdx.y;
  const int z = blockIdx.z * YCONV_BLOCKDIM_Z + threadIdx.z;

  if (x >= width || z >= depth) {
    return;
  }

  // global idx
  const long idx = z * long(height) * long(width) + y * long(width) + x;

  // idx of first entry in column (y = 0)
  const long g_col_base_idx = z * long(height) * long(width) + x;

  // idx of first entry in shared mem array column
  const int t_col_base_idx = threadIdx.z * (YCONV_BLOCKDIM_Y +offset*2) * YCONV_BLOCKDIM_X
                 + threadIdx.x + offset * YCONV_BLOCKDIM_X;

  // idx of first y coord in this block
  const int base_y = y - threadIdx.y;

  // Check if we need to deal with edge cases
  if (base_y - offset < 0 || base_y + YCONV_BLOCKDIM_Y - 1 + offset >= height) {
    for (int i = threadIdx.y - offset; i < YCONV_BLOCKDIM_Y + offset; i += YCONV_BLOCKDIM_Y) {
        int y_offset = base_y + i;
        if (y_offset < 0) // left edge case
            y_offset = 0;
        else if (y_offset >= height) // right edge case
            y_offset = height-1;
        src_s[t_col_base_idx + i*YCONV_BLOCKDIM_X] = src_d[g_col_base_idx + y_offset*width];
    }
  } else {
    for (int i = threadIdx.y - offset; i < YCONV_BLOCKDIM_Y + offset; i += YCONV_BLOCKDIM_Y) {
        int y_offset = base_y + i;
        src_s[t_col_base_idx + i*YCONV_BLOCKDIM_X] = src_d[g_col_base_idx + y_offset*width];
    }
  }
  if (y >= height)
    return;

  // XXX is this safe to do with early returns on all GPUs?
  __syncthreads();

  const IMAGE_T* src_s_offset = src_s + t_col_base_idx + threadIdx.y * YCONV_BLOCKDIM_X;
  const float* kernel_offset = kernel_d_c + offset;
  float value = 0.0f;
#pragma unroll
  for (int i = -offset; i <= offset; ++i) {
    value += src_s_offset[i*YCONV_BLOCKDIM_X] * kernel_offset[i];
  }

  convert_type_and_assign_val(dst_d, idx, value);
}


template <typename IMAGE_T, const int offset>
__global__ void convolution_z_kernel_s(const IMAGE_T* __restrict__ src_d,
    IMAGE_T* __restrict__ dst_d, int width, int height, int depth) {
  __shared__ IMAGE_T src_s[ZCONV_BLOCKDIM_Z * (ZCONV_BLOCKDIM_Y + 2*offset) * ZCONV_BLOCKDIM_X];
  const int x = blockIdx.x * ZCONV_BLOCKDIM_X + threadIdx.x;
  const int y = blockIdx.y * ZCONV_BLOCKDIM_Y + threadIdx.y;
  const int z = blockIdx.z * ZCONV_BLOCKDIM_Z + threadIdx.z;

  if (x >= width || y >= height) {
    return;
  }

  // global idx
  const long idx = z * long(height) * long(width) + y * long(width) + x;

  // idx of first entry in column (z = 0)
  const long g_base_col_idx = y * long(width) + x;

  // idx of first entry in shared mem array column
  const int t_base_col_idx = threadIdx.y * ZCONV_BLOCKDIM_X 
                             + threadIdx.x + offset * ZCONV_BLOCKDIM_X;
  const int base_z = z - threadIdx.z;
  const long heightWidth = long(height) * long(width);
  float value = 0.0f;

  // Check if we need to deal with edge cases
  if (base_z - offset < 0 || base_z + ZCONV_BLOCKDIM_Z - 1 + offset >= depth) {
    for (int i = threadIdx.z - offset; i < ZCONV_BLOCKDIM_Z + offset; i += ZCONV_BLOCKDIM_Z) {
        int z_offset = base_z + i;
        if (z_offset < 0) // left edge case
            z_offset = 0;
        else if (z_offset >= depth) // right edge case
            z_offset = depth-1;
        src_s[t_base_col_idx + i*ZCONV_BLOCKDIM_X*ZCONV_BLOCKDIM_Y]
              = src_d[g_base_col_idx + z_offset*heightWidth];
    }
  } else {
    for (int i = threadIdx.z - offset; i < ZCONV_BLOCKDIM_Z + offset; i += ZCONV_BLOCKDIM_Z) {
        int z_offset = base_z + i;
        src_s[t_base_col_idx + i*ZCONV_BLOCKDIM_X*ZCONV_BLOCKDIM_Y]
              = src_d[g_base_col_idx + z_offset*heightWidth];
    }
  }

  if (z >= depth)
    return;

  __syncthreads();

  const IMAGE_T* src_s_offset = src_s + t_base_col_idx + threadIdx.z
                              * ZCONV_BLOCKDIM_X*ZCONV_BLOCKDIM_Y;
  const float* kernel_offset = kernel_d_c + offset;
#pragma unroll
  for (int i = -offset; i <= offset; ++i) {
    value += src_s_offset[i*ZCONV_BLOCKDIM_X*ZCONV_BLOCKDIM_Y] * kernel_offset[i];
  }

  convert_type_and_assign_val(dst_d, idx, value);
}


template <typename IMAGE_T, const int offset>
__global__ void convolution_x_kernel(const IMAGE_T* __restrict__ src_d,
    IMAGE_T* __restrict__ dst_d, int width, int height, int depth) {
  const int x = blockIdx.x * XCONV_BLOCKDIM_X + threadIdx.x;
  const int y = blockIdx.y * XCONV_BLOCKDIM_Y + threadIdx.y;
  const int z = blockIdx.z * XCONV_BLOCKDIM_Z + threadIdx.z;

  if (x >= width || y >= height || z >= depth) {
    return;
  }

  const long idx = z * long(height) * long(width) + y * long(width) + x;
  const int offset_neg = x - offset >= 0 ? -offset : -x;
  const int offset_pos = x + offset < width ? offset : width - x - 1;
  const float* kernel_offset = kernel_d_c + offset;
  const IMAGE_T* src_idx = src_d + idx;
  float value = 0.0f;

  if (offset_neg == -offset && offset_pos == offset) {
#pragma unroll
    for (int i = -offset; i <= offset; ++i) {
      value += src_idx[i] * kernel_offset[i];
    }
  } else {
    // Handle boundary condition
    for (int i = -offset; i < offset_neg; ++i) {
      value += src_idx[offset_neg] * kernel_offset[i];
    }

    for (int i = offset_neg; i < offset_pos; ++i) {
      value += src_idx[i] * kernel_offset[i];
    }

    // Handle boundary condition
    for (int i = offset_pos; i <= offset; ++i) {
      value += src_idx[offset_pos] * kernel_offset[i];
    }
  }

  convert_type_and_assign_val(dst_d, idx, value);
}


template <typename IMAGE_T>
__global__ void convolution_x_kernel(const IMAGE_T* __restrict__ src_d,
    IMAGE_T* __restrict__ dst_d, int offset, int width, int height, int depth) {
  const int x = blockIdx.x * XCONV_BLOCKDIM_X + threadIdx.x;
  const int y = blockIdx.y * XCONV_BLOCKDIM_Y + threadIdx.y;
  const int z = blockIdx.z * XCONV_BLOCKDIM_Z + threadIdx.z;

  if (x >= width || y >= height || z >= depth) {
    return;
  }

  const long idx = z * long(height) * long(width) + y * long(width) + x;
  const int offset_neg = x - offset >= 0 ? -offset : -x;
  const int offset_pos = x + offset < width ? offset : width - x - 1;
  const float* kernel_offset = kernel_d_c + offset;
  const IMAGE_T* src_idx = src_d + idx;
  float value = 0.0f;

  // Handle boundary condition
  for (int i = -offset; i < offset_neg; ++i) {
    value += src_idx[offset_neg] * kernel_offset[i];
  }

  for (int i = offset_neg; i < offset_pos; ++i) {
    value += src_idx[i] * kernel_offset[i];
  }

  // Handle boundary condition
  for (int i = offset_pos; i <= offset; ++i) {
    value += src_idx[offset_pos] * kernel_offset[i];
  }

  convert_type_and_assign_val(dst_d, idx, value);
}


template <typename IMAGE_T, const int offset>
__global__ void convolution_y_kernel(const IMAGE_T* __restrict__ src_d,
    IMAGE_T* __restrict__ dst_d, int width, int height, int depth) {
  const int x = blockIdx.x * YCONV_BLOCKDIM_X + threadIdx.x;
  const int y = blockIdx.y * YCONV_BLOCKDIM_Y + threadIdx.y;
  const int z = blockIdx.z * YCONV_BLOCKDIM_Z + threadIdx.z;

  if (x >= width || y >= height || z >= depth) {
    return;
  }

  const long idx = z * long(height) * long(width) + y * long(width) + x;
  const int offset_neg = y - offset >= 0 ? -offset : -y;
  const int offset_pos = y + offset < height ? offset : height - y - 1;
  const float* kernel_offset = kernel_d_c + offset;
  const IMAGE_T* src_idx = src_d + idx;
  float value = 0.0f;

  if (offset_neg == -offset && offset_pos == offset) {
#pragma unroll
    for (int i = -offset; i <= offset; ++i) {
      value += src_idx[i*width] * kernel_offset[i];
    }
  } else {
    // Handle boundary condition
    for (int i = -offset; i < offset_neg; ++i) {
      value += src_idx[offset_neg*width] * kernel_offset[i];
    }

    for (int i = offset_neg; i < offset_pos; ++i) {
      value += src_idx[i*width] * kernel_offset[i];
    }

    // Handle boundary condition
    for (int i = offset_pos; i <= offset; ++i) {
      value += src_idx[offset_pos*width] * kernel_offset[i];
    }
  }

  convert_type_and_assign_val(dst_d, idx, value);
}


template <typename IMAGE_T>
__global__ void convolution_y_kernel(const IMAGE_T* __restrict__ src_d,
    IMAGE_T* __restrict__ dst_d, int offset, int width, int height, int depth) {
  const int x = blockIdx.x * YCONV_BLOCKDIM_X + threadIdx.x;
  const int y = blockIdx.y * YCONV_BLOCKDIM_Y + threadIdx.y;
  const int z = blockIdx.z * YCONV_BLOCKDIM_Z + threadIdx.z;

  if (x >= width || y >= height || z >= depth) {
    return;
  }

  const long idx = z * long(height) * long(width) + y * long(width) + x;
  const int offset_neg = y - offset >= 0 ? -offset : -y;
  const int offset_pos = y + offset < height ? offset : height - y - 1;
  const float* kernel_offset = kernel_d_c + offset;
  const IMAGE_T* src_idx = src_d + idx;
  float value = 0.0f;

  // Handle boundary condition
  for (int i = -offset; i < offset_neg; ++i) {
    value += src_idx[offset_neg*width] * kernel_offset[i];
  }

  for (int i = offset_neg; i < offset_pos; ++i) {
    value += src_idx[i*width] * kernel_offset[i];
  }

  // Handle boundary condition
  for (int i = offset_pos; i <= offset; ++i) {
    value += src_idx[offset_pos*width] * kernel_offset[i];
  }

  convert_type_and_assign_val(dst_d, idx, value);
}


template <typename IMAGE_T>
__global__ void convolution_z_kernel(const IMAGE_T* __restrict__ src_d,
    IMAGE_T* __restrict__ dst_d, int offset, int width, int height, int depth) {
  const int x = blockIdx.x * ZCONV_BLOCKDIM_X + threadIdx.x;
  const int y = blockIdx.y * ZCONV_BLOCKDIM_Y + threadIdx.y;
  const int z = blockIdx.z * ZCONV_BLOCKDIM_Z + threadIdx.z;

  if (x >= width || y >= height || z >= depth) {
    return;
  }

  const int idx = z * long(height) * long(width) + y * long(width) + x;
  const int offset_neg = z - offset >= 0 ? -offset : -z;
  const int offset_pos = z + offset < depth ? offset : depth - z - 1;
  const long heightWidth = long(height) * long(width);
  const float* kernel_offset = kernel_d_c + offset;
  const IMAGE_T* src_idx = src_d + idx;
  float value = 0.0f;

  // Handle boundary condition
  for (int i = -offset; i < offset_neg; ++i) {
    value += src_idx[offset_neg*heightWidth] * kernel_offset[i];
  }

  for (int i = offset_neg; i < offset_pos; ++i) {
    value += src_idx[i*heightWidth] * kernel_offset[i];
  }

  // Handle boundary condition
  for (int i = offset_pos; i <= offset; ++i) {
    value += src_idx[offset_pos*heightWidth] * kernel_offset[i];
  }

  convert_type_and_assign_val(dst_d, idx, value);
}


template <typename IMAGE_T, const int offset>
__global__ void convolution_z_kernel(const IMAGE_T* __restrict__ src_d,
    IMAGE_T* __restrict__ dst_d, int width, int height, int depth) {
  const int x = blockIdx.x * ZCONV_BLOCKDIM_X + threadIdx.x;
  const int y = blockIdx.y * ZCONV_BLOCKDIM_Y + threadIdx.y;
  const int z = blockIdx.z * ZCONV_BLOCKDIM_Z + threadIdx.z;

  if (x >= width || y >= height || z >= depth) {
    return;
  }

  const long idx = z * long(height) * long(width) + y * long(width) + x;
  const int offset_neg = z - offset >= 0 ? -offset : -z;
  const int offset_pos = z + offset < depth ? offset : depth - z - 1;
  const long heightWidth = long(height) * long(width);
  const float* kernel_offset = kernel_d_c + offset;
  const IMAGE_T* src_idx = src_d + idx;
  float value = 0.0f;

  if (offset_neg == -offset && offset_pos == offset) {
#pragma unroll
    for (int i = -offset; i <= offset; ++i) {
      value += src_idx[i*heightWidth] * kernel_offset[i];
    }
  } else {
    // Handle boundary condition
    for (int i = -offset; i < offset_neg; ++i) {
      value += src_idx[offset_neg*heightWidth] * kernel_offset[i];
    }

    for (int i = offset_neg; i < offset_pos; ++i) {
      value += src_idx[i*heightWidth] * kernel_offset[i];
    }

    // Handle boundary condition
    for (int i = offset_pos; i <= offset; ++i) {
      value += src_idx[offset_pos*heightWidth] * kernel_offset[i];
    }
  }


  convert_type_and_assign_val(dst_d, idx, value);
}


template <typename IMAGE_T>
void gaussian1D_x_cuda(IMAGE_T* src_d, IMAGE_T* dst_d, int kernel_size,
                       int width, int height, int depth) {
  long x_grid = (width + XCONV_BLOCKDIM_X - 1)/XCONV_BLOCKDIM_X;
  long y_grid = (height + XCONV_BLOCKDIM_Y - 1)/XCONV_BLOCKDIM_Y;
  long z_grid = (depth + XCONV_BLOCKDIM_Z - 1)/XCONV_BLOCKDIM_Z;
  dim3 block(XCONV_BLOCKDIM_X, XCONV_BLOCKDIM_Y, XCONV_BLOCKDIM_Z);
  dim3 grid(x_grid, y_grid, z_grid);

  // XXX TODO 
  // This comment applies to all gaussian1D_XX_cuda kernels.
  // We should probably figure out a few different sizes like
  // 7, 15, 25, 35, 47, 59, etc that are farther apart
  // and then for each blur round the kernel size up to the nearest
  // supported size. Since we are ~doubling the sigma after each round
  // of blurring this current scheme is inefficient as only a few are used
  // before falling through to the default case.
  // If we build the kernel using the same sigma but a larger kernel_size
  // it should generate the same (or slightly more accurate) results
  // and the perf gained of using a templated kernel is much greater
  // than the hit from using a slightly larger than needed kernel_size

  switch (kernel_size) {
    case 3:
    convolution_x_kernel_s<IMAGE_T, 3/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 5:
    convolution_x_kernel_s<IMAGE_T, 5/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 7:
    convolution_x_kernel_s<IMAGE_T, 7/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 9:
    convolution_x_kernel_s<IMAGE_T, 9/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 11:
    convolution_x_kernel_s<IMAGE_T, 11/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 13:
    convolution_x_kernel_s<IMAGE_T, 13/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 15:
    convolution_x_kernel_s<IMAGE_T, 15/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 17:
    convolution_x_kernel_s<IMAGE_T, 17/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 19:
    convolution_x_kernel_s<IMAGE_T, 19/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 21:
    convolution_x_kernel_s<IMAGE_T, 21/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    default:
    convolution_x_kernel<IMAGE_T><<<grid, block>>>(src_d, dst_d,
                               kernel_size/2, width, height, depth);
    break;
  }

  CUERR // check and clear any existing errors
}


template <typename IMAGE_T>
void gaussian1D_y_cuda(IMAGE_T* src_d, IMAGE_T* dst_d, int kernel_size,
                       int width, int height, int depth) {
  long x_grid = (width + YCONV_BLOCKDIM_X - 1)/YCONV_BLOCKDIM_X;
  long y_grid = (height + YCONV_BLOCKDIM_Y - 1)/YCONV_BLOCKDIM_Y;
  long z_grid = (depth + YCONV_BLOCKDIM_Z - 1)/YCONV_BLOCKDIM_Z;
  dim3 block(YCONV_BLOCKDIM_X, YCONV_BLOCKDIM_Y, YCONV_BLOCKDIM_Z);
  dim3 grid(x_grid, y_grid, z_grid);

  switch (kernel_size) {
    case 3:
    convolution_y_kernel_s<IMAGE_T, 3/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 5:
    convolution_y_kernel_s<IMAGE_T, 5/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 7:
    convolution_y_kernel_s<IMAGE_T, 7/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 9:
    convolution_y_kernel_s<IMAGE_T, 9/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 11:
    convolution_y_kernel_s<IMAGE_T, 11/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 13:
    convolution_y_kernel_s<IMAGE_T, 13/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 15:
    convolution_y_kernel_s<IMAGE_T, 15/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 17:
    convolution_y_kernel_s<IMAGE_T, 17/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 19:
    convolution_y_kernel_s<IMAGE_T, 19/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 21:
    convolution_y_kernel_s<IMAGE_T, 21/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    default:
    convolution_y_kernel<IMAGE_T><<<grid, block>>>(src_d, dst_d,
                               kernel_size/2, width, height, depth);
    break;
  }

  CUERR // check and clear any existing errors
}


template <typename IMAGE_T>
void gaussian1D_z_cuda(IMAGE_T* src_d, IMAGE_T* dst_d, int kernel_size,
                       int width, int height, int depth) {
  long x_grid = (width + ZCONV_BLOCKDIM_X - 1)/ZCONV_BLOCKDIM_X;
  long y_grid = (height + ZCONV_BLOCKDIM_Y - 1)/ZCONV_BLOCKDIM_Y;
  long z_grid = (depth + ZCONV_BLOCKDIM_Z - 1)/ZCONV_BLOCKDIM_Z;
  dim3 block(ZCONV_BLOCKDIM_X, ZCONV_BLOCKDIM_Y, ZCONV_BLOCKDIM_Z);
  dim3 grid(x_grid, y_grid, z_grid);

  switch (kernel_size) {
    case 3:
    convolution_z_kernel_s<IMAGE_T, 3/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 5:
    convolution_z_kernel_s<IMAGE_T, 5/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 7:
    convolution_z_kernel_s<IMAGE_T, 7/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 9:
    convolution_z_kernel_s<IMAGE_T, 9/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 11:
    convolution_z_kernel_s<IMAGE_T, 11/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 13:
    convolution_z_kernel_s<IMAGE_T, 13/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 15:
    convolution_z_kernel_s<IMAGE_T, 15/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 17:
    convolution_z_kernel_s<IMAGE_T, 17/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 19:
    convolution_z_kernel_s<IMAGE_T, 19/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    case 21:
    convolution_z_kernel_s<IMAGE_T, 21/2><<<grid, block>>>(src_d, dst_d,
                                          width, height, depth);
    break;
    default:
    convolution_z_kernel<IMAGE_T><<<grid, block>>>(src_d, dst_d,
                               kernel_size/2, width, height, depth);
    break;
  }

  CUERR // check and clear any existing errors
}


