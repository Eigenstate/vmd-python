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
 *      $RCSfile: CUDAWatershed.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.33 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA-accelerated Watershed image segmentation
 ***************************************************************************/
#include <cstdio>

#define WATERSHED_INTERNAL 1
#include "CUDAWatershed.h"
#include "ProfileHooks.h"

#define px_offset_d    (1)
#define py_offset_d    (width)
#define pz_offset_d    (heightWidth)
#define nx_offset_d    (-1)
#define ny_offset_d    (-width)
#define nz_offset_d    (-heightWidth)

#define nx_ny_offset_d (-1 - width)
#define nx_py_offset_d (-1 + width)
#define px_py_offset_d (1 + width)
#define px_ny_offset_d (1 - width)

#define px_pz_offset_d (1 + heightWidth)
#define nx_nz_offset_d (-1 - heightWidth)
#define nx_pz_offset_d (-1 + heightWidth)
#define px_nz_offset_d (1 - heightWidth)

#define py_pz_offset_d (width + heightWidth)
#define ny_nz_offset_d (-width - heightWidth)
#define ny_pz_offset_d (-width + heightWidth)
#define py_nz_offset_d (width - heightWidth)

//#define GPU_BLOCK_UPDATE
#define GPU_WARP_UPDATE

#ifdef GPU_BLOCK_UPDATE

#define BLOCK_X_DIM 128
#define BLOCK_Y_DIM 2
#define BLOCK_Z_DIM 2

#else

// BLOCK_X_DIM must be 2^n
#define BLOCK_X_DIM 64
#define BLOCK_Y_DIM 2
#define BLOCK_Z_DIM 2

#endif

// 
// GPU-side constant memory and change flag globals used 
// by the neighbor calculation and update kernels
//
__constant__ int neighbor_offsets_d[19];
__device__ unsigned int changes_d;

#define INST_DESTROY_GPU(G_T) \
template void destroy_gpu<G_T,float>(watershed_gpu_state_t<G_T,float> &gpu_state);\
template void destroy_gpu<G_T,unsigned short>(watershed_gpu_state_t<G_T,unsigned short> &gpu_state);\
template void destroy_gpu<G_T,unsigned char>(watershed_gpu_state_t<G_T,unsigned char> &gpu_state);

#define INST_INIT_GPU_ON_DEV(G_T) \
template bool init_gpu_on_device<G_T, float>(watershed_gpu_state_t<G_T,float> &gpu_state, float* image, int imageongpu,\
                                             unsigned int w, unsigned int h, unsigned int d);\
template bool init_gpu_on_device<G_T, unsigned short>(watershed_gpu_state_t<G_T,unsigned short> &gpu_state,\
                                unsigned short* image, int imageongpu, unsigned int w, unsigned int h, unsigned int d);\
template bool init_gpu_on_device<G_T, unsigned char>(watershed_gpu_state_t<G_T,unsigned char> &gpu_state,\
                                unsigned char* image, int imageongpu, unsigned int w, unsigned int h, unsigned int d);\

#define INST_INIT_GPU(G_T) \
template bool init_gpu<G_T, float>(state_t<G_T,float> &state, int *eq_and_lower,\
                             watershed_gpu_state_t<G_T,float> &gpu_state,\
                             unsigned int w, unsigned int h, unsigned int d);\
template bool init_gpu<G_T, unsigned short>(state_t<G_T,unsigned short> &state, int *eq_and_lower,\
                             watershed_gpu_state_t<G_T,unsigned short> &gpu_state,\
                             unsigned int w, unsigned int h, unsigned int d);\
template bool init_gpu<G_T, unsigned char>(state_t<G_T,unsigned char> &state, int *eq_and_lower,\
                             watershed_gpu_state_t<G_T,unsigned char> &gpu_state,\
                             unsigned int w, unsigned int h, unsigned int d);

#define INST_UPDATE_CUDA(G_T) \
template void update_cuda<G_T,float>(watershed_gpu_state_t<G_T,float> &gpu_state, \
                                G_T *final_segments_d);\
template void update_cuda<G_T,unsigned short>(watershed_gpu_state_t<G_T,unsigned short> &gpu_state, \
                                G_T *final_segments_d);\
template void update_cuda<G_T,unsigned char>(watershed_gpu_state_t<G_T,unsigned char> &gpu_state, \
                                G_T *final_segments_d);

INST_DESTROY_GPU(unsigned long)
INST_DESTROY_GPU(unsigned int)
INST_DESTROY_GPU(unsigned short)

INST_INIT_GPU_ON_DEV(unsigned long)
INST_INIT_GPU_ON_DEV(unsigned int)
INST_INIT_GPU_ON_DEV(unsigned short)

INST_INIT_GPU(unsigned long)
INST_INIT_GPU(unsigned int)
INST_INIT_GPU(unsigned short)

INST_UPDATE_CUDA(unsigned long)
INST_UPDATE_CUDA(unsigned int)
INST_UPDATE_CUDA(unsigned short)

//
// Explicit template instantiations so that all required variants are 
// compiled and available at link time.
//
// instantiate for: long, float
template void update_cuda<long,float>(watershed_gpu_state_t<long,float> &gpu_state, 
                                long *final_segments_d);
// instantiate for: long, unsigned short
template void update_cuda<long,unsigned short>(watershed_gpu_state_t<long,unsigned short> &gpu_state, 
                                long *final_segments_d);
// instantiate for: long, unsigned char
template void update_cuda<long,unsigned char>(watershed_gpu_state_t<long,unsigned char> &gpu_state, 
                                long *final_segments_d);

// instantiate for: int, float
template void update_cuda<int,float>(watershed_gpu_state_t<int,float> &gpu_state, 
                               int *final_segments_d);
// instantiate for: int, short
template void update_cuda<int,unsigned short>(watershed_gpu_state_t<int,unsigned short> &gpu_state, 
                               int *final_segments_d);
// instantiate for: int, unsigned char
template void update_cuda<int,unsigned char>(watershed_gpu_state_t<int,unsigned char> &gpu_state, 
                               int *final_segments_d);

// instantiate for: short, float
template void update_cuda<short,float>(watershed_gpu_state_t<short,float> &gpu_state,
                                 short *final_segments_d);
// instantiate for: short, short
template void update_cuda<short,unsigned short>(watershed_gpu_state_t<short,unsigned short> &gpu_state,
                                 short *final_segments_d);
// instantiate for: short, char
template void update_cuda<short,unsigned char>(watershed_gpu_state_t<short,unsigned char> &gpu_state,
                                 short *final_segments_d);


#define DIV_BY_32 5
#define GIGABYTE (1024.0f*1024.0f*1024.0f)

template <typename GROUP_T, typename IMAGE_T>
__global__ void update_kernel(GROUP_T* __restrict__ current_group,
                              GROUP_T* __restrict__ next_group,
                              IMAGE_T* __restrict__ current_value,
                              IMAGE_T* __restrict__ next_value,
                              int* __restrict__ eq_and_lower,
                              unsigned char* __restrict__ current_update,
                              unsigned char* __restrict__ next_update,
                              const int width,
                              const int height,
                              const int depth);

template <typename GROUP_T, typename IMAGE_T>
void init_neighbor_offsets(watershed_gpu_state_t<GROUP_T, IMAGE_T> &gpu_state) {
  int height = gpu_state.height;
  int width = gpu_state.width;
  int heightWidth = height * width;
  int neighbor_offsets_t[19] = { 0,
    px_offset,
    py_offset,
    pz_offset,
    nx_offset,
    ny_offset,
    nz_offset,

    nx_ny_offset,
    nx_py_offset,
    px_py_offset,
    px_ny_offset,

    px_pz_offset,
    nx_nz_offset,
    nx_pz_offset,
    px_nz_offset,

    py_pz_offset,
    ny_nz_offset,
    ny_pz_offset,
    py_nz_offset,
  };  

  cudaMemcpyToSymbolAsync(neighbor_offsets_d, neighbor_offsets_t, sizeof(int) * 19);
}


#define CALCULATE_NEIGHBORS_CUDA(offset_str) {\
  const long idx_offset = idx + offset_str##_offset_d;\
  slope = curr_intensity - image[idx_offset];\
  if (slope < smallest_slope) {\
    smallest_slope = slope;\
    curr_lower = offset_str##_idx;\
  } else if (slope >= -FLOAT_DIFF && slope <= FLOAT_DIFF) {\
    curr_n_eq |= offset_str;\
    if (idx_offset < min_group) {\
      min_group = idx_offset;\
    }\
  } \
  }


// kernel for computing initial neighbor arrays on the GPU
template <typename GROUP_T, typename IMAGE_T>
__global__ void calc_neighbors_kernel(const IMAGE_T* __restrict__ image,
                                      GROUP_T* __restrict__ next_group,
                                      IMAGE_T* __restrict__ curr_value,
                                      IMAGE_T* __restrict__ next_value,
                                      int* __restrict__ eq_and_lower,
                                      const int width,
                                      const int height,
                                      const int depth) {

  const int x = blockIdx.x * BLOCK_X_DIM + threadIdx.x;
  const int y = blockIdx.y * BLOCK_Y_DIM + threadIdx.y;
  const int z = blockIdx.z * BLOCK_Z_DIM + threadIdx.z;
  const long idx = z * long(height) * long(width) + y * long(width) + x;
  const int heightWidth = height * width;

  if (x >= width || y >= height || z >= depth) {
    return;
  }

  //
  // beginning of neighbor calculation macros 
  //
  float slope;
  float smallest_slope = -FLOAT_DIFF;
  IMAGE_T curr_intensity = image[idx];
  long min_group = idx; // Group number = idx to start
  int curr_n_eq = 0;
  long curr_lower = 0;
  //TODO change to be actual offset instead of index into offset array

#ifdef CONNECTED_18
  if (x > 0) {
    CALCULATE_NEIGHBORS_CUDA(nx);
  }
  if (x < width-1) {
    CALCULATE_NEIGHBORS_CUDA(px);
  }

  /* Calculate n_lower and n_eq */
  if (y > 0) {
    CALCULATE_NEIGHBORS_CUDA(ny);
    if (x < width - 1) {
      CALCULATE_NEIGHBORS_CUDA(px_ny);
    }
  }
  if (y < height-1) {
    CALCULATE_NEIGHBORS_CUDA(py);
    if (x < width - 1) {
      CALCULATE_NEIGHBORS_CUDA(px_py);
    }
  }

  if (z > 0) {
    CALCULATE_NEIGHBORS_CUDA(nz);
    if (x < width - 1) {
      CALCULATE_NEIGHBORS_CUDA(px_nz);
    }
    if (x > 0) {
      CALCULATE_NEIGHBORS_CUDA(nx_nz);
    }
    if (y > 0 ) {
      CALCULATE_NEIGHBORS_CUDA(ny_nz);
    }
    if (y < height - 1) {
      CALCULATE_NEIGHBORS_CUDA(py_nz);
    }
  }
  if (z < depth-1) {
    CALCULATE_NEIGHBORS_CUDA(pz);
    if (x < width - 1) {
      CALCULATE_NEIGHBORS_CUDA(px_pz);
    }
    if (x > 0) {
      CALCULATE_NEIGHBORS_CUDA(nx_pz);
    }
    if (y < height - 1) {
      CALCULATE_NEIGHBORS_CUDA(py_pz);
    }
    if (y > 0) {
      CALCULATE_NEIGHBORS_CUDA(ny_pz);
    }
  }
#else // 6 connected neighbors
  if (x > 0) {
    CALCULATE_NEIGHBORS_CUDA(nx);
  }
  if (x < width-1) {
    CALCULATE_NEIGHBORS_CUDA(px);
  }

  /* Calculate n_lower and n_eq */
  if (y > 0) {
    CALCULATE_NEIGHBORS_CUDA(ny);
  }
  if (y < height-1) {
    CALCULATE_NEIGHBORS_CUDA(py);
  }

  if (z > 0) {
    CALCULATE_NEIGHBORS_CUDA(nz);
  }
  if (z < depth-1) {
    CALCULATE_NEIGHBORS_CUDA(pz);
  }
#endif // CONNECTED_18

  /* Update current voxel */
  eq_and_lower[idx] = MAKE_EQUAL_AND_LOWER(curr_n_eq, curr_lower);
  if (curr_lower == 0) {
    /* Minimum */
    next_group[idx] = min_group;
    next_value[idx] = image[idx];
    curr_value[idx] = image[idx];
  } else {
    /* Not minimum */
    const long nIdx = idx + neighbor_offsets_d[curr_lower];
    next_value[idx] = image[nIdx];
    curr_value[idx] = image[nIdx];
    next_group[idx] = nIdx;
  }
}


template <typename GROUP_T, typename IMAGE_T>
void set_gpu_state(state_t<GROUP_T, IMAGE_T> &state, int *eq_and_lower, 
                   watershed_gpu_state_t<GROUP_T, IMAGE_T> &gpu_state,
                   unsigned long nVoxels) {
  cudaMemcpy(gpu_state.segments_d, state.group, 
             nVoxels * sizeof(GROUP_T), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_state.current_value_d, state.value, 
             nVoxels * sizeof(IMAGE_T), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_state.next_value_d, gpu_state.current_value_d, 
             nVoxels * sizeof(IMAGE_T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(gpu_state.eq_and_lower_d, eq_and_lower, 
             nVoxels * sizeof(int), cudaMemcpyHostToDevice);
}


// Completely GPU-based initialization routine, to replace CPU-based initialization
template <typename GROUP_T, typename IMAGE_T>
bool init_gpu_on_device(watershed_gpu_state_t<GROUP_T, IMAGE_T> &gpu_state, 
                        IMAGE_T* image, int imageongpu,
                        unsigned int w, unsigned int h, unsigned int d) {
  PROFILE_PUSH_RANGE("CUDAWatershed::init_gpu_on_device()", 0);

  unsigned int changes = 0;
  unsigned long nVoxels = long(h) * long(w) * long(d);
  gpu_state.width = w;
  gpu_state.height = h;
  gpu_state.depth = d;
  IMAGE_T* image_d;

  cudaMemcpyToSymbolAsync(changes_d, &changes, sizeof(unsigned int));

  if (imageongpu) {
    image_d = image;
  } else {  
    cudaMalloc(&image_d, nVoxels * sizeof(IMAGE_T));
    cudaMemcpyAsync(image_d, image, nVoxels * sizeof(IMAGE_T), cudaMemcpyHostToDevice);
  }

  cudaMalloc(&gpu_state.segments_d, nVoxels * sizeof(GROUP_T));
  cudaMalloc(&gpu_state.current_value_d, nVoxels * sizeof(IMAGE_T));
  cudaMalloc(&gpu_state.next_value_d, nVoxels * sizeof(IMAGE_T));
  cudaMalloc(&gpu_state.eq_and_lower_d, nVoxels * sizeof(int));

#ifdef GPU_BLOCK_UPDATE
  long t_width = ceil((float)gpu_state.width / BLOCK_X_DIM);
  long t_height = ceil((float)gpu_state.height / BLOCK_Y_DIM);
  long t_depth = ceil((float)gpu_state.depth / BLOCK_Z_DIM);
  long num_blocks = t_width * t_height * t_depth;
  cudaMalloc(&gpu_state.next_update_d, num_blocks * sizeof(char));
  cudaMalloc(&gpu_state.current_update_d, num_blocks * sizeof(char));
  cudaMemsetAsync(gpu_state.next_update_d, 0, num_blocks * sizeof(char));
  cudaMemsetAsync(gpu_state.current_update_d, 1, num_blocks * sizeof(char));
#endif // GPU_BLOCK_UPDATE

#ifdef GPU_WARP_UPDATE
  long warp_x_dim = ceil(gpu_state.width/32.0f);
  long num_warps = ceil(gpu_state.height/32.0f) * 32L * ceil(gpu_state.depth/32.0f)* 32L * warp_x_dim;
  cudaMalloc(&gpu_state.next_update_d, num_warps * sizeof(char));
  cudaMalloc(&gpu_state.current_update_d, num_warps * sizeof(char));
  cudaMemsetAsync(gpu_state.next_update_d, 0, num_warps * sizeof(char));
  cudaMemsetAsync(gpu_state.current_update_d, 1, num_warps * sizeof(char));
#endif // GPU_WARP_UPDATE

  init_neighbor_offsets(gpu_state);

  const dim3 block(BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM);
  size_t x_grid = ceil((float)gpu_state.width/BLOCK_X_DIM);
  size_t y_grid = ceil((float)gpu_state.height/BLOCK_Y_DIM);
  size_t z_grid = ceil((float)gpu_state.depth/BLOCK_Z_DIM);
  const dim3 grid(x_grid, y_grid, z_grid);
  calc_neighbors_kernel<<<grid, block>>>(image_d, gpu_state.segments_d, gpu_state.current_value_d,
                                         gpu_state.next_value_d, gpu_state.eq_and_lower_d, w, h, d);

#if 0
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error initializing device: %s\n", cudaGetErrorString(error));
    destroy_gpu(gpu_state);
    PROFILE_POP_RANGE();
    return false;
  }
#endif

  gpu_state.init = true;

  // only free the image if it was a temp copy from the host
  if (!imageongpu) 
    cudaFree(image_d);

  PROFILE_POP_RANGE();

  return true;
}


template <typename GROUP_T, typename IMAGE_T>
bool init_gpu(state_t<GROUP_T, IMAGE_T> &state, int *eq_and_lower, 
              watershed_gpu_state_t<GROUP_T, IMAGE_T> &gpu_state,
              unsigned int w, unsigned int h, unsigned int d) {
  PROFILE_MARK("CUDAWatershed::init_gpu()", 0);

  unsigned int changes = 0;
  unsigned long nVoxels = long(h) * long(w) * long(d);
  gpu_state.width = w;
  gpu_state.height = h;
  gpu_state.depth = d;

  cudaMalloc(&gpu_state.segments_d, nVoxels * sizeof(long));
  cudaMalloc(&gpu_state.current_value_d, nVoxels * sizeof(IMAGE_T));
  cudaMalloc(&gpu_state.next_value_d, nVoxels * sizeof(IMAGE_T));
  cudaMalloc(&gpu_state.eq_and_lower_d, nVoxels * sizeof(int));

#ifdef GPU_BLOCK_UPDATE
  long t_width = ceil((float)gpu_state.width / BLOCK_X_DIM);
  long t_height = ceil((float)gpu_state.height / BLOCK_Y_DIM);
  long t_depth = ceil((float)gpu_state.depth / BLOCK_Z_DIM);
  long num_blocks = t_width * t_height * t_depth;
  cudaMalloc(&gpu_state.next_update_d, num_blocks * sizeof(char));
  cudaMalloc(&gpu_state.current_update_d, num_blocks * sizeof(char));
  cudaMemset(gpu_state.next_update_d, 0, num_blocks * sizeof(char));
  cudaMemset(gpu_state.current_update_d, 1, num_blocks * sizeof(char));
#endif // GPU_BLOCK_UPDATE

#ifdef GPU_WARP_UPDATE
  long warp_x_dim = ceil(gpu_state.width/32.0f);
  long num_warps = ceil(gpu_state.height/32.0f) * 32L * ceil(gpu_state.depth/32.0f) * 32L * warp_x_dim;
  cudaMalloc(&gpu_state.next_update_d, num_warps * sizeof(char));
  cudaMalloc(&gpu_state.current_update_d, num_warps * sizeof(char));
  cudaMemset(gpu_state.next_update_d, 0, num_warps * sizeof(char));
  cudaMemset(gpu_state.current_update_d, 1, num_warps * sizeof(char));
#endif // GPU_WARP_UPDATE

  cudaMemcpyToSymbol(changes_d, &changes, sizeof(unsigned int));
  init_neighbor_offsets(gpu_state);
  gpu_state.init = true;

#if 0
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error initializing device2: %s\n", cudaGetErrorString(error));
    destroy_gpu(gpu_state);
    return false;
  }
#endif

  // copy host-side arrays to the GPU
  set_gpu_state(state, eq_and_lower, gpu_state, nVoxels);

  return true;
}


template <typename GROUP_T, typename IMAGE_T>
void destroy_gpu(watershed_gpu_state_t<GROUP_T, IMAGE_T> &gpu_state) {
#if defined(VERBOSE)
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error before destroy_gpu: %s\n", cudaGetErrorString(error));
  }
#endif

  if (gpu_state.init) {
    if (gpu_state.segments_d != NULL) {
      cudaFree(gpu_state.segments_d);
      gpu_state.segments_d = NULL;
    }

    if (gpu_state.current_value_d != NULL) {
      cudaFree(gpu_state.current_value_d);
      gpu_state.current_value_d = NULL;
    }
    if (gpu_state.next_value_d != NULL) {
      cudaFree(gpu_state.next_value_d);
      gpu_state.next_value_d = NULL;
    }


    if (gpu_state.eq_and_lower_d != NULL) {
      cudaFree(gpu_state.eq_and_lower_d);
      gpu_state.eq_and_lower_d = NULL;
    }

#ifdef GPU_BLOCK_UPDATE
    if (gpu_state.current_update_d != NULL) {
      cudaFree(gpu_state.current_update_d);
      gpu_state.current_update_d = NULL;
    }
    if (gpu_state.next_update_d != NULL) {
      cudaFree(gpu_state.next_update_d);
      gpu_state.next_update_d = NULL;
    }
    if (gpu_state.current_update_d != NULL) {
      cudaFree(gpu_state.current_update_d);
      gpu_state.current_update_d = NULL;
    }
    if (gpu_state.next_update_d != NULL) {
      cudaFree(gpu_state.next_update_d);
      gpu_state.next_update_d = NULL;
    }
#endif // GPU_BLOCK_UPDATE
    gpu_state.init = false;
  }

#if defined(VERBOSE)
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Error after destroy_gpu: %s\n", cudaGetErrorString(error));
  }
#endif
}


template <typename GROUP_T, typename IMAGE_T>
void update_cuda(watershed_gpu_state_t<GROUP_T, IMAGE_T> &gpu_state, GROUP_T *final_segments_d) {
  const dim3 block(BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM);
  size_t x_grid = ceil((float)gpu_state.width/BLOCK_X_DIM);
  size_t y_grid = ceil((float)gpu_state.height/BLOCK_Y_DIM);
  size_t z_grid = ceil((float)gpu_state.depth/BLOCK_Z_DIM);
  const dim3 grid(x_grid, y_grid, z_grid);
  unsigned int changes = 1;
  const unsigned int zero = 0;
  GROUP_T *current_group_d = gpu_state.segments_d;;
  GROUP_T *next_group_d = final_segments_d;

#ifdef VERBOSE
  printf("Block =  %d x %d x %d = %d\n", block.x, block.y, block.z, block.x*block.y*block.z);
  printf("Grid =  %d x %d x %d = %d\n", grid.x, grid.y, grid.z, grid.x*grid.y*grid.z);
  printf("Threads = %d\n", grid.x*grid.y*grid.z*block.x*block.y*block.z);
  printf("Image = %d x %d x %d\n", gpu_state.height, gpu_state.width, gpu_state.depth);
  printf("Voxels  = %d\n", gpu_state.height * gpu_state.width * gpu_state.depth);
#endif // VERBOSE

  //
  // run update_kernel in a loop until no changes are flagged in changes_d
  // XXX we may want to modify this scheme to reduce or eliminate the
  //     cudaMemcpyToSymbol() calls, by making an "eager" loop that
  //     only checks the outcome every N iterations (along with a kernel
  //     early exit scheme), by using writes via host-mapped memory, 
  //     or a similar technique.
  while (changes) {
    cudaMemcpyToSymbolAsync(changes_d, &zero, sizeof(changes_d));

    update_kernel<GROUP_T><<<grid, block>>>(current_group_d, next_group_d, gpu_state.current_value_d,
                                      gpu_state.next_value_d, gpu_state.eq_and_lower_d,
                                      gpu_state.current_update_d, gpu_state.next_update_d,
                                      gpu_state.height, gpu_state.width, gpu_state.depth);

    SWAP(current_group_d, next_group_d, GROUP_T*);
    SWAP(gpu_state.current_value_d, gpu_state.next_value_d, IMAGE_T*);
    SWAP(gpu_state.current_update_d, gpu_state.next_update_d, unsigned char*);

    cudaMemcpyFromSymbol(&changes, changes_d, sizeof(changes));

#if defined(VERBOSE)
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("Error in update: %s\n", cudaGetErrorString(error));
      destroy_gpu(gpu_state);
      exit(-1);
    }
#endif // VERBOSE
  }

  // copy segment groups if necessary
  if (current_group_d != final_segments_d) {
    long nVoxels = gpu_state.height * gpu_state.width * gpu_state.depth;
    cudaMemcpy(final_segments_d, current_group_d,
               nVoxels * sizeof(GROUP_T), cudaMemcpyDeviceToDevice);
  }
}


#define UPDATE_VOXEL_CUDA(and_value, idx, curr_group, smallest_value) {\
  if (n_eq & and_value) {\
    const long offset_idx = idx + and_value##_offset_d;\
    if (current_value[offset_idx] + FLOAT_DIFF < smallest_value) {\
      smallest_value = current_value[offset_idx];\
      offset_number = and_value##_idx;\
      next_idx = idx + and_value##_offset_d;\
    }\
    curr_group = current_group[offset_idx] < curr_group ?\
    current_group[offset_idx] : curr_group;\
  }}


template <typename GROUP_T, typename IMAGE_T>
__global__ void update_kernel(GROUP_T* __restrict__ current_group,
                              GROUP_T* __restrict__ next_group,
                              IMAGE_T* __restrict__ current_value,
                              IMAGE_T* __restrict__ next_value,
                              int* __restrict__ eq_and_lower,
                              unsigned char* __restrict__ current_update,
                              unsigned char* __restrict__ next_update,
                              const int width,
                              const int height,
                              const int depth) {

  const int tx = blockIdx.x * BLOCK_X_DIM + threadIdx.x;
  const int ty = blockIdx.y * BLOCK_Y_DIM + threadIdx.y;
  const int tz = blockIdx.z * BLOCK_Z_DIM + threadIdx.z;
  const long idx = tz * long(height) * long(width) + ty * long(width) + tx;
  const int heightWidth = height * width;

  if (tx >= width || ty >= height || tz >= depth) {
    return;
  }

#ifdef GPU_BLOCK_UPDATE
  const long block_x = blockIdx.x;
  const long block_y = blockIdx.y;
  const long block_z = blockIdx.z;
  const long block_idx = block_z * gridDim.y * gridDim.x + block_y * gridDim.x + block_x;
  if (current_update[block_idx] == 0) {
    return;
  }
  current_update[block_idx] = 0;
#endif // GPU_BLOCK_UPDATE

#ifdef GPU_WARP_UPDATE
  const long warp_x = tx >> DIV_BY_32;
  const long warp_y = ty;
  const long warp_z = tz;
  const long warp_x_dim = (BLOCK_X_DIM * gridDim.x) >> DIV_BY_32;
  const long warp_y_dim = BLOCK_Y_DIM * gridDim.y;
  const long warp_z_dim = BLOCK_Z_DIM * gridDim.z;
  const long warp_idx = warp_z * warp_y_dim * warp_x_dim + warp_y * warp_x_dim + warp_x;

  if (current_update[warp_idx] == 0) {
    return;
  }
  current_update[warp_idx] = 0;
#endif // GPU_WARP_UPDATE

  bool update = false;
  const int equal_and_lower = eq_and_lower[idx];
  const int offset_number = GET_N_LOWER(equal_and_lower);
  if (offset_number != 0) {
    const long nidx = idx + neighbor_offsets_d[offset_number];
    if (next_group[idx] != current_group[nidx] || next_value[idx] != current_value[nidx]) {
      next_group[idx] = current_group[nidx];
      next_value[idx] = current_value[nidx];
      update = true;
    }
  } else {
    const int n_eq = GET_N_EQ(equal_and_lower);
    IMAGE_T smallest_value = current_value[idx];
    GROUP_T curr_group = current_group[idx];
    int offset_number = 0;
    long next_idx = 0;

#ifdef CONNECTED_18
    /* +-x */
    UPDATE_VOXEL_CUDA(nx, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(px, idx, curr_group, smallest_value);

    /* +-x, -y*/
    UPDATE_VOXEL_CUDA(nx_ny, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(   ny, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(px_ny, idx, curr_group, smallest_value);

    /* +-x, +y*/
    UPDATE_VOXEL_CUDA(nx_py, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(   py, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(px_py, idx, curr_group, smallest_value);

    /* +-x, +z*/
    UPDATE_VOXEL_CUDA(px_pz, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(nx_pz, idx, curr_group, smallest_value);
    /* +-y, +z*/
    UPDATE_VOXEL_CUDA(py_pz, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(   pz, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(ny_pz, idx, curr_group, smallest_value);

    /* +-x, -z*/
    UPDATE_VOXEL_CUDA(nx_nz, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(   nz, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(px_nz, idx, curr_group, smallest_value);
    /* +-y, -z*/
    UPDATE_VOXEL_CUDA(py_nz, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(ny_nz, idx, curr_group, smallest_value);

#else // 6 connected neighbors
    UPDATE_VOXEL_CUDA(nx, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(px, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(ny, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(py, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(nz, idx, curr_group, smallest_value);
    UPDATE_VOXEL_CUDA(pz, idx, curr_group, smallest_value);
#endif // CONNECTED_18

    if (offset_number != 0) {
      eq_and_lower[idx] = MAKE_EQUAL_AND_LOWER(n_eq, offset_number);
      next_value[idx] = smallest_value;
      next_group[idx] = current_group[next_idx];
      update = true;
    } else if (curr_group != next_group[idx]) {
      next_group[idx] = curr_group;
      update = true;
    }
  }

  if (update) {
    changes_d = 1;

#ifdef GPU_BLOCK_UPDATE
    next_update[block_idx] = 1;

    if (block_x > 0) {
      next_update[block_idx - 1] = 1;
    }
    if (block_x < gridDim.x - 1) {
      next_update[block_idx + 1] = 1;
    }
    if (block_y > 0) {
      next_update[block_idx - gridDim.x] = 1;
    }
    if (block_y < gridDim.y - 1) {
      next_update[block_idx + gridDim.x] = 1;
    }
    if (block_z > 0) {
      next_update[block_idx - (gridDim.x * gridDim.y)] = 1;
    }
    if (block_z < gridDim.z - 1) {
      next_update[block_idx + (gridDim.x * gridDim.y)] = 1;
    }
#endif // GPU_BLOCK_UPDATE

#ifdef GPU_WARP_UPDATE
    next_update[warp_idx] = 1;

    if (warp_x > 0) {
      next_update[warp_idx - 1] = 1;
    }
    if (warp_x < warp_x_dim - 1) {
      next_update[warp_idx + 1] = 1;
    }
    if (warp_y > 0) {
      next_update[warp_idx - warp_x_dim] = 1;
    }
    if (warp_y < warp_y_dim - 1) {
      next_update[warp_idx + warp_x_dim] = 1;
    }
    if (warp_z > 0) {
      next_update[warp_idx - (warp_x_dim * warp_y_dim)] = 1;
    }
    if (warp_z < warp_z_dim - 1) {
      next_update[warp_idx + (warp_x_dim * warp_y_dim)] = 1;
    }
#endif // GPU_WARP_UPDATE
  }
}

