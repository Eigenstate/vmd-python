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
 *      $RCSfile: CUDASegmentation.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.28 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   GPU-accelerated scale-space variant of Watershed image segmentation 
 *   intended for use on 3-D cryo-EM density maps.
 ***************************************************************************/

#include <stdio.h>
#include "CUDAParPrefixOps.h"
#include "CUDASegmentation.h"

#if 0
#define CUERR { cudaError_t err; \
  cudaDeviceSynchronize(); \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  }}
#else
#define CUERR
#endif

#define GROUP_BLOCK_SIZE 1024
#define VOXEL_BLOCK_SIZE 1024


//
// GPU-side global state variable listing the number of resulting groups
//
__device__ unsigned long nGroups_d;

void free_gpu_temp_storage(gpuseg_temp_storage *tmp) {
  if (tmp != NULL && tmp->tmp_d != NULL) {
    cudaFree(tmp->tmp_d);
  }
  tmp->tmp_d = NULL;
  tmp->sz = 0;
}


template <typename GROUP_T>
__global__ void update_groups_from_map(GROUP_T* __restrict__ groups_d,
                                       const GROUP_T* __restrict__ group_map_d,
                                       const long nVoxels) {
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (long i=idx; i<nVoxels; i+=gridDim.x * blockDim.x) {
    const GROUP_T old_group = groups_d[i];
    groups_d[i] = group_map_d[old_group];
  }
}


template <typename GROUP_T>
__global__ void init_group_map(const GROUP_T* __restrict__ groups_d,
                               GROUP_T* __restrict__ group_map_d,
                               const long nVoxels) {
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (long i=idx; i<nVoxels; i+=gridDim.x * blockDim.x) {
    const GROUP_T group_id = groups_d[i];
    group_map_d[group_id] = 1;
  }
}


template <typename GROUP_T>
__global__ void update_group_map(GROUP_T* __restrict__ group_map_d,
                                 const GROUP_T* __restrict__ scan_results,
                                 const long nGroups) {
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (long i = idx; i < nGroups; i += gridDim.x * blockDim.x) {
    if (i == nGroups - 1) {
      nGroups_d = group_map_d[i] + scan_results[i];
    }
    if (group_map_d[i] == 1) {
      group_map_d[i] = scan_results[i];
    }
  }
}


template <typename GROUP_T>
long sequentialize_groups_cuda(GROUP_T* groups_d, GROUP_T* group_map_d, 
                               unsigned long nVoxels, unsigned long nGroups, 
                               gpuseg_temp_storage *tmp,
                               gpuseg_temp_storage *scanwork) {
  int num_voxel_blocks = (nVoxels + VOXEL_BLOCK_SIZE-1) / VOXEL_BLOCK_SIZE;
  int num_group_blocks = (nGroups + GROUP_BLOCK_SIZE-1) / GROUP_BLOCK_SIZE;
  GROUP_T* scanout_d = NULL;
  void *scanwork_d = NULL;
  long scanwork_sz = 0;

  if (num_voxel_blocks > 65535)
    num_voxel_blocks = 65535;
  if (num_group_blocks > 65535)
    num_group_blocks = 65535;

  CUERR

  // if we've been provided with a tmp workspace handle, we use it,
  // otherwise we allocate and free tmp workspace on-the-fly
  if (tmp != NULL) {
    if (tmp->tmp_d == NULL || tmp->sz < nGroups * long(sizeof(GROUP_T))) {
      if (tmp->tmp_d != NULL)
        cudaFree(tmp->tmp_d);
      tmp->tmp_d = NULL;
      tmp->sz = 0;

      tmp->sz = nGroups * sizeof(GROUP_T);
      cudaMalloc(&tmp->tmp_d, tmp->sz);
    }
    scanout_d = static_cast<GROUP_T*>(tmp->tmp_d);
  } else {
    // XXX this is useful for profiling examples for GTC18, but after that 
    //     the non-persistent buffer code path should go away.
    cudaMalloc(&scanout_d, nGroups * sizeof(GROUP_T));
  } 
  CUERR // check and clear any existing errors 

#if defined(VMDUSECUB)
  // if we've been provided with a tmp workspace handle, we use it,
  // otherwise we allocate and free tmp workspace on-the-fly
  if (scanwork != NULL) {
    long sz = dev_excl_scan_sum_tmpsz(group_map_d, nGroups, scanout_d,
                                      GROUP_T(0));
    if (scanwork->tmp_d == NULL || scanwork->sz < sz) {
      if (scanwork->tmp_d != NULL)
        cudaFree(scanwork->tmp_d);
      scanwork->tmp_d = NULL;
      // scanwork->sz = 0;
      scanwork->sz = sz;
      cudaMalloc(&scanwork->tmp_d, scanwork->sz);
    }

    scanwork_d = scanwork->tmp_d;
    scanwork_sz = scanwork->sz;
  } 
#endif

  CUERR // check and clear any existing errors 

  // Init array to 0
  cudaMemsetAsync(group_map_d, 0, nGroups * sizeof(GROUP_T));
  CUERR // check and clear any existing errors 

  // Set groups with >= 1 voxel to 1
  init_group_map<GROUP_T><<<num_voxel_blocks, VOXEL_BLOCK_SIZE>>>(groups_d, group_map_d, nVoxels);
  CUERR // check and clear any existing errors 


  dev_excl_scan_sum(group_map_d, nGroups, scanout_d,
                    scanwork_d, scanwork_sz, GROUP_T(0));
  CUERR // check and clear any existing errors 

  // set new group numbers in group_map
  update_group_map<GROUP_T><<<num_group_blocks, GROUP_BLOCK_SIZE>>>(group_map_d, scanout_d, nGroups);
  CUERR // check and clear any existing errors 

  // update the groups_d array with new group numbers
  update_groups_from_map<<<num_voxel_blocks, VOXEL_BLOCK_SIZE>>>(groups_d, group_map_d, nVoxels);
  CUERR // check and clear any existing errors 

  // obtain the resulting group count from the GPU-side total
  cudaMemcpyFromSymbol(&nGroups, nGroups_d, sizeof(unsigned long));

  if (tmp == NULL) {
    cudaFree(scanout_d);
  } 

  return nGroups;
}


template <typename IN_T>
__device__ void getOrderedInt( IN_T input , int* retAddr) {
  retAddr[0] = (int)input;
}

template <> __device__ void getOrderedInt<float>( float input , int* retAddr) {
   int intVal = __float_as_int( input );
   if (intVal < 0) {
     intVal ^= 0x7FFFFFFF;
   }
   retAddr[0] = intVal;
}


template <typename IN_T>
int getOrderedIntHost(IN_T floatVal) {
  return (int)floatVal;
}

template <> int getOrderedIntHost<float>(float floatVal) {
   int intVal = *((int *) &floatVal);
   if (intVal < 0) {
     intVal ^= 0x7FFFFFFF;
   }
   return intVal;
}


template <typename IMAGE_T>
__global__ void init_max_array(int* max_values, long nGroups) {
  long idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (long i=idx; i<nGroups; i+=gridDim.x * blockDim.x) {
    getOrderedInt((int)-INT_MAX, &max_values[i]);
  }
}


template <typename GROUP_T, typename IMAGE_T>
__global__ void find_max_values(const GROUP_T* __restrict__ groups_d,
                                const IMAGE_T* __restrict__ image_d,
                                int* __restrict__ max_values,
                                const long nVoxels) {
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (long i=idx; i<nVoxels; i+=gridDim.x * blockDim.x) {
    const GROUP_T group_num = groups_d[i];
    int voxel_value;
    getOrderedInt(image_d[i], &voxel_value);
    atomicMax(&max_values[group_num], voxel_value);
  }
}


// When nGroups is less than or equal to the number of threads per block, we
// perform majority of updates to shared memory, and we then propagate 
// to global mem at the very end of the kernel.
template <typename GROUP_T, typename IMAGE_T>
__global__ void find_max_values_shm(const GROUP_T* __restrict__ groups_d,
                                    const IMAGE_T* __restrict__ image_d,
                                    int* __restrict__ max_values,
                                    const long nVoxels, 
                                    int initval, long nGroups) {
  extern __shared__ int max_vals_shm[];
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;

  // initialize shared memory contents
  //  getOrderedInt(-10000, &max_vals_shm[threadIdx.x]);
  max_vals_shm[threadIdx.x] = initval;

  // loop over all values updating shared memory
  for (long i=idx; i<nVoxels; i+=gridDim.x * blockDim.x) {
    const GROUP_T group_num = groups_d[i];
    int voxel_value;
    getOrderedInt(image_d[i], &voxel_value);
    atomicMax(&max_vals_shm[group_num], voxel_value);
  }
  __syncthreads();

  // propagate final values to to global memory
  if (threadIdx.x < nGroups) {
    atomicMax(&max_values[threadIdx.x], max_vals_shm[threadIdx.x]);
  }
}


template <typename GROUP_T, typename IMAGE_T>
__global__ void find_max_values_shm_2xl(const GROUP_T* __restrict__ groups_d,
                                        const IMAGE_T* __restrict__ image_d,
                                        int* __restrict__ max_values,
                                        const long nVoxels, 
                                        int initval, long nGroups) {
  extern __shared__ int max_vals_shm[];
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;

  // initialize shared memory contents
  int gdx = threadIdx.x;
  max_vals_shm[gdx] = initval;
  gdx = blockDim.x + threadIdx.x;
  max_vals_shm[gdx] = initval;

  // loop over all values updating shared memory
  for (long i=idx; i<nVoxels; i+=gridDim.x * blockDim.x) {
    const GROUP_T group_num = groups_d[i];
    int voxel_value;
    getOrderedInt(image_d[i], &voxel_value);
    atomicMax(&max_vals_shm[group_num], voxel_value);
  }
  __syncthreads();

  // propagate final values to to global memory
  gdx = threadIdx.x;
  if (gdx < nGroups) {
    atomicMax(&max_values[gdx], max_vals_shm[gdx]);
  }
  gdx = blockDim.x + threadIdx.x;
  if (gdx < nGroups) {
    atomicMax(&max_values[gdx], max_vals_shm[gdx]);
  }
}


template <typename GROUP_T, typename IMAGE_T>
__global__ void find_max_values_shm_4xl(const GROUP_T* __restrict__ groups_d,
                                        const IMAGE_T* __restrict__ image_d,
                                        int* __restrict__ max_values,
                                        const long nVoxels, 
                                        int initval, long nGroups) {
  extern __shared__ int max_vals_shm[];
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;

  // initialize shared memory contents
  int gdx = threadIdx.x;
  max_vals_shm[gdx] = initval;
  gdx = blockDim.x + threadIdx.x;
  max_vals_shm[gdx] = initval;
  gdx = 2*blockDim.x + threadIdx.x;
  max_vals_shm[gdx] = initval;
  gdx = 3*blockDim.x + threadIdx.x;
  max_vals_shm[gdx] = initval;

  // loop over all values updating shared memory
  for (long i=idx; i<nVoxels; i+=gridDim.x * blockDim.x) {
    const GROUP_T group_num = groups_d[i];
    int voxel_value;
    getOrderedInt(image_d[i], &voxel_value);
    atomicMax(&max_vals_shm[group_num], voxel_value);
  }
  __syncthreads();

  // propagate final values to to global memory
  gdx = threadIdx.x;
  if (gdx < nGroups) {
    atomicMax(&max_values[gdx], max_vals_shm[gdx]);
  }
  gdx = blockDim.x + threadIdx.x;
  if (gdx < nGroups) {
    atomicMax(&max_values[gdx], max_vals_shm[gdx]);
  }
  gdx = 2*blockDim.x + threadIdx.x;
  if (gdx < nGroups) {
    atomicMax(&max_values[gdx], max_vals_shm[gdx]);
  }
  gdx = 3*blockDim.x + threadIdx.x;
  if (gdx < nGroups) {
    atomicMax(&max_values[gdx], max_vals_shm[gdx]);
  }
}



template <typename GROUP_T, typename IMAGE_T>
__global__ void update_group_idx(const GROUP_T* __restrict__ groups_d,
                                 const IMAGE_T* __restrict__ image_d,
                                 const int* __restrict max_values,
                                 unsigned long* __restrict__ group_idx,
                                 const unsigned long nVoxels) {
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (long i=idx; i<nVoxels; i+=gridDim.x * blockDim.x) {
    const GROUP_T group_num = groups_d[i];
    int max_val;
    getOrderedInt(image_d[i], &max_val);
    if (max_values[group_num] == max_val) {
      group_idx[group_num] = i;
    }
  }
}

// XXX right now max_idx is only unsigned int, so if nVoxels > UINT_MAX then won't run on GPU
template <typename GROUP_T, typename IMAGE_T>
void find_groups_max_idx_cuda(GROUP_T* groups_d, IMAGE_T* image_d, unsigned long* max_idx, unsigned long nVoxels,
                              unsigned long nGroups, gpuseg_temp_storage *tmp) {
  int num_voxel_blocks = (nVoxels + VOXEL_BLOCK_SIZE-1) / VOXEL_BLOCK_SIZE;
  int num_group_blocks = (nGroups + GROUP_BLOCK_SIZE-1) / GROUP_BLOCK_SIZE;
  int *max_values;

  if (num_voxel_blocks > 65535)
    num_voxel_blocks = 65535;
  if (num_group_blocks > 65535)
    num_group_blocks = 65535;

  // if we've been provided with a tmp workspace handle, we use it,
  // otherwise we allocate and free tmp workspace on-the-fly
  if (tmp != NULL) {
    if (tmp->tmp_d == NULL || tmp->sz < nGroups * long(sizeof(int))) {
// printf("Reallocing GPU max values tmp buffer...\n");
      if (tmp->tmp_d != NULL)
        cudaFree(tmp->tmp_d);
      tmp->tmp_d = NULL;
      tmp->sz = 0;

      cudaMalloc(&tmp->tmp_d, nGroups * sizeof(int));
      tmp->sz = nGroups * sizeof(int);
    }
    max_values = static_cast<int *>(tmp->tmp_d);
  } else {
    // XXX this is useful for profiling examples for GTC18, but after that
    //     the non-persistent buffer code path should go away.
    cudaMalloc(&max_values, nGroups * sizeof(int));
  }

  CUERR // check and clear any existing errors 

  init_max_array<IMAGE_T><<<num_group_blocks, GROUP_BLOCK_SIZE>>>(max_values, nGroups);
  CUERR // check and clear any existing errors 

  // If we have compacted groups into a continguous range that fits 
  // within the range [0:VOXEL_BLOCK_SIZE], we can use a fast shared memory
  // based max value kernel
  int initval = getOrderedIntHost((int)-INT_MAX);
  if (nGroups <= VOXEL_BLOCK_SIZE) {
    find_max_values_shm<GROUP_T, IMAGE_T><<<num_voxel_blocks, VOXEL_BLOCK_SIZE, VOXEL_BLOCK_SIZE*sizeof(int)>>>(groups_d, image_d, max_values, nVoxels, initval, nGroups);
#if 1 
  } else if (nGroups <= 2*VOXEL_BLOCK_SIZE) {
    find_max_values_shm_2xl<GROUP_T, IMAGE_T><<<num_voxel_blocks, VOXEL_BLOCK_SIZE, 2*VOXEL_BLOCK_SIZE*sizeof(int)>>>(groups_d, image_d, max_values, nVoxels, initval, nGroups);
  } else if (nGroups <= 4*VOXEL_BLOCK_SIZE) {
    find_max_values_shm_4xl<GROUP_T, IMAGE_T><<<num_voxel_blocks, VOXEL_BLOCK_SIZE, 4*VOXEL_BLOCK_SIZE*sizeof(int)>>>(groups_d, image_d, max_values, nVoxels, initval, nGroups);
#endif
  } else {
    // slow kernel always drives atomic updates out to global memory
    find_max_values<GROUP_T, IMAGE_T><<<num_voxel_blocks, VOXEL_BLOCK_SIZE>>>(groups_d, image_d, max_values, nVoxels);
  }
  CUERR // check and clear any existing errors 

  update_group_idx<GROUP_T, IMAGE_T><<<num_voxel_blocks, VOXEL_BLOCK_SIZE>>>(groups_d, image_d, max_values, max_idx, nVoxels);
  CUERR // check and clear any existing errors 

  if (tmp == NULL) {
    cudaFree(max_values);
  } 
}


#define NUM_N 18

template <typename GROUP_T, typename IMAGE_T>
__global__ void hill_climb_kernel(IMAGE_T* image_d, unsigned long* max_int_d, GROUP_T* group_map_d, GROUP_T* groups_d,
                                  int height, int width, int depth, unsigned long nGroups) {
  __shared__ IMAGE_T neighbor_vals[NUM_N];
  int tIdx = threadIdx.x;
  if (tIdx >= NUM_N + 1) {
    return;
  }
  for (int g = blockIdx.x; g < nGroups; g += gridDim.x) {
    long curr_idx;
    long new_idx = max_int_d[g];
    do {
      int offsets[NUM_N + 1] = { 0, 1, -1, width, -width, height*width, -height*width, // 0 - 6
                                 1 + width, 1 - width, -1 + width, -1 - width,         // 7 - 10
                                 1 + height*width, 1 - height*width, -1 + height*width, -1 - height*width, // 11-14
                                 width + height*width, width - height*width, -width + height*width, -width - height*width}; // 15-18
      curr_idx = new_idx;
      int z = new_idx / (height * width);
      int r = new_idx % (height * width);
      int y = r / width;
      int x = r % width;

      if (x >= width - 1) {
        offsets[1] = 0;
        offsets[7] = 0;
        offsets[8] = 0;
        offsets[11] = 0;
        offsets[12] = 0;
      }

      if (x <= 0) {
        offsets[2] = 0;
        offsets[9] = 0;
        offsets[10] = 0;
        offsets[13] = 0;
        offsets[14] = 0;
      }

      if (y >= height - 1) {
        offsets[3] = 0;
        offsets[7] = 0;
        offsets[9] = 0;
        offsets[15] = 0;
        offsets[16] = 0;
      }

      if (y <= 0) {
        offsets[4] = 0;
        offsets[8] = 0;
        offsets[10] = 0;
        offsets[17] = 0;
        offsets[18] = 0;
      }

      if (z >= depth - 1) {
        offsets[5] = 0;
        offsets[11] = 0;
        offsets[13] = 0;
        offsets[15] = 0;
        offsets[17] = 0;
      }

      if (z <= 0) {
        offsets[6] = 0;
        offsets[12] = 0;
        offsets[14] = 0;
        offsets[16] = 0;
        offsets[18] = 0;
      }

      int offset = offsets[tIdx];
      neighbor_vals[tIdx] = image_d[curr_idx + offset];
      __syncthreads();
      IMAGE_T curr_val = neighbor_vals[0];
      int new_offset = 0;
      for (int i = 1; i <= NUM_N; ++i) {
        if (neighbor_vals[i] > curr_val) {
          curr_val = neighbor_vals[i];
          new_offset = offsets[i];
        }
      }
      new_idx += new_offset;
    } while (curr_idx != new_idx);

    if (tIdx == 0) {
      group_map_d[g] = groups_d[new_idx];
    }
  }
}


template <typename GROUP_T, typename IMAGE_T>
void hill_climb_merge_cuda(GROUP_T* groups_d, IMAGE_T* image_d, unsigned long* max_idx_d, GROUP_T* group_map_d,
                             int height, int width, int depth, unsigned long nGroups) {
  long nVoxels = long(height) * long(width) * long(depth);
  int num_voxel_blocks = (nVoxels + VOXEL_BLOCK_SIZE-1) / VOXEL_BLOCK_SIZE;
  int nBlocks = nGroups > 65535 ? 65535 : nGroups;

  CUERR

  hill_climb_kernel<<<nBlocks, 32>>>(image_d, max_idx_d, group_map_d, groups_d,
                               height, width, depth, nGroups);

  CUERR
  update_groups_from_map<<<num_voxel_blocks, VOXEL_BLOCK_SIZE>>>(groups_d, group_map_d, nVoxels);

  CUERR
}

template <typename GROUP_T, typename IMAGE_T>
void watershed_hill_climb_merge_cuda(GROUP_T* segments_d, GROUP_T* new_segments_d, IMAGE_T* image_d, GROUP_T* group_map_d,
                                     unsigned long* max_idx_d, long height, long width, long depth, unsigned long nGroups) {
  long nVoxels = long(height) * long(width) * long(depth);
  int num_voxel_blocks = (nVoxels + VOXEL_BLOCK_SIZE-1) / VOXEL_BLOCK_SIZE;
  int nBlocks = nGroups > 65535 ? 65535 : nGroups;

  CUERR

  hill_climb_kernel<<<nBlocks, 32>>>(image_d, max_idx_d, group_map_d, new_segments_d,
                               height, width, depth, nGroups);

  CUERR
  update_groups_from_map<<<num_voxel_blocks, VOXEL_BLOCK_SIZE>>>(segments_d, group_map_d, nVoxels);

  CUERR
}

template <typename IN_T, typename OUT_T>
__global__ void copy_and_convert_kernel(IN_T* in, OUT_T* out, long num_elements) {
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (long i=idx; i<num_elements; i+=gridDim.x * blockDim.x) {
    out[i] = (OUT_T)in[i];
  }
}

template <typename IN_T, typename OUT_T>
void copy_and_convert_type_cuda(IN_T* in, OUT_T* out, long num_elements) {
  int nBlocks = (num_elements + VOXEL_BLOCK_SIZE-1) / VOXEL_BLOCK_SIZE;
  nBlocks = nBlocks > 65535 ? 65535 : nBlocks;

  copy_and_convert_kernel<<<nBlocks, 32>>>(in, out, num_elements);
}

#define COPY_CONV_INST(T) \
  template void copy_and_convert_type_cuda<T,unsigned long>(T* in, unsigned long* out, long num_elements);\
  template void copy_and_convert_type_cuda<T,unsigned int>(T* in, unsigned int* out, long num_elements);\
  template void copy_and_convert_type_cuda<T,unsigned short>(T* in, unsigned short* out, long num_elements);\
  template void copy_and_convert_type_cuda<T,unsigned char>(T* in, unsigned char* out, long num_elements);\
  template void copy_and_convert_type_cuda<T,long>(T* in, long* out, long num_elements);\
  template void copy_and_convert_type_cuda<T,int>(T* in, int* out, long num_elements);\
  template void copy_and_convert_type_cuda<T,short>(T* in, short* out, long num_elements);\
  template void copy_and_convert_type_cuda<T,char>(T* in, char* out, long num_elements);\
  template void copy_and_convert_type_cuda<T,float>(T* in, float* out, long num_elements);\

#define INST_SEQ_GROUPS_CUDA(G_T) \
template long sequentialize_groups_cuda<G_T>(G_T* groups_d, G_T* group_map_d,\
                                         unsigned long nVoxels, unsigned long nGroups,\
                                         gpuseg_temp_storage *tmp,\
                                         gpuseg_temp_storage *scanwork);\

#define INST_FIND_MAX_IDX_CUDA(G_T) \
template void find_groups_max_idx_cuda<G_T, float>(G_T* groups_d, float* image_d, unsigned long* max_idx,\
                                                   unsigned long nVoxels, unsigned long nGroups,\
                                                   gpuseg_temp_storage *tmp);\
template void find_groups_max_idx_cuda<G_T, unsigned short>(G_T* groups_d, unsigned short* image_d, unsigned long* max_idx,\
                                                   unsigned long nVoxels, unsigned long nGroups,\
                                                   gpuseg_temp_storage *tmp);\
template void find_groups_max_idx_cuda<G_T, unsigned char>(G_T* groups_d, unsigned char* image_d, unsigned long* max_idx,\
                                                   unsigned long nVoxels, unsigned long nGroups,\
                                                   gpuseg_temp_storage *tmp);

#define INST_HILL_CLIMB_MERGE_CUDA(G_T) \
template void hill_climb_merge_cuda<G_T, float>(G_T* groups_d, float* image_d, unsigned long* max_idx_d,\
                          G_T* group_map_d, int height, int width, int depth, unsigned long nGroups);\
template void hill_climb_merge_cuda<G_T, unsigned short>(G_T* groups_d, unsigned short* image_d, unsigned long* max_idx_d,\
                          G_T* group_map_d, int height, int width, int depth, unsigned long nGroups);\
template void hill_climb_merge_cuda<G_T, unsigned char>(G_T* groups_d, unsigned char* image_d, unsigned long* max_idx_d,\
                          G_T* group_map_d, int height, int width, int depth, unsigned long nGroups);

#define INST_WATERSHED_HILL_CLIMB_MERGE_CUDA(G_T) \
template void watershed_hill_climb_merge_cuda<G_T, float>(G_T* segments_d, G_T* new_segments_d, float* image_d, G_T* group_map_d,\
                                     unsigned long* max_idx_d, long height, long width, long depth, unsigned long nGroups);\
template void watershed_hill_climb_merge_cuda<G_T, unsigned short>(G_T* segments_d, G_T* new_segments_d,\
                                     unsigned short* image_d, G_T* group_map_d,\
                                     unsigned long* max_idx_d, long height, long width, long depth, unsigned long nGroups);\
template void watershed_hill_climb_merge_cuda<G_T, unsigned char>(G_T* segments_d, G_T* new_segments_d,\
                                     unsigned char* image_d, G_T* group_map_d,\
                                     unsigned long* max_idx_d, long height, long width, long depth, unsigned long nGroups);


COPY_CONV_INST(long)
COPY_CONV_INST(unsigned long)
COPY_CONV_INST(int)
COPY_CONV_INST(unsigned int)
COPY_CONV_INST(short)
COPY_CONV_INST(unsigned short)
COPY_CONV_INST(char)
COPY_CONV_INST(unsigned char)
COPY_CONV_INST(float)


INST_SEQ_GROUPS_CUDA(unsigned long)
INST_SEQ_GROUPS_CUDA(unsigned int)
INST_SEQ_GROUPS_CUDA(unsigned short)

INST_FIND_MAX_IDX_CUDA(unsigned long)
INST_FIND_MAX_IDX_CUDA(unsigned int)
INST_FIND_MAX_IDX_CUDA(unsigned short)


INST_HILL_CLIMB_MERGE_CUDA(unsigned long)
INST_HILL_CLIMB_MERGE_CUDA(unsigned int)
INST_HILL_CLIMB_MERGE_CUDA(unsigned short)

INST_WATERSHED_HILL_CLIMB_MERGE_CUDA(unsigned long)
INST_WATERSHED_HILL_CLIMB_MERGE_CUDA(unsigned int)
INST_WATERSHED_HILL_CLIMB_MERGE_CUDA(unsigned short)
