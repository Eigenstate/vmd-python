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
 *      $RCSfile: Watershed.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.36 $        $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Watershed image segmentation
 ***************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <climits>
#include "WKFUtils.h"
#include "ProfileHooks.h"

#define WATERSHED_INTERNAL 1
#include "Watershed.h"
#if defined(VMDCUDA)
#include "CUDAWatershed.h"
#endif

#define UPDATE_VOXEL(and_value, idx, curr_group, smallest_value, smallest_offset) {\
  if (n_eq & and_value) {\
    const int idx_offset = idx + and_value##_offset;\
    if (current_state.value[idx_offset] + FLOAT_DIFF < smallest_value) {\
      smallest_value = current_state.value[idx_offset];\
      offset_number = and_value##_idx;\
      smallest_offset = and_value##_offset;\
    }\
    curr_group = current_state.group[idx_offset] < curr_group ?\
    current_state.group[idx_offset] : curr_group;\
  }}

#define CALCULATE_NEIGHBORS(offset_str) {\
  const int idx_offset = idx + offset_str##_offset;\
  slope = curr_intensity - image[idx_offset];\
  if (slope < smallest_slope) {\
    smallest_slope = slope;\
    curr_lower = offset_str##_idx;\
  } else if (slope >= -FLOAT_DIFF && slope <= FLOAT_DIFF) {\
    curr_n_eq |= offset_str;\
    if (idx_offset < min_group) {\
      min_group = idx_offset;\
    }\
  } }


////// Constructor and desctructor
template <typename GROUP_T, typename IMAGE_T>
Watershed<GROUP_T, IMAGE_T>::Watershed(unsigned int h, unsigned int w, unsigned int d, bool cuda) {
  height = h;
  width = w;
  depth = d;
  heightWidth = height * width;
  nVoxels = long(heightWidth) * long(depth);

#ifdef VMDCUDA
  use_cuda = cuda; //this defaults to true unless overridden
#else
  use_cuda = false;
#endif
  memset(&gpu_state, 0, sizeof(gpu_state));

  current_state.group = new GROUP_T[nVoxels];
  current_state.value = new IMAGE_T[nVoxels];
  next_state.group = new GROUP_T[nVoxels];
  next_state.value = new IMAGE_T[nVoxels];
  equal_and_lower = new int[nVoxels];
  current_update = NULL;
  next_update = NULL;

  // If using block updates then allocate the current/next_update arrays
#ifdef BLOCK_UPDATES
  update_width = ((width+UPDATE_SIZE) / UPDATE_SIZE) * UPDATE_SIZE;
  int buf_size = (depth * height * update_width) >> UPDATE_SHIFT;
  update_offset = 0;
#ifdef FAST_UPDATES
  update_offset = update_width * height;
#endif // FAST_UPDATES
  current_update = new unsigned char[buf_size + 2L * update_offset];
  next_update = new unsigned char[buf_size + 2L * update_offset];
  memset(current_update, 1, (2L*update_offset + buf_size) * sizeof(unsigned char));
  memset(next_update, 0, (2L*update_offset + buf_size) * sizeof(unsigned char));
#endif //BLOCK_UPDATES

  //memcpy(intensity, data, nVoxels * sizeof(float));
}


template <typename GROUP_T, typename IMAGE_T>
Watershed<GROUP_T, IMAGE_T>::~Watershed() {
#if defined(VMDCUDA)
  destroy_gpu(gpu_state);
#endif
  if (current_state.group != NULL) {
    delete [] current_state.group;
    delete [] current_state.value;
    current_state.group = NULL;
  }
  if (next_state.group != NULL) {
    delete [] next_state.group;
    delete [] next_state.value;
    next_state.group = NULL;
  }
  if (equal_and_lower == NULL) {
    delete [] equal_and_lower;
    equal_and_lower = NULL;
  }
  if (current_update != NULL) {
    delete [] current_update;
    current_update = NULL;
  }
  if (next_update != NULL) {
    delete [] next_update;
    next_update = NULL;
  }
}


template <typename GROUP_T, typename IMAGE_T>
void Watershed<GROUP_T, IMAGE_T>::watershed(IMAGE_T* image, int imageongpu, GROUP_T* segments, bool verbose) { 
  if (imageongpu)
    PROFILE_PUSH_RANGE("Watershed::watershed(Image on GPU)", 0);
  else 
    PROFILE_PUSH_RANGE("Watershed::watershed(Image on host)", 0);

  // XXX we need to change this launch so that we always use the
  // GPU unless forbidden, and that we fallback to the CPU if the GPU
  // is forbidden, or we ran out of GPU memory, or an error occured
  // during execution on the GPU, so the calculation always proceeds.
#ifdef TIMER
  wkf_timerhandle init_timer = wkf_timer_create();    
  wkf_timerhandle total_timer = wkf_timer_create();    
  wkf_timerhandle update_timer = wkf_timer_create();    
  wkf_timer_start(total_timer);
  wkf_timer_start(init_timer);
#endif // TIMER

#if defined(VMDCUDA)
  if (use_cuda) {
    init_gpu_on_device(gpu_state, image, imageongpu, width, height, depth);
  } else 
#endif
  {
    // XXX deal with imageongpu if necessary, or eliminate possibility
    init(image);
  }

#ifdef TIMER
  wkf_timer_stop(init_timer);
  wkf_timer_start(update_timer);
#endif // TIMER

#if defined(VMDCUDA)
  if (use_cuda) {
    watershed_gpu(segments);
  } else 
#endif
    watershed_cpu(segments);

#ifdef TIMER
  wkf_timer_stop(update_timer);
  wkf_timer_stop(total_timer);
  double update_time_sec = wkf_timer_time(update_timer);
  double total_time_sec = wkf_timer_time(total_timer);
  double init_time_sec = wkf_timer_time(init_timer);
  wkf_timer_destroy(init_timer);
  wkf_timer_destroy(update_timer);
  wkf_timer_destroy(total_timer);
  if (verbose) {
    printf("Watershed init:   %f seconds\n", init_time_sec);
    printf("Watershed update: %f seconds\n", update_time_sec);
    printf("Watershed total:  %f seconds\n", total_time_sec);
  }
#endif

  PROFILE_POP_RANGE();
}


#if defined(VMDCUDA)

template <typename GROUP_T, typename IMAGE_T>
void Watershed<GROUP_T, IMAGE_T>::watershed_gpu(GROUP_T* segments_d) {
  PROFILE_PUSH_RANGE("Watershed::watershed_gpu()", 0);

  // XXX we may want to unify the timing approach between
  // both the CPU and GPU code paths at some point.  We can
  // also change this to do runtime tests for verbose timing output,
  // as is done in many other cases in VMD.
  update_cuda(gpu_state, segments_d);

  PROFILE_POP_RANGE();
}

#endif


template <typename GROUP_T, typename IMAGE_T>
void Watershed<GROUP_T, IMAGE_T>::watershed_cpu(GROUP_T* segments) {
  PROFILE_PUSH_RANGE("Watershed::watershed_cpu()", 1);

  unsigned int changes;
#ifdef STATS
  unsigned int num_updates;
  unsigned int num_block_updates;
  unsigned long int total_updates = 0;
  unsigned long int total_blocks = 0;
  unsigned long int total_block_updates = 0;
  unsigned int numRounds = 0;
#endif // STATS

  // keep performing update step of watershed algorithm until there are no changes
  do {
#ifdef STATS
    changes = update_blocks_stats(num_updates, num_block_updates);
    total_updates += num_updates;
    ++numRounds;
    total_block_updates += num_block_updates;
    total_blocks += height * depth * (width / UPDATE_SIZE);
#else // NOT STATS

#ifdef BLOCK_UPDATES
    changes = update_blocks();
#else // NOT BLOCK_UPDATES
    changes = update();
#endif //BLOCK_UPDATES

#endif // STATS
    SWAP(current_state.group, next_state.group, GROUP_T*);
    SWAP(current_state.value, next_state.value, IMAGE_T*);
  } while (changes);

#ifdef STATS
  printf("Total_updates: %llu\n", total_updates);
  printf("Total_voxels: %llu\n", nVoxels);
  printf("Total_block_updates: %llu\n", total_block_updates);
  printf("Total_blocks: %llu\n", total_blocks);
  printf("Number of watershed rounds: %d\n", numRounds);
  double update_percent = 100.0 * total_updates / (double)(numRounds * nVoxels);
  double block_percent = 100.0 * total_block_updates / (double)total_blocks;
  printf("Block update percentage: %f %% \n", block_percent);
  printf("Update percentage: %f %% \n", update_percent);
#endif //STATS

  memcpy(segments, current_state.group, nVoxels * sizeof(GROUP_T));

  PROFILE_POP_RANGE();
}


template <typename GROUP_T, typename IMAGE_T>
void Watershed<GROUP_T, IMAGE_T>::init_neighbor_offsets() {
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
  memcpy(neighbor_offsets, neighbor_offsets_t, sizeof(neighbor_offsets_t));
}

// XXX TODO Description of scheme to incrementally initialize watershed:
// create an init function with signature: partial_init(IMAGE_T* image, long start_idx, long stop_idx, long startGroupNum)
// Call the function multiple times looping the start_idx - stop_idx over the image, where startGroupNum is
// the number of groups that have been initialized so far.
// The function is similar to the init() function below, except long min_group = idx + startGroupNum
// After each time the function runs, we should call a modified sequentialize groups that only acts on indexes 0-stop_idx.
// This scheme will work as described assuming a sequential CPU initialization because the only dependency is on
// group numbers < the current group, so if those are already initialized there is no problem.
//
// One way to make this scheme work in parallel is to save an offset number (look at the min_group var for an example).
// We would save an offset number in one kernel, and then immediately run a second kernel that 'chases' these offset numbers
// until it finds a 0, and sets the group == to the group number at that index.
//

template <typename GROUP_T, typename IMAGE_T>
void Watershed<GROUP_T, IMAGE_T>::init(IMAGE_T* image) {
  // XXX the initialization method accounts for about 50%
  //     of the total runtime for a CUDA-accelerated segmentation 
  PROFILE_PUSH_RANGE("Watershed::init()", 1);


  init_neighbor_offsets();
  memset(equal_and_lower, 0, nVoxels * sizeof(int));
#ifdef BLOCK_UPDATES
  if (!use_cuda) {
    unsigned int buf_size = (depth * height * update_width) >> UPDATE_SHIFT;
    // 1 update offset on each end of update buffer
    memset(current_update, 1, (2*update_offset + buf_size) * sizeof(unsigned char));
    memset(next_update, 0, (2*update_offset + buf_size) * sizeof(unsigned char));
  }
#endif // BLOCK_UPDATES


#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      unsigned long yz = long(heightWidth) * z + long(width)*y;
      for (int x = 0; x < width; ++x) {
        float slope;
        float smallest_slope = -FLOAT_DIFF;
        long idx = yz + x;
        IMAGE_T curr_intensity = image[idx];
        long min_group = idx; // Group number = idx to start
        int curr_n_eq = 0;
        long curr_lower = 0;
        //TODO change to be actual offset instead of index into offset array

#ifdef CONNECTED_18
        if (x > 0) {
          CALCULATE_NEIGHBORS(nx);
        }
        if (x < width-1) {
          CALCULATE_NEIGHBORS(px);
        }

        /* Calculate n_lower and n_eq */
        if (y > 0) {
          CALCULATE_NEIGHBORS(ny);
          if (x < width - 1) {
            CALCULATE_NEIGHBORS(px_ny);
          }
        }
        if (y < height-1) {
          CALCULATE_NEIGHBORS(py);
          if (x < width - 1) {
            CALCULATE_NEIGHBORS(px_py);
          }
        }

        if (z > 0) {
          CALCULATE_NEIGHBORS(nz);
          if (x < width - 1) {
            CALCULATE_NEIGHBORS(px_nz);
          }
          if (x > 0) {
            CALCULATE_NEIGHBORS(nx_nz);
          }
          if (y > 0 ) {
            CALCULATE_NEIGHBORS(ny_nz);
          }
          if (y < height - 1) {
            CALCULATE_NEIGHBORS(py_nz);
          }
        }
        if (z < depth-1) {
          CALCULATE_NEIGHBORS(pz);
          if (x < width - 1) {
            CALCULATE_NEIGHBORS(px_pz);
          }
          if (x > 0) {
            CALCULATE_NEIGHBORS(nx_pz);
          }
          if (y < height - 1) {
            CALCULATE_NEIGHBORS(py_pz);
          }
          if (y > 0) {
            CALCULATE_NEIGHBORS(ny_pz);
          }
        }
#else // 6 connected neighbors
        if (x > 0) {
          CALCULATE_NEIGHBORS(nx);
        }
        if (x < width-1) {
          CALCULATE_NEIGHBORS(px);
        }

        /* Calculate n_lower and n_eq */
        if (y > 0) {
          CALCULATE_NEIGHBORS(ny);
        }
        if (y < height-1) {
          CALCULATE_NEIGHBORS(py);
        }

        if (z > 0) {
          CALCULATE_NEIGHBORS(nz);
        }
        if (z < depth-1) {
          CALCULATE_NEIGHBORS(pz);
        }
#endif // CONNECTED_18

        /* Update current voxel */
        equal_and_lower[idx] = MAKE_EQUAL_AND_LOWER(curr_n_eq, curr_lower);
        if (curr_lower == 0) {
          /* Minimum */
          next_state.group[idx] = min_group;
          next_state.value[idx] = image[idx];
        } else {
          /* Not minimum */
          const long nIdx = idx + neighbor_offsets[curr_lower];
          next_state.value[idx] = image[nIdx];
          next_state.group[idx] = nIdx;
        }
      }
    }
  }

#if defined(VMDCUDA)
  if (use_cuda) {
    use_cuda = init_gpu(next_state, equal_and_lower, gpu_state, width, height, depth);
    if (!use_cuda)
      printf("Warning: Could not initialize GPU. Falling back to host.\n");
  } 
#endif
  if (!use_cuda){
    memcpy(current_state.group, next_state.group, nVoxels * sizeof(GROUP_T));
    memcpy(current_state.value, next_state.value, nVoxels * sizeof(IMAGE_T));
  }

  PROFILE_POP_RANGE();
}


template <typename GROUP_T, typename IMAGE_T>
unsigned int Watershed<GROUP_T, IMAGE_T>::update_blocks() {
  unsigned int changes = 0;
  const unsigned long update_heightWidth = height * update_width;
  const unsigned long width_shift = update_width >> UPDATE_SHIFT;
  const unsigned long heightWidth_shift = update_heightWidth >> UPDATE_SHIFT;
#ifdef USE_OMP
#pragma omp parallel for schedule(guided)
#endif
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      unsigned int local_changes = 0;
      const unsigned int update_yz = z * update_heightWidth + y * update_width + update_offset;
      const unsigned long yz = z * long(heightWidth) + y * long(width);
      for (int update_x = 0; update_x < width; update_x += UPDATE_SIZE) {

        const unsigned long update_idx = (update_yz + update_x) >> UPDATE_SHIFT;
        // only need to look at current block if it was marked as needing an update in prev. round
        if (current_update[update_idx]) {
          bool update = false;
          const int x_max = update_x + UPDATE_SIZE < width ? update_x + UPDATE_SIZE : width;
          for (int x = update_x; x < x_max; ++x) {
            const unsigned long idx = yz + x;
            const int curr_eq_lower = equal_and_lower[idx]; // packed equal Ns and lower Ns
            const int curr_offset_number = GET_N_LOWER(curr_eq_lower);

            if (curr_offset_number) {
              /* Not minimum */
              const unsigned int nidx = idx + neighbor_offsets[curr_offset_number];
              if ((next_state.group[idx] != current_state.group[nidx]) ||
                   next_state.value[idx] != current_state.value[nidx]) {
                next_state.group[idx] = current_state.group[nidx];
                next_state.value[idx] = current_state.value[nidx];
                update = true;
              }
            } else {
              /* Minimum */
              unsigned int n_eq = GET_N_EQ(curr_eq_lower);
              IMAGE_T smallest_value = current_state.value[idx];
              GROUP_T curr_group = current_state.group[idx];
              int smallest_offset = 0;
              int offset_number = 0;

#ifdef CONNECTED_18
              /* +- x */
              UPDATE_VOXEL(nx, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(px, idx, curr_group, smallest_value, smallest_offset);

              /* +- x, -y*/
              UPDATE_VOXEL(nx_ny, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(   ny, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(px_ny, idx, curr_group, smallest_value, smallest_offset);

              /* +- x, +y*/
              UPDATE_VOXEL(nx_py, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(   py, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(px_py, idx, curr_group, smallest_value, smallest_offset);


              /* +- x, +z*/
              UPDATE_VOXEL(px_pz, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(nx_pz, idx, curr_group, smallest_value, smallest_offset);
              /* +- y, +z*/
              UPDATE_VOXEL(py_pz, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(   pz, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(ny_pz, idx, curr_group, smallest_value, smallest_offset);

              /* +- x, -z*/
              UPDATE_VOXEL(nx_nz, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(   nz, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(px_nz, idx, curr_group, smallest_value, smallest_offset);
              /* +- y, -z*/
              UPDATE_VOXEL(py_nz, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(ny_nz, idx, curr_group, smallest_value, smallest_offset);

#else
              UPDATE_VOXEL(nx, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(px, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(ny, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(py, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(nz, idx, curr_group, smallest_value, smallest_offset);
              UPDATE_VOXEL(pz, idx, curr_group, smallest_value, smallest_offset);
#endif

              /* 
               * curr_group is now the minimum group of all the
               * equal neighbors and current group
               */
              if (smallest_offset != 0) {
                equal_and_lower[idx] = MAKE_EQUAL_AND_LOWER(n_eq, offset_number);
                next_state.value[idx] = smallest_value;
                next_state.group[idx] = current_state.group[idx + smallest_offset];
                update = true;
              } else if (curr_group != next_state.group[idx]) {
                next_state.group[idx] = curr_group;
                update = true;
              }
            }
          }
          current_update[update_idx] = 0;
          if (update) {
            local_changes = 1;
            next_update[update_idx] = 1;
#ifdef FAST_UPDATES
            next_update[update_idx - 1] = 1;
            next_update[update_idx + 1] = 1;
            next_update[update_idx - width_shift] = 1;
            next_update[update_idx + width_shift] = 1;
            next_update[update_idx - heightWidth_shift] = 1;
            next_update[update_idx + heightWidth_shift] = 1;
#else
            if (update_x > 0) {
              next_update[update_idx - 1] = 1;
            }
            if (update_x < update_width-UPDATE_SIZE) {
              next_update[update_idx + 1] = 1;
            }
            if (y > 0) {
              next_update[update_idx - width_shift] = 1;
            }
            if (y < height - 1) {
              next_update[update_idx + width_shift] = 1;
            }
            if (z > 0) {
              next_update[update_idx - heightWidth_shift] = 1;
            }
            if (z < depth-1) {
              next_update[update_idx + heightWidth_shift] = 1;
            }
#endif // FAST_UPDATES
          }
        }
      }
      if (local_changes) {
        changes = 1;
      }
    }
  }
  SWAP(current_update, next_update, unsigned char*);
  return changes;
}


template <typename GROUP_T, typename IMAGE_T>
unsigned int Watershed<GROUP_T, IMAGE_T>::update() {
  unsigned int changes = 0;
#ifdef USE_OMP
#pragma omp parallel for schedule(guided)
#endif
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      const unsigned long yz = z * long(heightWidth) + y * long(width);
      bool update = false;
      for (int x = 0; x < width; ++x) {
        const unsigned int idx = yz + x;
        const int curr_eq_lower = equal_and_lower[idx];
        const int curr_offset_number = GET_N_LOWER(curr_eq_lower);

        if (curr_offset_number) {
          /* Not minimum */
          const unsigned int nidx = idx + neighbor_offsets[curr_offset_number];
          if (next_state.group[idx] != current_state.group[nidx]) {
            next_state.group[idx] = current_state.group[nidx];
            next_state.value[idx] = current_state.value[nidx];
            update = true;
          }
        } else {
          /* Minimum */
          int n_eq = GET_N_EQ(curr_eq_lower);
          IMAGE_T smallest_value = current_state.value[idx];
          GROUP_T curr_group = current_state.group[idx];
          int smallest_offset = 0;
          int offset_number = 0;

#ifdef CONNECTED_18
          /* +- x */
          UPDATE_VOXEL(nx, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(px, idx, curr_group, smallest_value, smallest_offset);

          /* +- x, -y*/
          UPDATE_VOXEL(nx_ny, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(   ny, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(px_ny, idx, curr_group, smallest_value, smallest_offset);

          /* +- x, +y*/
          UPDATE_VOXEL(nx_py, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(   py, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(px_py, idx, curr_group, smallest_value, smallest_offset);


          /* +- x, +z*/
          UPDATE_VOXEL(px_pz, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(nx_pz, idx, curr_group, smallest_value, smallest_offset);
          /* +- y, +z*/
          UPDATE_VOXEL(py_pz, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(   pz, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(ny_pz, idx, curr_group, smallest_value, smallest_offset);

          /* +- x, -z*/
          UPDATE_VOXEL(nx_nz, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(   nz, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(px_nz, idx, curr_group, smallest_value, smallest_offset);
          /* +- y, -z*/
          UPDATE_VOXEL(py_nz, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(ny_nz, idx, curr_group, smallest_value, smallest_offset);

#else // 6 connected neighbors
          UPDATE_VOXEL(nx, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(px, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(ny, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(py, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(nz, idx, curr_group, smallest_value, smallest_offset);
          UPDATE_VOXEL(pz, idx, curr_group, smallest_value, smallest_offset);
#endif // CONNECTED_18

          /* 
           * curr_group is now the minimum group of all the
           * equal neighbors and current group
           */
          if (smallest_offset) {
            equal_and_lower[idx] = MAKE_EQUAL_AND_LOWER(n_eq, offset_number);
            next_state.value[idx] = smallest_value;
            next_state.group[idx] = current_state.group[idx + smallest_offset];
            update = true;
          } else if (curr_group != next_state.group[idx]) {
            next_state.group[idx] = curr_group;
            update = true;
          }
        }
      }
      if (update) {
        changes = 1;
      }
    }
  }
  return changes;
}

#define INST_WATERSHED(G_T) \
template class Watershed<G_T, float>;\
template class Watershed<G_T, unsigned short>;\
template class Watershed<G_T, unsigned char>;

INST_WATERSHED(unsigned long)
INST_WATERSHED(unsigned int)
INST_WATERSHED(unsigned short)
